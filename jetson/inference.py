import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
from pycuda import autoinit
import cv2
import time


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtYOLO:
    """Class for TensorRT inference on Jetson Nano
    """
    def __init__(self, engine_path: str, input_shape: list, conf=0.25, iou=0.45):
        """Class for loading .engine file with YOLOv8 and performing
        TensorRT inference on Jetson Nano

        Parameters
        ----------
        engine_path : str
            Path of engine file
        input_shape : tuple
            Model's input shape (width, height)
        conf : float
            Confidence YOLO threshold, by default 0.25
        iou : float
            IoU threshold for NMS algo, by default 0.45
        """
        self.engine_path = engine_path
        self.model_name = engine_path.split('/')[-1]
        self.dtype = np.float32
        self.conf = conf
        self.iou = iou
        self.input_shape = tuple(input_shape)
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        # load trt file
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = 1
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

     
    @staticmethod
    def load_engine(trt_runtime, engine_path):
        """Load engine trt file
        """
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    

    def allocate_buffers(self):
        """Allocate memory for inputs and outputs
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
       
    
    def  __nms(self, boxes: np.ndarray) -> np.ndarray:
        """Perform postprocess and NMS on output boxes

        Parameters
        ----------
        boxes : np.ndarray
            All boxes from model output

        Returns
        -------
        np.ndarray or None
            Filtered boxes
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        probs = boxes[:,4:]
        conf = np.max(probs, axis=1)
        classes = np.argmax(probs, axis=1)
        boxes = boxes[:,:6]
        boxes[:, 4] = conf
        boxes[:, 5] = classes
        areas = w * h  # compute areas of boxes
        ordered = conf.argsort()[::-1]  # get sorted indexes of scores in descending order
        keep = []  # boxes to keep
        while ordered.size > 0:
            # Index of the current element:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[ordered[1:]])
            yy1 = np.maximum(y[i], y[ordered[1:]])
            xx2 = np.minimum(x[i] + w[i], x[ordered[1:]] + w[ordered[1:]])
            yy2 = np.minimum(y[i] + h[i], y[ordered[1:]] + h[ordered[1:]])

            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where(iou <= self.iou)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        if len(keep) == 0:
            return ()
        boxes = boxes[keep]
        return boxes


    def __call__(self, x:np.ndarray) -> np.ndarray:
        """Perform object detection with YOLOv8 on given frame
        
        Parameters
        ----------
        x : np.ndarray
            Input image

        Returns
        -------
        np.ndarray or None
            Filtered boxes with (x, y, w, h, conf)
        """
        # preprocess
        ratio = (x.shape[1] / self.input_shape[0], x.shape[0] / self.input_shape[1])
        ratio = np.array(ratio)
        x = cv2.resize(x, self.input_shape)
        x = x[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())
        # transfer data on cuda device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        # run computations
        self.context.execute_async(batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)
        # get data from cuda device
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        self.stream.synchronize()
        boxes = self.outputs[0].host.reshape(14, -1).T  # TODO: change to n_classes + 4
        f = boxes[:, 4:] > self.conf
        boxes = boxes[np.any(f, axis=1), :]
        boxes = self.__nms(boxes)  # nms boxes
        if len(boxes) > 0:
            boxes[:, :2] = boxes[:, :2] * ratio  # scale boxes
            boxes[:, 2:4] = boxes[:, 2:4] * ratio
        return boxes
    

    def release(self) -> None:
        """Free allocated resources
        """
        del self.outputs
        del self.inputs
        del self.stream
        del self.bindings
        
        
    def draw_bboxes(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Draw red bounding boxes on given frame
        
        Parameters
        ----------
        frame : np.ndarray
            Input frame
        boxes : np.ndarray
            Bounding boxes from YOLO detection

        Returns
        -------
        np.ndarray
            Frame with boundung boxes
        """
        # convert from xywh to xyxy
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
        boxes[:, 2:4] = boxes[:, :2] + boxes[:, 2:4]
        for box in boxes:
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])), (0, 0, 255), 2)
            frame = cv2.putText(frame, 'PERSON {:.2f}'.format(box[4]), (int(box[0]), int(box[1])),
                                cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        return frame
        
        
if __name__ == "__main__":
    image = cv2.imread('test.jpg')
    yolo = TrtYOLO('baseline.engine', (640, 384))
    fps = 0.0
    tic = time.time()
    while True:
        boxes = yolo(image.copy())
        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc
        print("FPS:", fps)
