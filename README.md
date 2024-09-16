# YOLOv8-for-small-objects
This repository contains implementation for Dmitrii I. Krasnov, Sergey N. Yarishev, Victoria A. Ryzhova,  Todor S. Djamiykov paper **"Improved YOLOv8 Network for Small Objects Detection"** (link to publication is coming soon). Repo is a modification of original [Ultralytics](https://github.com/ultralytics/ultralytics) so `ultralytics` folder remains unchanged (almost).

## Abstract
This project is an implementation of the experimantal adaptation pipeline for object detection and semantic segmentation models. 
This pipeline specifically designed to improve model's architecture in order to adapt it to small objects.
`model_configs` contains set of .yaml configs for model building that were used in the experiment. `jetson` contains scripts for converting .onnx model (can be exported in default ultralytics way) to TensorRT egnine file and inference on Jetson Nano. AGPL-3.0 license is used since my project includes original ultralytics repo with this license type.

## Steps

### P2 detection head

P2 detection head and assosiated tensors pathes (W / 4, H / 4 of image resolution) added to the original network in order to increase number of small-sized bboxes detected. This step increases overall number of bounding boxes and slows down inference and post-process. However, resulting speed is enough for most of the practical applications (see benchmarks).

### P5 detection head

Simply remove it! P2 detection head and assosiated tensors responsible for large objects detection which is redundant for current task. Evaluation results shows, that this step is compensated with P2 detection head so there is no losses. Moreover this step helps to reduce overall number of model's paremeters by 46.8%.

### Bi-directional feature fusion

Concat layer in original YOLOv8 model was replaced with Bi-directrional concat. This block use feature maps with both higher and lower resolutions to fuse them with feature map of current resolution (see figure). This step allows simultaneous use of the deep and shallow features.

![Bi-directional feature fusion](/assets/bi_directional_fusion.png)

### CBAM attention module

A main building block of YOLOv8 C2f was enhanced with CBAM attention module. This module is a sequence of the channel attention block and spatial attention block. Attention module is necessary for supressing less informative channels and areas on feature maps. Input and output shapes of attention module are equal which makes it possible to place this module everywhere in the network's architecture.

![CBAM](/assets/cbam.png)

### Final

Final improved YOLOv8 architecture is shown below. Numbers of channels are actual for nano scale.

![Final](/assets/arch.png)

Predictions example on the VisDrone-DET-test samples.

![Examples](/assets/predictions.png)

## Results

### Dataset

![VisDrone](/assets/vis_drone.png)

The experiment consisted of training five models on the VisDrone-DET-train dataset. The first model was the baseline YOLOv8n model, and the others were obtained using the methods described in the previous section. The second model used the P2 tensor, the third removed the P5 tensor and reduced the number of channels, the fourth added a bidirectional feature fusion module, and the fifth added a CBAM attention module.

The models described above were trained on NVIDIA GeForce RTX 3060 for 1000 epochs with a batch size equal to 4. The VisDrone-DET-val and VisDrone-DET-test datasets were used to test the trained models. The mean average precision mAP50 and mAP50-95 were used as metrics for model evaluation. All modifications to the baseline model resulted in a step-by-step increase in the accuracy of small-sized object recognition.

### Validation

The table below shows results on the VisDrone-DET-val dataset.

| Model                                                     | mAP50   | mAP50-95 | Number of parameters | Weights |
| --------                                                  | :-----: | :------: | :------------------: | ------- |
| Baseline (YOLOv8n)                                        | 0.349   | 0.203    | 3.001M               | [link](https://drive.google.com/file/d/188Wgp5tGGnmmaDP-J9C20gh8taauxX12/view?usp=sharing) |
| Baseline + P2                                             | 0.366   | 0.216    | 1.737M               | [link](https://drive.google.com/file/d/1e5iVQoKbI6C1AQPIQN4uOeQJrtb0A9pW/view?usp=sharing) |
| Baseline + P2 – P5                                        | 0.364   | 0.219    | 1.379M               | [link](https://drive.google.com/file/d/1Za6zRlKpeCx1KbDnT2xEAF3kGCmh2Trn/view?usp=sharing) |
| Baseline + P2 – P5 + bi-directional feature fusion        | 0.364   | 0.219    | 1.461M               | [link](https://drive.google.com/file/d/15SYiEF58HBiJ2djZyiOY0uroG7cdpvM0/view?usp=sharing) |
| Baseline + P2 – P5 + bi-directional feature fusion + CBAM | 0.398   | 0.241    | 1.542M               | [link](https://drive.google.com/file/d/1pt2wfe24ZGJz730o2hHgYuWX0cO_LBA7/view?usp=sharing) |


### Test

The table below shows results on the VisDrone-DET-test dataset.

| Model                                                     | mAP50   | mAP50-95 | Number of parameters |
| --------                                                  | :-----: | :------: | :------------------: |
| Baseline (YOLOv8n)                                        | 0.275   | 0.154    | 3.001M               |
| Baseline + P2                                             | 0.287   | 0.161    | 1.737M               |
| Baseline + P2 – P5                                        | 0.295   | 0.171    | 1.379M               |
| Baseline + P2 – P5 + bi-directional feature fusion        | 0.301   | 0.171    | 1.461M               |
| Baseline + P2 – P5 + bi-directional feature fusion + CBAM | 0.322   | 0.184    | 1.542M               |

### Perfomance

The table below shows inference speed results on the Jetson Nano.

| Model | Inference, ms | Post-process, ms |
| ---- | :------------:| :-------------: |
| Baseline (YOLOv8n) | 31.8 | 3.4 |
| Baseline + P2 – P5 + bi-directional feature fusion + CBAM | 60.6 | 8.3 |

Note that improved model is slower than baseline. Therefore there exists a trade-off between precision and speed (as always).

## Installation and use

### Insights about customizing ultralytics layers

Customizing ultralytics model is not as difficult as is seems. All changes to backbone which do not have influence on the training process can be easily performed with steps below.

1. Write your torch.nn.Module and implement forward pass in `ultralytics/nn/modules/block.py` (look at this file in the repo for example). Add it to `ultralytics/nn/modules/__init__.py` as well.

2. Import your custom block in `ultralytics/nn/tasks.py` and then write it's logic in **parse_model** function in `ultralytics/nn/tasks.py` (look at this file in the repo for example).

3. Customize model's .yaml config with your new custom block. See `model_configs` for examples.

### Use

Just clone this repo and train model as regular ultralytics model since main backend is not changed.

```
from ultralytics import YOLO

yolo = YOLO(model='model_configs/yolov8-p2_woP5.yaml')
yolo.train(data='ultralytics/cfg/datasets/VisDrone.yaml', epochs=1000, patience=150, batch=2, save_period=200)
```

### TensorRT inference

Export model in .onnx with simplify flag.

```
from ultralytics import YOLO

yolo = YOLO(model='path/to/best.pt')
yolo.export(format='onnx', imgsz=(384, 640), simplify=True)
```

Run `jetson/onnx2trt.py` to export in TensorRT (RUN IT ON TARGET PLATFORM, e.g. Jetson Nano)

`python3 onnx2trt.py -v --onnx-file <path to onnx file> --img <size of input image: width height, e.g. 640 384>`

Customize and run `jetson/inference.py`