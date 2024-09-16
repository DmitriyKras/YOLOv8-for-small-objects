from __future__ import print_function
import os
import argparse
import tensorrt as trt

MAX_BATCH_SIZE = 1


def load_onnx(onnx_path):
    """Read the ONNX file."""
    if not os.path.isfile(onnx_path):
        print('ERROR: file (%s) not found!  You might want to run yolo_to_onnx.py first to generate it.' % onnx_path)
        return None
    else:
        with open(onnx_path, 'rb') as f:
            return f.read()


def set_net_batch(network, batch_size):
    """Set network input batch size.

    The ONNX file might have been generated with a different batch size,
    say, 64.
    """
    if trt.__version__[0] >= '7':
        shape = list(network.get_input(0).shape)
        shape[0] = batch_size
        network.get_input(0).shape = shape
    return network


def build_engine(onnx_path, img_size, model_name, dla_core, verbose=False):
    """Build a TensorRT engine from ONNX using the older API."""

    print('Loading the ONNX file...')
    onnx_data = load_onnx(onnx_path)
    if onnx_data is None:
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    EXPLICIT_BATCH = [] if trt.__version__[0] < '7' else \
        [1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        if not parser.parse(onnx_data):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
        network = set_net_batch(network, MAX_BATCH_SIZE)

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        if trt.__version__[0] < '7':  # older API: build_cuda_engine()
            if dla_core >= 0:
                raise RuntimeError('DLA core not supported by old API')
            builder.max_batch_size = MAX_BATCH_SIZE
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = True  # alternative: builder.platform_has_fast_fp16
            engine = builder.build_cuda_engine(network)
        else:  # new API: build_engine() with builder config
            builder.max_batch_size = MAX_BATCH_SIZE
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            config.set_flag(trt.BuilderFlag.FP16)
            profile = builder.create_optimization_profile()
            profile.set_shape(
                '000_net',                          # input tensor name
                (MAX_BATCH_SIZE, 3, img_size[1], img_size[0]),  # min shape
                (MAX_BATCH_SIZE, 3, img_size[1], img_size[0]),  # opt shape
                (MAX_BATCH_SIZE, 3, img_size[1], img_size[0]))  # max shape
            config.add_optimization_profile(profile)
            if dla_core >= 0:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = dla_core
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                print('Using DLA core %d.' % dla_core)
            engine = builder.build_engine(network, config)

        if engine is not None:
            print('Completed creating engine.')
        return engine


def onnx_to_tensorrt(onnx_file, img_size, dla_core, verbose):
    """Create a TensorRT engine for ONNX-based YOLO."""
    model_name = onnx_file.split('/')[-1].split('.')[0]

    engine = build_engine(onnx_file, img_size, model_name, dla_core, verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = '%s.engine' % model_name
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='enable verbose output (for debugging)')
    parser.add_argument("--onnx-file", default="model.onnx",
                        help="path to onnx file")
    parser.add_argument('--dla-core', type=int, default=-1,
                        help='id of DLA core for inference (0 ~ N-1)')
    parser.add_argument('--img', type=int, nargs='+', 
                        help='inference image size width, height')
    args = parser.parse_args()

    onnx_to_tensorrt(args.onnx_file, args.img, args.dla_core, args.verbose)

