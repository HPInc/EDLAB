
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
from PIL import Image
import random
import os
from preprocessing_factory import get_preprocessing


def get_calib_mg(height, width, channels):
    preprocessing = get_preprocessing(preprocessing_name)
    pathDir = os.listdir(calibration_image)
    sample = random.sample(pathDir, 1)
    image = Image.open(calibration_image + sample[0])
    image = image.convert("RGB")
    image = preprocessing(image, height, width)
    image = image.reshape((height, width, channels))
    images = [image]
    return images


def create_graph_with_frozen():
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
        input_tensor = tensor_name_list[0] + ":0"
        output_tensor = tensor_name_list[-1] + ":0"
        input_shape = tf.compat.v1.get_default_graph().get_tensor_by_name(input_tensor).shape

        return graph_def, input_tensor, output_tensor, int(input_shape[1]), int(input_shape[2]), int(input_shape[3])


def convert_main():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)):

        frozen_graph, input_tensor, output_tensor, height, width, channels = create_graph_with_frozen()
        print(input_tensor, output_tensor, height, width, channels)

        if precision_mode == "FP32":
            # FP32
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph, nodes_blacklist=[output_tensor],
                                              precision_mode="FP32")
            trt_graph = converter.convert()

            with tf.gfile.GFile(model_path[0:-3] + "_FP32.pb", 'wb') as f:
                f.write(trt_graph.SerializeToString())

        if precision_mode == "FP16":
            # FP16
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph, nodes_blacklist=[output_tensor],
                                              precision_mode="FP16")
            trt_graph = converter.convert()

            with tf.gfile.GFile(model_path[0:-3] + "_FP16.pb", 'wb') as f:
                f.write(trt_graph.SerializeToString())

        if precision_mode == "INT8":
            # INT8
            images = get_calib_mg(height, width, channels)
            converter = trt.TrtGraphConverter(input_graph_def=frozen_graph, nodes_blacklist=[output_tensor],
                                              precision_mode="INT8", use_calibration=True)
            trt_graph = converter.convert()

            trt_graph = converter.calibrate(fetch_names=[output_tensor], num_runs=1,
                                            feed_dict_fn=lambda: {input_tensor: images})

            with tf.gfile.GFile(model_path[0:-3] + "_INT8.pb", 'wb') as f:
                f.write(trt_graph.SerializeToString())

    print("TensorRT model is successfully stored!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="The path to a frozen model file.", required=True)
    parser.add_argument("--calib_img", help="The image for calibration", required=True)
    parser.add_argument("--precision_mode", help="The precision mode", required=True)
    parser.add_argument("--preprocessing_name", help="The name of the preprocessing to use.", required=True)

    args = parser.parse_args()

    model_path = args.model_path
    calibration_image = args.calib_img
    preprocessing_name = args.preprocessing_name
    precision_mode = args.precision_mode

    convert_main()
