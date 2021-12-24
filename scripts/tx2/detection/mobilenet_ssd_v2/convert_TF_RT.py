# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import argparse
from PIL import Image
import random
import os


def get_calib_mg(height, width, channels):
    pathDir = os.listdir(calibration_image)
    sample = random.sample(pathDir, 1)
    image = Image.open(calibration_image + sample[0])
    image = image.convert("RGB")
    image = image.resize((height, width), resample=0)
    image = np.array(image)
    image = image.reshape((height, width, channels))

    images = [image]
    return images


def convert_main():
    detection_graph = tf.Graph()
    with detection_graph.as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True

        with tf.gfile.GFile(model_path, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=detection_graph, config=config) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            height = 300
            width = 300
            channels = 3
            print(image_tensor, detection_boxes, detection_scores, detection_classes, num_detections, height, width,
                  channels)

            if precision_mode == "FP32":
                # FP32
                converter = trt.TrtGraphConverter(input_graph_def=graph_def,
                                                  nodes_blacklist=[detection_boxes, detection_scores, detection_classes,
                                                                   num_detections],
                                                  precision_mode="FP32")
                trt_graph = converter.convert()

                with tf.gfile.GFile(model_path[0:-3] + "_FP32.pb", 'wb') as f:
                    f.write(trt_graph.SerializeToString())

            if precision_mode == "FP16":
                # FP16
                converter = trt.TrtGraphConverter(input_graph_def=graph_def,
                                                  nodes_blacklist=[detection_boxes, detection_scores, detection_classes,
                                                                   num_detections],
                                                  precision_mode="FP16")
                trt_graph = converter.convert()

                with tf.gfile.GFile(model_path[0:-3] + "_FP16.pb", 'wb') as f:
                    f.write(trt_graph.SerializeToString())

            if precision_mode == "INT8":
                # INT8
                images = get_calib_mg(height, width, channels)
                converter = trt.TrtGraphConverter(input_graph_def=graph_def,
                                                  nodes_blacklist=[detection_boxes, detection_scores, detection_classes,
                                                                   num_detections],
                                                  precision_mode="INT8", use_calibration=False)
                trt_graph = converter.convert()

                # todo: add calibration for detection
                # trt_graph = converter.calibrate(
                #   fetch_names=[detection_boxes, detection_scores, detection_classes, num_detections], num_runs=1,
                #  input_map_fn=lambda: {image_tensor: images})

                with tf.gfile.GFile(model_path[0:-3] + "_INT8.pb", 'wb') as f:
                    f.write(trt_graph.SerializeToString())

    print("TensorRT model is successfully stored!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="The path to a frozen model file.", required=True)
    parser.add_argument("--calib_img", help="The image dir for calibration", required=True)
    parser.add_argument("--precision_mode", help="The precision mode", required=True)

    args = parser.parse_args()

    model_path = args.model_path
    calibration_image = args.calib_img

    precision_mode = args.precision_mode

    convert_main()
