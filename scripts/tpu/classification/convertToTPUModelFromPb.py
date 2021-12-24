# coding: utf-8

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import tensorflow as tf
import time
import json
import re
import argparse
from tensorflow.python import saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.tools import freeze_graph
import data_gen
import random
import sys
from preprocessing_factory import *

from PIL import Image

# Pathes for generated saved_model and TPU_model
export_path = "saved_models/"
TPU_model_folder = "TPU_models/"


def getTensorInfo(filename):
    """
    Get the Input and Output Tensor
    :param filename: The ".pb" format Tensorflow model
    :return: The name of inputTensor and outputTensor. The value of input's height, weight, and channels
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()

        # read the original model
        with tf.gfile.GFile(filename, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')

        # Get the inputTensor, outputTensor, inputShape
        tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        tensor_values_list = [tensor.values() for tensor in tf.get_default_graph().get_operations()]
        inputTensor = tensor_name_list[0] + ":0"
        outputTensor = tensor_name_list[-1] + ":0"
        inputshape = tensor_values_list[0]

        # Get the input's shape
        a = str(inputshape)
        parse_items = a.split(' ')
        id_name1 = re.findall(r"\d+\.?\d*", parse_items[3])
        id_name2 = re.findall(r"\d+\.?\d*", parse_items[4])
        id_name3 = re.findall(r"\d+\.?\d*", parse_items[5])

        try:
            (height, width, channels) = (int(id_name1[0]), int(id_name2[0]), int(id_name3[0]))
        except IndexError:
            print("Error: The input tensor's dimension is not fixed")
            print(inputshape)
            sys.exit(0)

        return inputTensor, outputTensor, height, width, channels


def savedModelGenerator(modelpath, dataset, height, width, channels, outTensor, inTensor, modelname):
    """
    Generate the saved_model used for post training quantization
    :param modelpath: The original model's path
    :param dataset: The training dataset (Note that here just one image is used to run this model)
    :param height: The input tensor's height
    :param width: The input tensor's height
    :param channels: The input tensor's channel
    :param outTensor: The name of output Tensor
    :param inTensor: The name of input Tensor
    :param normalize: Whether to normalize
    :return: Saved_model
    """
    # Randomly choose one image
    filename = []
    image_batch = []
    files = os.listdir(dataset)

    # Filter the gray image
    if channels > 1:
        for filei in files:
            imagei = Image.open(dataset + '/' + filei)
            imagei = np.array(imagei)
            if len(imagei.shape) > 2:
                filename.append(filei)
                break
    else:
        filename.append(random.choice(files))

    # Preprocessing is neural network dependent: resize, trans the channels, and add one dimension
    image = Image.open(dataset + '/' + filename[0])
    if image.mode == 'L':
        image = image.convert("RGB")
    bn, h, w, c = (1, height, width, channels)
    # image = image.resize((w, h), resample=0)
    preprocess_t = get_preprocessing(modelname)
    image = preprocess_t(image, h, w)
    image = np.expand_dims(image, axis=0)
    image_batch.append(image)

    # Loading the model
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(modelpath, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')

    # make the savel_model dir
    cwd = os.getcwd()

    if os.path.exists(export_path):
        os.chdir(export_path)
        os.system("rm -rf ./*")
        os.chdir(cwd)
    builder = saved_model.builder.SavedModelBuilder(export_path)

    # run the model to generate the saved_model
    with detection_graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(graph=detection_graph, config=config) as sess:

            input_tensor = detection_graph.get_tensor_by_name(inTensor)
            softmax_tensor = detection_graph.get_tensor_by_name(outTensor)

            signature = predict_signature_def(inputs={"myinput": input_tensor}, outputs={"myoutput": softmax_tensor})
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                 signature_def_map={'predict': signature})

            _ = sess.run(softmax_tensor, {inTensor: image_batch[0]})

            builder.save()


def convertToTpu(dg, mn):
    """
    Convert to the TPU-compatible model from the generated saved_model
    :param dg: The data_generator used for post training quantization
    :param mn: The corresponding model name
    :return: The converted model
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=export_path, signature_key="predict")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.allow_custom_ops = True
    converter.inference_output_type = tf.uint8
    converter.inference_input_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(dg.representative_dataset_gen)

    tflite_quant_model = converter.convert()

    if not os.path.exists(TPU_model_folder):
        os.mkdir(TPU_model_folder)

    with open(TPU_model_folder + mn + ".tflite", "wb") as wf:
        wf.write(tflite_quant_model)

    # Compile the tflite model to tpu-compatible tflite model
    os.chdir(TPU_model_folder)
    os.system("edgetpu_compiler -s " + mn + ".tflite")

    print("Successfully generate TPU model, stored at " + TPU_model_folder)


def main():

    # parse the args
    parser = argparse.ArgumentParser()

    # add new argument here
    parser.add_argument("-m", "--modelpath", type=str, default='./test/inception_v3_frozen_graph.pb',
                        help="model path", required=True)
    parser.add_argument("-t", "--trainingset", type=str, default="/home/lsq/Documents/HP-NTU/EdgeTPU/dataSet5000",
                        help="post training quantization imagedir", required=True)
    parser.add_argument("-vn", "--validnum", type=int, default=30, help="images num to be used in post "
                                                                        "training quantization")
    parser.add_argument("-mn", "--modelname", type=str, default="inceptionv3", help="model name", required=True)

    args = parser.parse_args()

    # Check if the provided images are enough. If not enough, the same images may be utilized more than once
    check_files = os.listdir(args.trainingset)
    if len(check_files) < args.validnum:
        print("Please provide more data than " + str(args.validnum) + " or change the value using -vn value "
                                                                      "(the default is 30")
        return 0

    in_tensor, out_tensor, height, width, channels = getTensorInfo(args.modelpath)

    savedModelGenerator(args.modelpath, args.trainingset, height, width, channels,
                        out_tensor, in_tensor, args.modelname)

    # New an object of data_generator
    dg = data_gen.dataset_generator(args.trainingset, height, width, channels, args.validnum, args.modelname)

    convertToTpu(dg, args.modelname)


if __name__ == "__main__":
    main()
