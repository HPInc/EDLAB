
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import os
import time
from collections import OrderedDict
import numpy as np
import accuracy
import json
import sys
import logging
from power import serialUtil
from multiprocessing import Process
import threading

logger = logging.getLogger()
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(filename="latency_summary.txt", filemode="a", format="%(asctime)s--%(levelname)s--%(message)s", datefmt=DATE_FORMAT)
logger.setLevel(logging.INFO)

alive = True


def cifarnet_preprocessing():
    return


def lenet_preprocessing():
    return


def get_preprocessing(name):
    preprocessing_fn_map = {
        'cifarnet': cifarnet_preprocessing,
        'inception': inception_preprocessing,
        'inception_v1': inception_preprocessing,
        'inception_v2': inception_preprocessing,
        'inception_v3': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'lenet': lenet_preprocessing,
        'mobilenet_v1': inception_preprocessing,
        'mobilenet_v2': inception_preprocessing,
        'mobilenet_v2_035': inception_preprocessing,
        'mobilenet_v3_small': inception_preprocessing,
        'mobilenet_v3_large': inception_preprocessing,
        'mobilenet_v3_small_minimalistic': inception_preprocessing,
        'mobilenet_v3_large_minimalistic': inception_preprocessing,
        'mobilenet_edgetpu': inception_preprocessing,
        'mobilenet_edgetpu_075': inception_preprocessing,
        'mobilenet_v2_140': inception_preprocessing,
        'nasnet_mobile': inception_preprocessing,
        'nasnet_large': inception_preprocessing,
        'pnasnet_mobile': inception_preprocessing,
        'pnasnet_large': inception_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'resnet_v1_101': vgg_preprocessing,
        'resnet_v1_152': vgg_preprocessing,
        'resnet_v1_200': vgg_preprocessing,
        'resnet_v2_50': vgg_preprocessing,
        'resnet_v2_101': vgg_preprocessing,
        'resnet_v2_152': vgg_preprocessing,
        'resnet_v2_200': vgg_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
        'mnasnet_b1': inception_preprocessing
    }

    return preprocessing_fn_map[name]


def central_crop(image: Image, central_fraction: float):
    # image is PIL Image Format

    img_h = image.size[1]
    img_w = image.size[0]

    bbox_h_start = int((1.0 * img_h - img_h * central_fraction) / 2)
    bbox_w_start = int((1.0 * img_w - img_w * central_fraction) / 2)

    bbox_h_size = img_h - bbox_h_start * 2
    bbox_w_size = img_w - bbox_w_start * 2

    bbox = (bbox_w_start, bbox_h_start, bbox_w_start + bbox_w_size, bbox_h_start + bbox_h_size)
    return image.crop(bbox)


def inception_preprocessing(image: Image, height: int, width: int, central_fraction=0.875):
    image = central_crop(image, central_fraction)

    if height and width:
        image = image.resize((width, height), Image.BILINEAR)

    return image


def vgg_preprocessing(image: Image, height: int, width: int, resize_side=256):
    img_h = image.size[1]
    img_w = image.size[0]
    if img_h > img_w:
        scale = 1.0 * resize_side / img_w
    else:
        scale = 1.0 * resize_side / img_h

    new_height = int(img_h * scale)
    new_width = int(img_w * scale)
    image = image.resize((new_width, new_height), Image.BILINEAR)
    offset_height = (new_height - height) / 2
    offset_width = (new_width - width) / 2
    image = image.crop((offset_width, offset_height, offset_width + width, offset_height + height))

    return image


def power():

    # Initialize the serial port
    su = serialUtil.SerialBlueTooth("/dev/rfcomm0")
    su.connect()
    # Read the data
    with open("power_results.txt", 'w') as wf:
        while alive:
            wf.write(str(su.read())+'\n')


def main():

    global alive

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', help='Dataset Path', type=str, required=True)
    parser.add_argument('--model', help='File path of Tflite model.', type=str, required=True)
    parser.add_argument('--number', help='Running number to test.', type=int, required=True)
    parser.add_argument('--label', type=str, help="real label path", required=True)
    parser.add_argument('--modelname', type=str, help="model name", required=True)

    args = parser.parse_args()

    # Initialize engine
    lm_start = time.time()
    engine = ClassificationEngine(args.model)
    lm_end = time.time()

    input_shape = engine.get_input_tensor_shape()

    image_files = {}
    i = 0

    # Force the order of images read as the number order in original dataset (ImageNet)
    for filett in os.listdir(args.data):
        image_files[i] = filett
        i += 1

    print("Total {0} images are tested".format(len(image_files)))

    ori_total_infer = 0
    total_save_time = 0

    logger.info("Running " + args.model + " for " + str(args.number) + " begins")
    p = threading.Thread(target=power)
    p.start()
    total_start = time.time()

    # Run inference.
    with open("temp_result", 'w') as wf:
        with open("temp_result_5", 'w') as wf1:
            for i in range(args.number):
                for key in image_files:
                    image_t = Image.open(args.data + '/' + image_files[key])
                    if image_t.mode == 'L':
                        image_t = image_t.convert("RGB")
                    # To resize the image
                    preprocess_t = get_preprocessing(args.modelname)
                    image_t = preprocess_t(image_t, input_shape[1], input_shape[2])
                    # Execute the engine
                    results = engine.classify_with_image(image_t, top_k=1, threshold=1e-10)
                    # Get the inference time
                    origin_inf = engine.get_inference_time()
                    # logger.info("Iteration " + str(i) + " runs " + str(origin_inf) + " ms")
                    ori_total_infer = ori_total_infer + origin_inf
                    save_begin = time.time()
                    for result in results:
                       wf.write(image_files[key] + ' ' + str(result[0]) + '\n')

                    results = engine.classify_with_image(image_t, top_k=5, threshold=1e-10)
                    for result in results:
                       wf1.write(image_files[key] + ' ' + str(result[0]) + '\n')
                    save_end = time.time()
                    total_save_time = total_save_time + save_end - save_begin

            end_time = time.time()
            alive = False
            p.join()
            print("Total time taken {0} seconds".format(end_time - total_start))
            print("Loading model time taken {0} seconds".format(lm_end - lm_start))
            print("Total inference time {0} seconds".format(ori_total_infer/1000))

            logger.info("Per image inference runs {0} ms".format(ori_total_infer/args.number))
            logger.info("Running " + args.model + " finishes")

            with open("power_results.txt", 'r') as rf:
                line = rf.readline()
                count = 0
                temp = 0.0
                while line:
                    line = line.strip()
                    if line == "None":
                        line = rf.readline()
                        continue
                    else:
                        count += 1
                        temp += float(line)
                        line = rf.readline()

                print("Average power is {}".format(temp / count))
            # print("Average power is 1.0")
#    print("Total save time {0} seconds".format(total_save_time))

    print("Top-1 accuracy:", end='')
    accuracy.accuracy(args.label, "temp_result", len(image_files))
    print("Top-5 accuracy:", end='')
    accuracy.accuracy(args.label, "temp_result_5", len(image_files))


if __name__ == '__main__':
    main()







