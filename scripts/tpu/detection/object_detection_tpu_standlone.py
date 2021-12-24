
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import time
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import json
import os
import sys
import numpy as np
import logging
from power import serialUtil
from multiprocessing import Process
import threading

logger = logging.getLogger()
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(filename="latency_summary.txt", filemode="a", format="%(asctime)s--%(levelname)s--%(message)s", datefmt=DATE_FORMAT)
logger.setLevel(logging.INFO)

alive = True


def power():

    # Initialize the serial port
    su = serialUtil.SerialBlueTooth("/dev/rfcomm0")
    print("Connect device")
    su.connect()

    # Read the data
    with open("power_results_detection.txt", 'w') as wf:
        while alive:
            wf.write(str(su.read())+'\n')


def main():

    global alive

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        help='Path of the detection model, it must be a SSD model with postprocessing operator.',
                        required=True)
    parser.add_argument('--input', help='Directory of the input image.', required=True)
    parser.add_argument('--label', help='Label.', required=True)
    parser.add_argument('--number', help='Running number to test.', type=int, required=True)
    parser.add_argument(
        '--keep_aspect_ratio',
        dest='keep_aspect_ratio',
        action='store_true',
        help=(
            'keep the image aspect ratio when down-sampling the image by adding '
            'black pixel padding (zeros) on bottom or right. '
            'By default the image is resized and reshaped without cropping. This '
            'option should be the same as what is applied on input images during '
            'model training. Otherwise the accuracy may be affected and the '
            'bounding box of detection result may be stretched.'))
    parser.set_defaults(keep_aspect_ratio=False)
    args = parser.parse_args()

    # Initialize engine.
    lm_start = time.time()
    engine = DetectionEngine(args.model)
    lm_end = time.time()

    print("Total {0} images are tested".format(len(os.listdir(args.input))))

    images = []
    # Open image.
    for filet in os.listdir(args.input):
        images.append(filet)

    ori_total_infer = 0
    total_save_time = 0
    total_start = time.time()

    p = threading.Thread(target=power)
    p.start()

    # logger.info("Running " + args.model + " for " + str(running_count) + " begins")
    k = 0
    anns = []
    imageid = []

    # Run inference.
    with open('result_detect_tpu.json', 'w') as wfile:
        for i in range(args.number):
            for item in images:
                image_t = Image.open(args.input + '/' + item)
                image_s = image_t
                # image_t = image_t.resize((300, 300))
                if image_t.mode == "L":
                    image_t = image_t.convert("RGB")
                ans = engine.detect_with_image(
                    image_t, threshold=0.05, keep_aspect_ratio=args.keep_aspect_ratio, relative_coord=False, top_k=1)
                # Get the inference time
                origin_inf = engine.get_inference_time()
                # logger.info("Iteration " + str(i) + " runs " + str(origin_inf) + " ms")
                ori_total_infer = ori_total_infer + origin_inf

                # Save result.
                if ans:
                    for obj in ans:
                        label_id = obj.label_id + 1

                        score = obj.score
                        box = obj.bounding_box.flatten().tolist()

                        ann_i_j = {
                            'id': k,
                            'image_id': int(item[13:-4]),
                            'category_id': label_id,
                            'segmentation': [],
                            'bbox': [float(box[0]), float(box[1]), float(box[2]) - float(box[0]) + 1,
                                     float(box[3]) - float(box[1]) + 1],
                            'score': float(score),
                            'iscrowd': 0
                        }

                        anns.append(ann_i_j)
                        imageid.append(int(item[13:-4]))
                        k = k + 1

                else:
                    logger.info('No object detected!')

            save_begin = time.time()
            json.dump(anns, wfile, ensure_ascii=False)
            save_end = time.time()
            total_save_time = total_save_time + save_end - save_begin

        alive = False
        p.join()

        with open("power_results_detection.txt", 'r') as rf:
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

            print("average power is {}".format(temp / count))

        logger.info("Per image inference runs {0} ms".format(ori_total_infer/args.number))
        logger.info("Running " + args.model + " finishes")

    end_time = time.time()

    resFile = 'result_detect_tpu.json'

    print("Total time taken {0} seconds".format(end_time - total_start))
    print("Loading model time taken {0} seconds".format(lm_end - lm_start))
    print("Total inference time {0} seconds".format(ori_total_infer / 1000))
    print("Total save time {0} seconds".format(total_save_time))


if __name__ == '__main__':
    main()
