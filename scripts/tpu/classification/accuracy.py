
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
import json


def accuracy(label_path, self_label_path, image_num):
    labels = OrderedDict()
    with open(label_path, 'r') as rf:
        # labelJson = json.load(rf)
        line = rf.readline()
        while line:
            temp_t = line.strip().split(' ')
            labels[temp_t[0][1:-1]] = int(temp_t[1])
            line = rf.readline()

    # for key in labels.keys():
    #     print(key, labels[key])

    succ_count = 0
    with open(self_label_path, 'r') as rf1:
        line = rf1.readline()
        while line:
            line = line.strip()
            temp = line.split(' ')
            c_index = temp[0]
            if int(temp[1]) == labels[c_index]:
                # print(line)
                succ_count += 1
                # print("success")
            line = rf1.readline()
    print(" %0.3f" % (succ_count / image_num))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=str, help="real label path", required=True)
    parser.add_argument("--result", type=str, help="own result", required=True)
    parser.add_argument("--imagenum", type=int, help="image num", required=True)

    args = parser.parse_args()

    accuracy(args.label, args.result, args.imagenum)

