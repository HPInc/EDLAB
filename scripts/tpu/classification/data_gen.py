
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import tensorflow as tf
import time
import os
from PIL import Image
import random
import argparse
from preprocessing_factory import *


class dataset_generator:
    """
        The data generator used for post training quantization
    """
    def __init__(self, imagedir, height, width, channel, num_vali, modelname):
        self.iidir = imagedir
        self.height = height
        self.width = width
        self.channel = channel
        self.num_vali = num_vali
        self.modelname = modelname

    def read_image(self):
        t_images = np.zeros((self.num_vali, self.height, self.width, self.channel), dtype=np.float32)

        filelist = os.listdir(self.iidir)

        for i in range(self.num_vali):
            im = Image.open(self.iidir + '/' + random.choice(filelist))
            try:
                preprocess_t = get_preprocessing(self.modelname)
                im = preprocess_t(im, self.height, self.width)
                t_images[i] = im
            except ValueError:
                i = i - 1
                continue
        return t_images

    def representative_dataset_gen(self):
        t_images = self.read_image()

        for i in range(self.num_vali):
            data = t_images[i]
            data = data[np.newaxis, :]
            yield [data]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, default="/home/lsq/Documents/HP-NTU/EdgeTPU/dataSet5000", help="imagedir")

    args = parser.parse_args()