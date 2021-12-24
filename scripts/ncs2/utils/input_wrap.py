
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import numpy as np
import logging as log
from PIL import Image, ImageFile
from .preprocessing_factory import inception_preprocessing, vgg_preprocessing

ImageFile.LOAD_TRUNCATED_IMAGES = True

# class Images:
    # def __init__(self, input_dir, image_number, input_shape, preprocess):
        
    #     # self.image_number = image_number
    #     # self.input_shape = input_shape
    #     # self.image_ids = []
    #     # self.image_paths = self.get_image_path(input_dir)
    #     # self.image_orig_size = np.empty([len(self.image_paths),2],dtype=int)
    #     # self. images = self.read_images(preprocess)
    
    # def get_image_path(self, input_dir):
    #     image_paths = []
    #     image_names = os.listdir(input_dir)
    #     image_names.sort()
    #     if (len(image_names) < self.image_number):
    #         self.image_number = len(image_names) - (len(image_names) % self.input_shape[0])
    #         log.warning('Not enough images. Actual number of images: {}'.format(self.image_number))

    #     for i in range(self.image_number):
    #         image_path = os.path.join(input_dir, image_names[i])
    #         image_paths.append(image_path)
    #         self.image_ids.append(image_names[i][:12])

    #     return image_paths
    
    # def read_images(self, preprocess):
    #     n, c, h, w = self.input_shape
    #     images = []
    #     batch = []

    #     for i in range(self.image_number):
    #         image = Image.open(self.image_paths[i])
    #         self.image_orig_size[i] = image.size

    #         if image.mode == 'L':
    #             image = image.convert('RGB')
            
    #         if preprocess == 'vgg':
    #             resized_image = vgg_preprocessing(image, h, w)
    #         else:
    #             resized_image = inception_preprocessing(image, h, w)

    #         np_image = resized_image.transpose((2, 0, 1))
    #         batch.append(np_image)
    #         if ((i + 1) % n == 0):
    #             np_batch = np.array(batch).reshape((n, c, h, w))
    #             images.append(np_batch)
    #             batch.clear()
        
    #     return images

    # def read_idx_image(self, idx):
    #     return Image.open(self.image_paths[idx])

    # def normalize(self):
    #     for i in range(len(self.images)):
    #         self.images[i] =  self.images[i] / 255.0 

def get_images(image_dir, num_images, shape, preprocess):
    images = []
    batch = []
    names = os.listdir(image_dir)
    names.sort()
    n, c, h, w = shape

    if len(names) < num_images:
        num_images = len(names)
    
    for i in range(num_images):
        image = Image.open(os.path.join(image_dir, names[i]))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if preprocess == 'vgg':
            image = vgg_preprocessing(image, h, w)
        else if preprocess == 'inception':
            image = inception_preprocessing(image, h, w)
        else:
            raise ValueError('No such pre-processing: ' + preprocess)
        
        image = image.transpose((2, 0, 1))
        batch.append(image)
        if (i + 1) % n == 0:
            batch = np.array(batch).reshape((n, c, h, w))
            images.append(batch)
            batch.clear()
        
        return images

def inception_preprocessing(image: Image, height: int, width: int, central_fraction=0.875):
    image = central_crop(image, central_fraction)

    if height and width:
        image = image.resize((width, height), Image.BILINEAR)

    image = np.array(image)
    image = (image / 255.0 - 0.5) * 2.0
    # -0.5
    # *2.0

    return image


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