
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image
import numpy as np


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
    }

    return preprocessing_fn_map[name]


def inception_preprocessing(image: Image, height: int, width: int, central_fraction=0.875):
    image = central_crop(image, central_fraction)

    if height and width:
        image = image.resize((width, height), Image.BILINEAR)

    image = np.array(image)
    image = (image / 255.0 - 0.5) * 2.0
    # -0.5
    # *2.0

    return image


def cifarnet_preprocessing():
    return


def lenet_preprocessing():
    return


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

    image = np.array(image)

    means = np.array([123.68, 116.78, 103.94] * (width * height))

    means = means.reshape((height, width, 3))

    return image - means


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
