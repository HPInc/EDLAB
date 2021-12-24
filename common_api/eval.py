
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from argparse import ArgumentParser
# import numpy as np
# import skimage.io as io
# import pylab

# pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def build_args():
    parser = ArgumentParser()

    parser.add_argument('--annFile', help='The ground truth file', required=True)
    parser.add_argument('--resFile', help='The experimental results', required=True)
    parser.add_argument('--imgIds', help='The id file of tested images', required=True)

    return parser

args = build_args().parse_args()

annType = 'bbox'
# annFile = '/home/mlab/Documents/konghao/workspace/hp-ntu/dataset/coco2014/annotations/instances_val2014.json'
cocoGt = COCO(args.annFile)

# resFile = '/home/mlab/Documents/konghao/workspace/hp-ntu/HP-Tool/output/results.json'
cocoDt = cocoGt.loadRes(args.resFile)

# minival_ids = '/home/mlab/Documents/konghao/workspace/hp-ntu/dataset/coco2014/mscoco_minival_ids.txt'
with open(args.imgIds, 'r') as f:
    imgIds = f.read()
imgIds = imgIds.split()
imgIds = list(map(int,imgIds))

imgIds = sorted(imgIds)

cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()