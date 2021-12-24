
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import csv
import argparse
import time
import os
import sys
import io
dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(dir_name) + "/../../../common_api")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def accuracy(imgIds, label_path, result_path):
    annFile = label_path
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(result_path)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    cocoEval.summarize()
    coco_summary = sys.stdout.getvalue()
    sys.stdout = _stdout

    mAP = ((coco_summary.split('\n')[0]).split('|')[-1]).split('=')[-1].strip()

    return float(mAP)*100


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, required=True)
    args = parser.parse_args()

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    platform = "Edgetpu"
    model = args.m + '.pb'
    Batch_size = 1

    imageIds = []
    for filee in os.listdir('dataset/coco2014_test/'):
        imageIds.append(int(filee[13:-4]))

    with open("result.txt", 'r') as rf:
        line = rf.readline()
        imagenum = int(line.strip().split(' ')[1])
        line = rf.readline()
        power = float(line.strip().split(' ')[3])
        line = rf.readline()
        line = rf.readline()
        line = rf.readline()
        infer_time = float(line.strip().split(' ')[3])
        line = rf.readline()

    accuracy1 = accuracy(imageIds, "dataset/coco2014_test.gtruth", "result_detect_tpu.json")

    edp = str(power * infer_time * infer_time / imagenum / imagenum)
    ledp = str((1 - (1.0 * accuracy1 / imagenum)) * power * infer_time / imagenum / imagenum)

    data = [date, platform, model, 'NA', 'NA', str(accuracy1), Batch_size,
            str(infer_time*1000/imagenum), str(imagenum), str(power), edp, 'NA', 'NA']

    target_path = 'result.csv'
    print(target_path)
    size = os.path.exists(target_path) and os.path.getsize(target_path) > 0
    with open(target_path, 'a+') as f:
        csv_write = csv.writer(f)
        if not size:
            csv_write.writerow(
                ['Date', 'Platform', 'Model', 'Top1 (%)', 'Top5 (%)', 'mAP (%)', 'Batch Size',
                 'Latency (ms)', 'Dataset size', 'Power (W)', 'EDP', 'Claimed Accuracy', 'LEDP'])
            csv_write.writerow(data)
        else:
            csv_write.writerow(data)
