
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import csv
import argparse
import time
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, required=True)
    parser.add_argument('-m', type=str, required=True)
    args = parser.parse_args()

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    platform = "Edgetpu"
    model = args.m + '.pb'
    Batch_size = 1

    with open(args.d + "/result.txt", 'r') as rf:
        line = rf.readline()
        imagenum = int(line.strip().split(' ')[1])
        line = rf.readline()
        line = rf.readline()
        line = rf.readline()
        infer_time = float(line.strip().split(' ')[3])
        line = rf.readline()
        power = float(line.strip().split(' ')[3])
        line = rf.readline()
        accuracy1 = float(line.strip().split(' ')[2])
        line = rf.readline()
        accuracy5 = float(line.strip().split(' ')[2])

    edp = str(power * infer_time * infer_time / imagenum / imagenum)
    ledp = str((1 - (1.0 * accuracy1 / imagenum)) * power * infer_time / imagenum / imagenum)

    data = [date, platform, model, str(accuracy1*100), str(accuracy5*100), 'NA', Batch_size,
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
