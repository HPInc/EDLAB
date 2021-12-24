# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

# coding: utf-8
import csv
import io
import subprocess
import numpy as np
import os
import tensorflow as tf
import time
import json
from PIL import Image
import argparse
import sys

dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(dir_name) + "/../../../../common_api")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def open_images(test_image_txt, height, width, channels, current_number, total_number):
    images = []
    image_batch = []
    global image_size
    (h, w, c) = (height, width, channels)
    have_load = current_number * load_number
    count = 0
    batch_count = 0
    if total_number - have_load > load_number:
        current_load = load_number
    else:
        current_load = total_number - have_load
    for i in range(current_load):
        image = Image.open(dataset_name + test_image_txt[i + have_load])
        image_size.append(image.size)
        image = image.convert("RGB")
        image = image.resize((w, h), resample=0)
        image = np.array(image)
        image = image.reshape((h, w, c))

        if count % batch_size == 0:
            image_batch = []
        image_batch.append(image)
        if count % batch_size == (batch_size - 1):
            images.append(image_batch)
            batch_count = batch_count + 1
        count = count + 1

    if batch_count * batch_size < count:
        images.append(image_batch)

    return images


def write_result(path, data):
    size = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        if not size:
            csv_write.writerow(
                ['Date', 'Platform', 'Model', 'Top1 (%)', 'Top5 (%)', 'mAP (%)' 'Batch Size', 'Latency (ms)',
                 'Dataset size', 'Power (W)', 'EDP', 'Claimed Accuracy', 'LEDP'])
            csv_write.writerow(data)
        else:
            csv_write.writerow(data)


def run_main():
    test_images = []
    temp_path = os.path.join(dataset_name)
    for line in os.listdir(temp_path):
        test_images.append(line)

    total_images_num = len(test_images)
    total_iteration = int(total_images_num / load_number) + 1

    if report_power:
        global power_all
        zero_in_powers = 0

    times = []
    result_box = []
    result_score = []
    result_class = []
    result_num = []

    detection_graph = tf.Graph()
    with detection_graph.as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True

        with tf.gfile.GFile(model_path, 'rb') as fid:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')

        with tf.Session(graph=detection_graph, config=config) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            for iteration in range(total_iteration):
                images_load = open_images(test_images, 300, 300, 3, iteration, total_images_num)

                if report_power:
                    power_process = subprocess.Popen(
                        [str(dir_name) + "/../../support/power",
                         "/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input"], shell=False,
                        stderr=subprocess.PIPE)

                for images in images_load:
                    start = time.time()
                    (boxes, scores, classes, nums) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: images})

                    times.append(time.time() - start)
                    for ii in range(len(nums)):
                        box = boxes[ii]
                        num = nums[ii]
                        class_i = classes[ii]
                        score = scores[ii]
                        if len(box) == 0:
                            continue
                        result_box.append(box)
                        result_score.append(score)
                        result_class.append(class_i)
                        result_num.append(int(num))
                del images_load

                if report_power:

                    power_process.terminate()
                    powers = str(power_process.stderr.read(), 'utf-8').split()
                    powers = list(map(float, powers))
                    if len(powers) == 0:
                        zero_in_powers = zero_in_powers + 1
                    else:
                        power_all.append(0.001 * sum(powers) / len(powers))
                    del powers

    image_ids = save_results(total_images_num, test_images, result_box, result_score, result_class, result_num)
    MAP = accuracy(image_ids, labels_path, results_name)

    save_power = 'NA'
    save_top1 = 'NA'
    save_top5 = 'NA'
    save_latency = str(1000.0 * sum(times) / len(test_images))
    save_mAP = str(MAP)
    save_batchsize = str(batch_size)
    save_platform = 'TX2_' + precision_mode
    save_dataset = str(len(test_images))
    save_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_ledp = 'NA'
    save_edp = 'NA'
    save_pbname = str(model_path).split('/')[-1]
    save_claimed = 'NA'

    if report_power:
        if zero_in_powers > 0:
            print("Missing " + str(zero_in_powers) + " loads power!!!")
        save_power = sum(power_all) / len(power_all)
        save_edp = str(save_power * sum(times) / len(test_images) / len(test_images) * sum(times))
        save_power = str(save_power)
        print("Avg Power: ", save_power)

    print("Execution time: ", sum(times))
    print("Batch Size: ", batch_size)
    print("Load Number: ", load_number)
    print("Precision mode: ", precision_mode)

    save_all = [save_date, save_platform, save_pbname, save_top1,save_top5,save_mAP, save_batchsize, save_latency, save_dataset,
                save_power,save_edp, save_claimed, save_ledp]

    write_result(str(dir_name) + '/../../../../result.csv', save_all)



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


def save_results(total_images_num, test_images, result_box, result_score, result_class, result_num):
    anns = []
    image_ids = []
    k = 0

    for i in range(total_images_num):
        x_size = image_size[i][0]
        y_size = image_size[i][1]
        image_id = int(test_images[i].split('_')[-1][0:12])

        for j in range(result_num[i]):
            (y1, x1, y2, x2) = (result_box[i][j][0], result_box[i][j][1], result_box[i][j][2], result_box[i][j][3])
            category = int(result_class[i][j])
            score = float(result_score[i][j])
            xs = int(x1 * x_size)
            ys = int(y1 * y_size)
            ws = int(x2 * x_size - xs + 1)
            hs = int(y2 * y_size - ys + 1)
            ann_k = {
                'id': k,
                'image_id': image_id,
                'category_id': category,
                'segmentation': [],
                'bbox': [xs, ys, ws, hs],
                'score': score,
                'iscrowd': 0,
                'image': test_images[i]
            }
            anns.append(ann_k)
            image_ids.append(image_id)
            k = k + 1

    with open(results_name, 'w+', encoding='utf-8') as file:
        json.dump(anns, file, ensure_ascii=False)
    return image_ids


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="The path to a frozen model file.", required=True)
    parser.add_argument("--dataset_name", help="The name of the dataset to load.", required=True)
    parser.add_argument("--labels_path", help="The path to the true label file.", required=True)
    parser.add_argument("--batch_size", help="The number of samples in each batch.", type=int, default=64)
    parser.add_argument("--load_number", help="The number of samples in every load", type=int, default=1024)
    parser.add_argument("--precision_mode", help="The precision mode", required=True)
    parser.add_argument("--resultname", help="The path of the result save file", required=True)
    parser.add_argument("--report_power", help="If need report power consumption", type=int, default=1)

    args = parser.parse_args()

    model_path = args.model_path
    dataset_name = args.dataset_name
    labels_path = args.labels_path
    batch_size = args.batch_size
    load_number = args.load_number
    results_name = args.resultname

    report_power = args.report_power

    if report_power:
        power_all = []

    precision_mode = args.precision_mode

    if precision_mode == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if precision_mode != "CPU" and precision_mode != "GPU":
        import tensorflow.contrib.tensorrt as trt

    image_size = []
    run_main()
