# coding: utf-8

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import tensorflow as tf
import time
from PIL import Image
import subprocess
import csv

from preprocessing_factory import get_preprocessing


def open_images(test_image_txt, height, width, channels, current_number, total_number, preprocessing):
    images = []
    image_batch = []
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
        image = image.convert("RGB")
        image = preprocessing(image, h, w)
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


def create_graph():
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
        input_tensor = tensor_name_list[0] + ":0"
        output_tensor = tensor_name_list[-1] + ":0"
        input_shape = tf.compat.v1.get_default_graph().get_tensor_by_name(input_tensor).shape

        return input_tensor, output_tensor, int(input_shape[1]), int(input_shape[2]), int(input_shape[3])


def get_accuracy(result_all, test_images):
    top_1 = []
    top_5 = []
    id = []
    for i in range(len(result_all)):
        id.append(int(test_images[i][15:23]))
        sorted_result = result_all[i].argsort()
        top_1.append(sorted_result[-1])
        top_5.append(sorted_result[-5:])

    labels = []
    top_1_num = 0
    top_5_num = 0
    with open(labels_path, 'rb') as file:
        for line in file:
            lab_tmp = line.decode().strip().split(' ')
            labels.append(int(lab_tmp[1]))

    for i in range(len(id)):
        if (labels[id[i] - 1] - labels_offset) == top_1[i]:
            top_1_num = top_1_num + 1
            top_5_num = top_5_num + 1
        elif (labels[id[i] - 1] - labels_offset) in top_5[i]:
            top_5_num = top_5_num + 1

    del labels
    return top_1_num, top_5_num


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
    preprocessing = get_preprocessing(preprocessing_name)

    test_images = []
    temp_path = os.path.join(dataset_name)
    for files in os.listdir(temp_path):
        test_images.append(files)

    total_images_num = len(test_images)
    total_iteration = int(total_images_num / load_number) + 1
    result_all = []
    times = []

    if report_power:
        global power_all
        zero_in_powers = 0

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        input_tensor, output_tensor, height, width, channels = create_graph()
        with tf.Session(graph=detection_graph, config=config) as sess:
            image_tensor = detection_graph.get_tensor_by_name(output_tensor)
            for iteration in range(total_iteration):
                images_load = open_images(test_images, height, width, channels, iteration, total_images_num,
                                          preprocessing)

                if report_power:
                    power_process = subprocess.Popen(
                        [str(dir_name) + "/../support/power",
                         "/sys/bus/i2c/drivers/ina3221x/0-0041/iio_device/in_power0_input"], shell=False,
                        stderr=subprocess.PIPE)

                for images in images_load:
                    start = time.time()
                    predictions = sess.run(image_tensor, {input_tensor: images})
                    times.append(time.time() - start)
                    for precision in predictions:
                        result_all.append(precision)

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

    top_1_num, top_5_num = get_accuracy(result_all, test_images)

    save_power = 'NA'
    save_top1 = str(100.0 * top_1_num / len(test_images))
    save_top5 = str(100.0*top_5_num/len(test_images))
    save_mAP = 'NA'
    save_latency = str(1000.0 * sum(times) / len(test_images))
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
        save_power = sum(power_all) / len(power_all)  # average power

        save_edp = str(save_power * sum(times) / len(test_images) / len(test_images) * sum(times))
        save_power = str(save_power)
        print("Avg Power: ", save_power)

    print('Top-1 Accuracy: ', top_1_num / len(test_images))
    print('Top-5 Accuracy: ', top_5_num / len(test_images))
    print("Execution time: ", save_latency)
    print("Batch Size: ", batch_size)
    print("Load Number: ", load_number)
    print("Precision mode: ", precision_mode)

    save_all = [save_date, save_platform, save_pbname, save_top1,save_top5,save_mAP, save_batchsize, save_latency, save_dataset,
                save_power,save_edp, save_claimed, save_ledp]

    write_result(str(dir_name) + '/../../../result.csv', save_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="The path to a frozen model file.", required=True)
    parser.add_argument("--dataset_name", help="The name of the dataset to load.", required=True)
    parser.add_argument("--labels_path", help="The path to the true label file.", required=True)
    parser.add_argument("--batch_size", help="The number of samples in each batch.", type=int, default=64)
    parser.add_argument("--load_number", help="The number of samples in every load", type=int, default=1024)
    parser.add_argument("--preprocessing_name", help="The name of the preprocessing to use.", required=True)
    parser.add_argument("--labels_offset", help="An offset for the labels in the dataset. This flag is primarily used "
                                                "to evaluate the VGG and ResNet architectures which do not use a "
                                                "background class for the ImageNet dataset.", type=int, default=0)
    parser.add_argument("--precision_mode", help="The precision mode", required=True)
    parser.add_argument("--report_power", help="If need report power consumption", type=int, default=1)

    args = parser.parse_args()

    model_path = args.model_path
    dataset_name = args.dataset_name
    labels_path = args.labels_path
    batch_size = args.batch_size
    load_number = args.load_number
    preprocessing_name = args.preprocessing_name
    labels_offset = args.labels_offset

    report_power = args.report_power
    dir_name = os.path.dirname(os.path.realpath(__file__))

    if report_power:
        power_all = []

    precision_mode = args.precision_mode

    if precision_mode == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if precision_mode != "CPU" and precision_mode != "GPU":
        import tensorflow.contrib.tensorrt as trt

    run_main()
