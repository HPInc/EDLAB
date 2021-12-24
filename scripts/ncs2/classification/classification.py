
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
lib_path = os.path.abspath(os.path.join('scripts/ncs2'))
sys.path.append(lib_path)
import numpy as np
import time
import csv
import threading
import re
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IECore
from utils.infer_request_wrap import InferRequestsQueue, InferReqWrap
from PIL import Image, ImageFile
from utils.preprocessing_factory import inception_preprocessing, vgg_preprocessing
from utils.serialUtil import SerialBlueTooth

################################################################
dir_name = os.path.dirname(os.path.realpath(__file__))
################################################################

ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_args():
    parser = ArgumentParser()

    parser.add_argument('--model', help='The model file in .XML format', required=True)
    parser.add_argument('--data', help='Path to dataset', required=True)
    parser.add_argument('--label', help='Caffe ground truth labels', required=True)
    parser.add_argument('--num_images', help='The number of input images', type=int, default=1)
    parser.add_argument('--mode', help='sync or async', default='async')
    parser.add_argument('--outdir', help='Directory for results', required=True)
    parser.add_argument('--num_requests', help='The number of inference requests', type=int, default=8)
    parser.add_argument('--preprocess', help='vgg or inception', default='inception')
    parser.add_argument('--device', help='Benchmark hardware platform', default='MYRIAD')
    parser.add_argument('--offset', help='The offset of results', type=int, default=0)
    parser.add_argument('--power', help='yes or no', default='yes')
    parser.add_argument('--port', help='The serial port for power measurement', default='/dev/rfcomm0')

    return parser

class Model:
    def __init__(self, xml_file):
        self.model = xml_file
        self.weights = os.path.splitext(xml_file)[0] + '.bin'

def get_power(result, port):
    serial = SerialBlueTooth(port)
    serial.connect()
    
    while True:
        power = serial.read()
        if power:
            result.append(power)
        time.sleep(0.005)

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
        elif preprocess == 'inception':
            image = inception_preprocessing(image, h, w)
        else:
            raise ValueError('No such pre-processing: ' + preprocess)
        
        image = image.transpose((2, 0, 1))
        batch.append(image)
        if (i + 1) % n == 0:
            np_batch = np.array(batch).reshape((n, c, h, w))
            images.append(np_batch)
            batch.clear()
        
    return images

def post_result(out_blob, results):
    for i in range(len(results)):
        results[i] = results[i][out_blob]
    
    results = np.array(results)
    results.resize(results.shape[0] * results.shape[1], results.shape[2])

    return results

def print_result(num_images, time, mode, top1, top5, power, outdir):
    print('---------Benchmark Result---------')
    print('Input: {} images'.format(num_images))
    print('Accuracy TOP 1: {}'.format(top1 * 100))
    print('Accuracy TOP 5: {}'.format(top5 * 100))
    print('Mode: {}'.format(mode))
    print('Time: {}'.format(time))
    print('Power: {}'.format(power))
    print('Output: {}'.format(outdir))
    print('---------Benchmark Result---------')

def get_accuracy(results, label_file, offset):
    top1 = 0
    top5 = 0

    with open(label_file, 'r') as f:
        labels = f.read()
    labels = labels.split()
    labels = labels[1::2]

    for i in range(len(results)):
        if (int(labels[i]) - offset) in np.argpartition(results[i], -5)[-5:]:
            top5 += 1
        if (int(labels[i]) - offset) == np.argmax(results[i]):
            top1 += 1
        
    return top1 / len(results), top5 / len(results)

#################################################################
def write_result(path, data):
    size = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        if not size:
            csv_write.writerow(
                ['Date', 'Platform', 'Model' , \
                'Top1 (%)', 'Top5 (%)', 'mAP (%)', 'Batch Size', \
                'Latency (ms)', 'Dataset size', \
                'Power (W)', 'EDP', 'Claimed Accuracy', 'LEDP'])
            csv_write.writerow(data)
        else:
            csv_write.writerow(data)
#################################################################

def main():
    args = build_args().parse_args()

    model = Model(args.model)
    ie_core = IECore()
    net = IENetwork(model=model.model, weights=model.weights)
    exec_net = ie_core.load_network(network=net, device_name=args.device, num_requests=args.num_requests)
    request_queue = InferRequestsQueue(exec_net.requests)

    for input_blob in net.inputs:
        pass
    for out_blob in net.outputs:
        pass

    print('Start preparing input...')
    inputs = get_images(args.data, args.num_images, net.inputs[input_blob].shape, args.preprocess)
    num_batches = len(inputs)
    
    power = []
    if args.power == 'yes':
        t = threading.Thread(target=get_power, args=(power, args.port, ), daemon=True)
        t.start()

    print('Start inference...')
    results = []
    start_infer = time.time()

    for i in range(num_batches):
        infer_request = request_queue.getIdleRequest()
        infer_input = inputs[i]

        if not infer_request:
            raise Exception('No idle requests')

        if args.mode == 'async':
            mode = 'async'
            infer_request.startAsync({input_blob: infer_input}, results)
        else:
            mode = 'sync'
            result = infer_request.infer({input_blob: infer_input})
            results.append(result)
    
    request_queue.waitAll()
    end_infer = time.time()

    results = post_result(out_blob, results)

    top1, top5 = get_accuracy(results, args.label, args.offset)
    print_result(num_batches * net.batch_size, end_infer - start_infer, mode, top1, top5, np.mean(power), args.outdir)

############################################################################
    num_images = num_batches * net.batch_size
    latency = (end_infer - start_infer) * 1000 / num_images    
    
    save_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_platform = 'NCS2'
    save_pbname = re.split(r"/|\.", args.model)[-2] + '.pb'
    save_top1 = str(top1 * 100)
    save_top5 = str(top5 * 100)
    save_map = 'NA'
    save_batchsize = str(net.batch_size)
    save_time = str(round(latency, 2))
    save_dataset = str(num_images)
    save_ledp =  'NA'
    save_edp = 'NA'
    save_clmd_acc = 'NA'

    if args.power == 'yes':
        save_power = round(np.mean(power), 2)
        
    save_all = [save_date, save_platform, \
                save_pbname, save_top1, save_top5, \
                save_map, save_batchsize, save_time, \
                save_dataset, save_power, \
                save_edp, save_clmd_acc, save_ledp]

    write_result(str(dir_name) + '/../../../result.csv', save_all)
############################################################################

if __name__ == '__main__':
    sys.exit(main())