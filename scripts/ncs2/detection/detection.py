
# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

import sys, os
lib_path = os.path.abspath(os.path.join('scripts/ncs2'))
sys.path.append(lib_path)
import numpy as np
import time
import threading
import json
import csv
import io
import re
from argparse import ArgumentParser
from openvino.inference_engine import IENetwork, IECore
from utils.infer_request_wrap import InferRequestsQueue, InferReqWrap
from utils.serialUtil import SerialBlueTooth
from PIL import Image, ImageFile
# from preprocess.preprocessing_factory import inception_preprocessing, vgg_preprocessing

################################################################
dir_name = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(dir_name) + "/../../../common_api")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
################################################################

ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_args():
    parser = ArgumentParser()

    parser.add_argument('--model', help='The model file in .XML format', required=True)
    parser.add_argument('--data', help='Path to dataset', required=True)
    parser.add_argument('--outdir', help='Directory for results', required=True)
    parser.add_argument('--ann_file', help='The ground-truth file', required=True)
    parser.add_argument('--num_images', help='The number of input images', type=int, default=1)
    parser.add_argument('--mode', help='sync or async', default='async')
    parser.add_argument('--num_requests', help='The number of inference requests', type=int, default=8)
    parser.add_argument('--preprocess', help='vgg or inception', default='vgg')
    parser.add_argument('--device', help='Benchmark hardware platform', default='MYRIAD')
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
    sizes = []
    ids = []
    names = os.listdir(image_dir)
    names.sort()
    n, c, h, w = shape 

    if len(names) < num_images:
        num_images = len(names)
    
    for i in range(num_images):
        image = Image.open(os.path.join(image_dir, names[i]))
        sizes.append(image.size)
        ids.append(int(names[i][13:25]))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Image preprocess
        # if preprocess == 'vgg':
        #     image = vgg_preprocessing(image, h, w)
        # elif preprocess == 'inception':
        #     image = inception_preprocessing(image, h, w)
        # else:
        #     raise ValueError('No such pre-processing: ' + preprocess)
        
        image = image.resize((w, h), resample=0)
        image = np.array(image)
        image = image.transpose((2, 0, 1))
        batch.append(image)
        if (i + 1) % n == 0:
            np_batch = np.array(batch)
            images.append(np_batch)
            batch.clear()
         
    return images, sizes, ids

def post_result(out_blob, results):
    for i in range(len(results)):
        results[i] = results[i][out_blob]
    
    results = np.array(results)
    results.resize(results.shape[0] * results.shape[3], results.shape[4])
    # print(np.shape(results))

    return results

def print_result(num_images, time, mode, power):
    print('---------Benchmark Result---------')
    print('Input: {} images'.format(num_images))
    print('Mode: {}'.format(mode))
    print('Time: {}'.format(time))
    print('Power: {}'.format(power))
    print('---------Benchmark Result---------')

def save_results(outdir, results, sizes, ids):
    obj_idx = 0
    img_idx = 0
    objs = []

    while img_idx < len(ids):
        cur_idx = img_idx * 100 + obj_idx

        if obj_idx >= 100 or results[cur_idx][0] == -1:
            obj_idx = 0
            img_idx += 1
        else:
            xs = int(results[cur_idx][3] * sizes[img_idx][0])
            ys = int(results[cur_idx][4] * sizes[img_idx][1])
            ws = int(results[cur_idx][5] * sizes[img_idx][0] - xs + 1)
            hs = int(results[cur_idx][6] * sizes[img_idx][1] - ys + 1)

            obj = {
                'image_id': ids[img_idx],
                'category_id': int(results[cur_idx][1]),
                'bbox': [xs, ys, ws, hs],
                'score': float(results[cur_idx][2])
            }

            objs.append(obj)
            obj_idx += 1

    with open(outdir + '/ncs2_results.json', 'w', encoding='utf-8') as f:
        json.dump(objs, f, ensure_ascii=False)
    
    with open(outdir + '/ncs2_imageIds.txt', 'w') as f:
        for i in range(len(ids)):
            f.write(str(ids[i]) + '\n')

# ######################################################
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
########################################################

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

    print(net.inputs[input_blob].shape)
    print('Start prepare input...')
    inputs, img_sizes, img_ids = get_images(args.data, args.num_images, net.inputs[input_blob].shape, args.preprocess)
    num_images = len(img_ids)
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
    save_results(args.outdir, results, img_sizes, img_ids)

############################################################################    
    mAP = accuracy(img_ids, args.ann_file, args.outdir + '/ncs2_results.json')
    latency = (end_infer - start_infer) * 1000 / num_images

    save_power = 'NA'
    save_map = str(mAP)
    save_top1 = 'NA'
    save_top5 = 'NA'
    save_time = str(round(latency, 2))
    save_batchsize = str(net.batch_size)
    save_platform = 'NCS2'
    save_dataset = str(num_images)
    save_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_edp = 'NA'
    save_ledp = 'NA'
    save_clmd_acc = 'NA'
    save_pbname = re.split(r"/|\.", args.model)[-2] + '.pb'

    if args.power == 'yes':
        save_power = round(np.mean(power), 2)

    print_result(num_images, end_infer - start_infer, mode, save_power)

    save_all = [save_date, save_platform, \
                save_pbname, save_top1, save_top5, \
                save_map, save_batchsize, save_time, \
                save_dataset, save_power, \
                save_edp, save_clmd_acc, save_ledp]

    write_result(str(dir_name) + '/../../../result.csv', save_all)
#############################################################################

if __name__ == '__main__':
    sys.exit(main())