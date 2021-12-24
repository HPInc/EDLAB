#!/bin/bash

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

current_path=$(cd "$(dirname $0)";pwd)
source ${current_path}/../../config.properties
source ${current_path}/gpu.properties

null_error(){
  cat >&2 <<-EOF
Error: The value of $1 is NULL, but it is required.
EOF
  exit 1
}

no_exist_error(){
  cat >&2 <<-EOF
Error: The folder or file $1 does not exist, but it is required.
Are you sure you have execused download.sh or prepared required items manually?
EOF

  exit 1
}

run_models(){
  eval model="$""$1"
  if [ ${model} ]; then
    echo "Runing $1 on $2"
    eval model_name="$""$1_name"
    if [ ! ${model_name} ]; then
      null_error model_name
    fi
    modelpath="${current_path}/../../models/$1/${model_name}"
    eval model_dataset="$""$1_dataset"
    if [ ! ${model_dataset} ]; then
      null_error model_dataset
    fi
    datasetpath="${current_path}/../../dataset/${model_dataset}/"
    groundtruth="${current_path}/../../dataset/${model_dataset}.gtruth"

    if [ ${model} == "classification" ]; then
      eval preprocessing="$""$1_preprocessing"
      eval lables_offset="$""$1_labelsoffset"
      if [ $2 == "gpu" ]; then
        python3 ${current_path}/classification/classification.py --model_path=${modelpath} --dataset_name=${datasetpath} --preprocessing_name=${preprocessing} --precision_mode=GPU --labels_path=${groundtruth} --batch_size=${batchsize} --load_number=${loadnumber} --labels_offset=${lables_offset} --report_power=${report_power}
      fi
      if [ $2 == "cpu" ]; then
        python3 ${current_path}/classification/classification.py --model_path=${modelpath} --dataset_name=${datasetpath} --preprocessing_name=${preprocessing} --precision_mode=CPU --labels_path=${groundtruth} --batch_size=${batchsize} --load_number=${loadnumber} --labels_offset=${lables_offset} --report_power=${report_power}
      fi
    fi

    if [ ${model} == "detection" ]; then

      if [ $2 == "gpu" ]; then
        python3 ${current_path}/detection/$1/detection.py --model_path=${modelpath} --dataset_name=${datasetpath} --precision_mode=GPU --labels_path=${groundtruth} --batch_size=${batchsize} --load_number=${loadnumber} --report_power=${report_power} --resultname="result_$1.json"
      fi
      if [ $2 == "cpu" ]; then
        python3 ${current_path}/detection/$1/detection.py --model_path=${modelpath} --dataset_name=${datasetpath} --precision_mode=CPU --labels_path=${groundtruth} --batch_size=${batchsize} --load_number=${loadnumber} --report_power=${report_power} --resultname="result_$1.json"
      fi
    fi
  fi
}

run_models $1 $2
