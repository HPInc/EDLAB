#!/bin/bash

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

current_path=$(cd "$(dirname $0)";pwd)
root_path="${current_path}/../.."
openvino_dir="/opt/intel/openvino"
converter="${openvino_dir}/deployment_tools/model_optimizer/mo_tf.py"

source ${root_path}/config.properties
source ${current_path}/ncs2.properties
# source ${openvino_dir}/bin/setupvars.sh

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
  eval category="$""$1"
  if [ ${category} ]; then
    echo "Running $1 on Intel NCS2"
    eval model_name="$""$1_name"
    if [ ! ${model_name} ]; then
      null_error model_name
    fi
    model_dir="${root_path}/models/$1"
    model_pb="${model_dir}/${model_name}"
  
    eval dataset_name="$""$1_dataset"
    if [ ! ${dataset_name} ]; then
        null_error dataset_name
    fi
    dataset="${root_path}/dataset/${dataset_name}/"

    model_prefix=(${model_name//./ })
    model_prefix=${model_prefix[0]}
    ncs2_model_dir="${model_dir}/ncs2_model"
    results_dir="${root_path}/results/ncs2_$1"
    mkdir -p ${ncs2_model_dir}
    mkdir -p ${results_dir}
    eval batch="$""$1_batch"

    if [ ${category} == "classification" ]; then
      labels="${root_path}/dataset/${dataset_name}.gtruth"
      eval preprocessing="$""$1_preprocessing"

      python3 ${converter} --input_model=${model_pb} \
       --output_dir=${ncs2_model_dir} \
       --data_type=FP16 \
       --batch=${batch}

      if [ $1 == "resnet50_v1" ]; then
        python3 ${current_path}/classification/classification.py \
        --model=${ncs2_model_dir}/${model_prefix}.xml \
        --outdir=${results_dir} \
        --data=${dataset} \
        --label=${labels} \
        --num_images=50000 \
        --mode=async --num_requests=4 \
        --preprocess=${preprocessing} \
        --device=MYRIAD \
        --offset=1 \
        --power=yes \
        --port=/dev/rfcomm0
      else
        python3 ${current_path}/classification/classification.py \
        --model=${ncs2_model_dir}/${model_prefix}.xml \
        --outdir=${results_dir} \
        --data=${dataset} \
        --label=${labels} \
        --num_images=500 \
        --mode=async \
        --num_requests=4 \
        --preprocess=${preprocessing} \
        --device=MYRIAD \
        --offset=0 \
        --power=yes \
        --port=/dev/rfcomm0
      fi
    fi

    if [ ${category} == "detection" ]; then
      pipeline="${root_path}/models/$1/ssd_mobilenet_v2_coco.config"
      tf_extension="${openvino_dir}/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json"

      python3 ${converter} --input_model=${model_pb} \
      --output_dir=${ncs2_model_dir} \
      --data_type=FP16 --batch=${batch} \
      --tensorflow_object_detection_api_pipeline_config=${pipeline} \
      --tensorflow_use_custom_operations_config=${tf_extension}

      python3 ${current_path}/detection/detection.py \
      --model=${ncs2_model_dir}/${model_prefix}.xml \
      --data=${dataset} \
      --outdir=${results_dir} \
      --num_images=500 \
      --mode=async \
      --num_requests=4 \
      --device=MYRIAD \
      --ann_file=${root_path}/dataset/coco2014_test.gtruth \
      --power='yes' \
      --port='/dev/rfcomm0'

      # python3 ${root_path}/common_api/eval.py \
      # --annFile=${root_path}/dataset/coco2014_test.gtruth \
      # --resFile=${results_dir}/ncs2_results.json \
      # --imgIds=${results_dir}/ncs2_imageIds.txt
    fi
  fi
}

run_models $1