#!/bin/bash

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

current_path=$(cd "$(dirname $0)";pwd)

if [ ! -e ${current_path}/config.properties ]; then
  cat >&2 <<-'EOF'
Error: Can't find configuration file in this folder.
Please download config.properties from our site and put it in this folder
EOF
  exit 1
fi
source ${current_path}/config.properties

download(){
  echo "downloading $2"
  curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=$1" > /dev/null
  curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=$1" -o $2
  rm ./cookie
}

null_error(){
  cat >&2 <<-EOF
Error: The value of $1 is NULL, but it is required.
EOF
  exit 1
}

prepare_model(){
  # $1=inception_v3
  eval model="$""$1"
  if [ ${model} ]; then
    echo "Need Run $1: ${model}"
    #if [ ${model} == "YES" ]; then
    eval model_name="$""$1_name"
    if [ ! ${model_name} ]; then
      null_error model_name
    fi
    eval model_dataset="$""$1_dataset"
    if [ ! ${model_dataset} ]; then
      null_error model_dataset
    fi

    eval model_url="$""$1_url"
    if [ ! -e ${current_path}/models/$1/${model_name} ]; then
      if [ ! ${model_url} ]; then
        null_error model_url
      fi
      download ${model_url} "${current_path}/models/$1.tar.gz"
      cd ${current_path}/models
      tar -xzvf "$1.tar.gz"
      rm "$1.tar.gz"
      cd ..
    fi
    if [ ! -d "${current_path}/dataset/${model_dataset}" ]; then
      dataset_url=${!model_dataset}
      if [ ! ${dataset_url} ]; then
        null_error "$1_dataset"
      fi
      download ${dataset_url} "${current_path}/dataset/${model_dataset}.tar.gz"
      cd ${current_path}/dataset
      tar -xzvf "${model_dataset}.tar.gz"
      rm "${model_dataset}.tar.gz"
      cd ..
    fi
    if [ ! -e ${current_path}/dataset/"${model_dataset}.gtruth" ]; then
      eval dataset_gtruth="$""${model_dataset}_gtruth"
      if [ ! ${dataset_gtruth} ]; then
        null_error "${model_dataset}_gtruth"
      fi
      download ${dataset_gtruth} ${current_path}/dataset/"${model_dataset}.gtruth"
    fi
  fi
}

[ ! -d "${current_path}/models/" ] && (mkdir ${current_path}/models/)
[ ! -d "${current_path}/dataset/" ] && (mkdir ${current_path}/dataset/)

prepare_model $2