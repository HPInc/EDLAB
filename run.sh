#!/bin/bash

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

null_error(){
  cat >&2 <<-EOF
Error: The value of $1 is NULL, but it is required.
EOF
  exit 1
}

no_exist_error(){
  cat >&2 <<-EOF
Error: The folder or file $1 does not exist, but it is required.
Please make sure that you have executed download.sh or prepared all required packages manually.
EOF

  exit 1
}

usage(){
  echo "usge meassage"
}

current_path=$(cd "$(dirname $0)";pwd)
bash ${current_path}/download.sh $1 $2


if [ ! -d "${current_path}/models/" ]; then
  no_exist_error ${current_path}/models/
fi

if [ ! -d "${current_path}/dataset/" ]; then
  no_exist_error ${current_path}/dataset/
fi

if [ ! -d "${current_path}/scripts/" ]; then
  no_exist_error ${current_path}/scripts/
fi


case $1 in

  "tx2")
    echo "Running Nvidia TX2"
    bash ${current_path}/scripts/tx2/tx2.sh $2
    ;;
  "tpu")
    echo "Running Edge TPU"
    bash ${current_path}/scripts/tpu/tpu.sh $2
    ;;
  "ncs2")
    echo "Running Intel NCS2"
    bash ${current_path}/scripts/ncs2/ncs2.sh $2
    ;;
  "gpu")
    echo "Running Desktop GPU"
    bash ${current_path}/scripts/gpu/gpu.sh $2 gpu
    ;;
  "cpu")
    echo "Running Desktop CPU"
    bash ${current_path}/scripts/gpu/gpu.sh $2 cpu
    ;;
  *)
    usage
    ;;
esac
