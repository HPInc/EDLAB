#!/bin/bash

# Copyright (c) HP-NTU Digital Manufacturing Corporate Lab, Nanyang Technological University, Singapore.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.

source config.properties
current_path=$(cd "$(dirname $0)";pwd)
source ${current_path}/tpu.properties

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
        if [ ${model} == "classification" ]; then
            eval model_name="$""$1_name"
            echo "Runing $1 on Google Edge TPU"
            echo "Trying to convert $1.pb into Edge-TPU compatible model"
            python3 ${current_path}/classification/convertToTPUModelFromPb.py --modelpath ${current_path}/../../models/$1/${model_name} --trainingset ${current_path}/../../dataset/imagenet_test -vn 30 -mn $1
            if [ $? -ne 0 ]; then
                echo "Failed to convert the original model, trying to find provided one"
                eval edge_tpu_model="$""$1_edgetpu_model"
                if [ ${edge_tpu_model} == "none" ]; then
                    echo "Failed to convert and get the provided model"
                    exit 1
                else
                    eval model_final_path=${current_path}/../../models/$1/${edge_tpu_model}
                    if [ -f ${model_final_path} ]; then
                        echo "Found the provided model ${model_final_path}"
                    else
                        echo "No model file"
                    fi
                fi
            else
                rm -rf saved_models/
                mv TPU_models ${current_path}/classification/
                if [ $? -ne 0 ]; then
                    mv TPU_models/* ${current_path}/classification/TPU_models/
                    rm -rf TPU_models
                fi
                eval model_final_path=${current_path}/classification/TPU_models/$1_edgetpu.tflite
                echo "Convert successfully"
                echo "$1_edgetpu.tflite is saved as ${model_final_path}"
            fi
            echo "Transfering power components to EdgeTPU"
            eval device_ssh="${device_name}@${device_ip}"
            eval temp_dir="benchmarking_tpu"
            eval device_dir="${device_ssh}:/home/${device_name}/${temp_dir}/"
            echo "Creating working directory in EdgeTPU"
            ssh ${device_ssh} "mkdir $temp_dir"
            scp -r -q ${current_path}/classification/power $device_dir
            ssh ${device_ssh} "cd $temp_dir/power; chmod +x script.sh ; ./script.sh ${bluetooth_device}"
            echo "Transfering images and model to EdgeTPU"
            scp -r -q ${current_path}/../../dataset/imagenet_test $device_dir
            scp -q ${current_path}/../../dataset/imagenet_test.gtruth $device_dir
            scp -q ${model_final_path} $device_dir
            echo "Transfering source code to EdgeTPU"
            scp -q ${current_path}/classification/accuracy.py $device_dir
            scp -q ${current_path}/classification/classify_tpu_standlone.py $device_dir
            echo "Running inference on EdgeTPU"
            ssh ${device_ssh} "cd ${temp_dir}; python3.7 classify_tpu_standlone.py --data imagenet_test --model $1_edgetpu.tflite --number 1 --label imagenet_test.gtruth --modelname $1 > result.txt"
            scp -q $device_dir/result.txt .
            echo "Writing results"
            mv result.txt ${current_path}
            python3 ${current_path}/classification/result_writer.py -d ${current_path} -m $1
            rm ${current_path}/result.txt
            ssh ${device_ssh} "rm -rf ${temp_dir}"
            echo "Finished"
        elif [ ${model} == "detection" ]; then
            eval model_name="$""$1_edgetpu_model"
            eval model_final_path=${current_path}/../../models/$1/${model_name}
            if [ -f ${model_final_path} ]; then
                echo "Runing $1 on Google Edge TPU"
                echo "Transfering power components to EdgeTPU"
                eval device_ssh="${device_name}@${device_ip}"
                eval temp_dir="benchmarking_tpu"
                eval device_dir="${device_ssh}:/home/${device_name}/${temp_dir}/"
                ssh ${device_ssh} "mkdir $temp_dir"
                scp -r -q ${current_path}/classification/power $device_dir
                ssh ${device_ssh} "cd $temp_dir/power; chmod +x script.sh ; ./script.sh ${bluetooth_device}"
                echo "Transfering images and model to EdgeTPU"
                scp -q ${model_final_path} $device_dir
                scp -r -q ${current_path}/../../dataset/coco2014_test $device_dir
                scp -q ${current_path}/../../dataset/coco2014_test.gtruth $device_dir
                echo "Transfering source code to EdgeTPU"
                scp -q ${current_path}/detection/object_detection_tpu_standlone.py $device_dir
                echo "Running inference on EdgeTPU"
                ssh ${device_ssh} "cd ${temp_dir}; python3.7 object_detection_tpu_standlone.py --input coco2014_test --model ${model_name} --number 1 --label coco2014_test.gtruth > result.txt"
                scp -q $device_dir/result.txt .
                scp -q $device_dir/result_detect_tpu.json .
                python3 ${current_path}/detection/result_writer.py -m $1
                rm result.txt
                rm result_detect_tpu.json
                ssh ${device_ssh} "rm -rf ${temp_dir}"
                echo "Finished"
            else
                echo "No model file"
                exit 1
            fi
        fi
    fi
}

run_models $1