## Run on TPU

### Requirements
Please follow the [official setup](https://coral.ai/docs/dev-board/get-started) to flash your board. *Note that we use version **4.0 Day** in this tool*. Meanwhile, please keep update the corresponding softwares, e.g. *edgetpu_compiler* (We use *version 2.0.291256449*). Otherwise, in order to convert the original model into edge-tpu compatible model, the version of Tensorflow should >= 1.15.   

### Configuration

+ Please change the *device_ip* and *device_name* in *scripts/tpu/tpu.properties* for ssh, as the model is converted on host PC and run on TPU. By the way, please generate a pair of authentication keys for ssh login without password.
+ Due to the limitation of Edge-TPU compatible model, we develop a script to convert the original *.pb* model (Note that this script is mainly for classification model. There are lots of limitations for object detection models.). However, there may exist some models that can't be converted (Here we assume that all the object detection models can't be converted.). Please follow the ***Add Model*** part in *EDLAB/README.md*. Then, change the *model_name_edgetpu_model* value into that model file's name.

### Power Monitor

We use USB-meter (**UM25C**) to measure power. **Please pair with that device in the edgetpu board**. Meanwhile, please change the *bluetooth_device* which is the serial number of UM25C. in *scripts/tpu/tpu.properties*. 

### Commands

```shell
cd ./EDLAB
bash run.sh tpu model_name
```
