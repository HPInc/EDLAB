## Run on TX2

### Requirements
Please according to [NVIDIA SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager), 
installing it to your host PC, and following its stpes to complete the installation. Don't forget to install **Tensorflow** and **TensorRT** to your TX2 by tick these item options in the SDK Manager.

### Configuration
TX2 has some configurations in [tx2.properties](tx2.properties) for itself, you can change it for your purposes. The first five items means do you need to run any types of TensorRT Quantization and only use TX2 CPU/GPU? Following items is the number of batch size and images in memory. The last item is do you need report power, "0" for NO and "1" for YES.

### Power meter
As TX2 has internal power sensor, we do not need any devices to measure its power consumption.

### Commands
``` shell script
(optional)
sudo nvpmodel -m 0
sudo jetson_clocks

cd ./EDLAB
bash run.sh tx2 <model_name>
```
