## Run on GPU

### Requirements
To fully utilize the GPU, you need to install CUDA and cuDnn at first. Follow the [CUDA install guidence](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) to install CUDA and verify the installation step by step. 
Some necessary python modules are list as follows:  
numpy==1.17.3  
Pillow==3.3.0  
tensorflow-gpu==1.14.0  
csv  
argparse  

### Configuration
GPU configuration is listed in [gpu.properties](gpu.properties). Please change it to facilitate your needs. The three features you can set are: 
batchsize  
loadnumber  
report_power: '0' or '1' // Yes or No  

### Power meter
Some GPUs have their internal power sensors, try "nvidia-smi --query-gpu=power.draw" to get the power.  

### Commands
``` shell script 
cd ./EDLAB 
bash run.sh gpu <model_name> 
```
