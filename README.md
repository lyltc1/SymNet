# SYM-Net
This repo provides the PyTorch implementation of the work:

**SYM-Net: Symmetry aware Net.

## Overview


## Requirements
* Ubuntu 16.04/18.04/20.04/22.04, CUDA, python >= 3.6, PyTorch >= 1.6, torchvision

## Installation
One way is to set up the environment with docker.See [this](./docker/README.md)
Another way is to install the following parts.

* Install `detectron2` from [detectron2](https://github.com/facebookresearch/detectron2).
* `sh scripts/install_deps.sh`
* Download with ```git clone --recurse-submodules``` so that ```bop_toolkit``` will also be cloned.
The structure of this project should look like below:
```
.
├── asserts
├── bop_toolkit
├── ...
└── ...
```

## Datasets
recommend using soft links (ln -sf)

* Download the 6D pose datasets (LM, LM-O, YCB-V) from the [BOP website](https://bop.felk.cvut.cz/datasets/).

* Download [VOC 2012 Train/Validation Data(1.9GB)](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) for background images.

* Please also download the `binary_code` and `models_GT_color` and `detections`from here.

* Download the `detections` from here.

* Download the [pretrained resnet34 backbone](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW), save them under `./pretrained_backbone`.

The structure of this project should look like below:
```
# recommend using soft links (ln -sf)
.
├── asserts
├── bop_toolkit
├── datasets/
    ├── BOP_DATASETS
        ├──lm
        ├──lmo
        ├──ycbv
        └── ...
    ├── binary_code
    ├── models_GT_color
    ├── VOCdevkit
    ├── detections
├──pretrained_backbone/
    └── resnet34-333f7ec4.pth
├── docker
├── ...
└── ...
```


## Training symnet
`python core/symn/run_train.py <config_path> <gpu_ids> <obj_id>(other args)`

Example:

Train in one gpu：
```
python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --obj_id 4
```
Train in mulit-gpu：
```
python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --gpus 0 1 2 3 4 5 --obj_id 4
```
Train in debug mode (smaller batch size set in code):
```
python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --obj_id 4 --debug True
```

## Evaluation
`python core/symn/run_evaluate.py <eval_folder> --debug (other args)`

Example:
```
python core/symn/run_evaluate.py --eval_folder output/SymNet_tless_obj4_20221225_171440 --debug
```
## Docker

### 1.get docker image 

option1: (easiest) download docker images from [docker](docker.com).
```
docker pull lyltc1/env:cuda116-torch112-detectron2-bop-0.0.6
```
option2: (no tested) build image locally.
```
cd docker
docker build -t lyltc1/env:cuda116-torch112-detectron2-bop-0.0.6 .
```
### 2.run docker
In local mechine, prepare a folder ```dataset``` which contains all the data needed, include ```pbr``` which contains datasets like ```tless```,```ycbv``` from BOP, ```pretrained_backone```, ```binary_code```, ```VOCdevkit```. These folders need to be set soft links inside docker.
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --gpus all -p 8025:22 --shm-size 12G --device=/dev/dri --group-add video --volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" --env="QT_X11_NO_MITSHM=1" --name symnet1 -v /home/lyl/dataset/:/home/dataset:ro -v /home/lyl/git/SymNet/:/home/Symnet:rw symnet:1.0.0 /bin/bash
```
Note:The docker doesn't contain the code of Symnet, I use -v, you can also use git clone to download the code to docker:/home/Symnet.

If you have container stopped, run
```
docker exec -it CONTAINER_ID /bin/bash
```

Inside the container, check the volumes.
```
root@f6093b96bdc3:/home# ls /home
Symnet  dataset
root@f6093b96bdc3:/home# ls /home/Symnet
LICENSE  README.md  assets  configs  core  deprecated  docker  lib  ref  requirements.txt  scripts  tools
root@f6093b96bdc3:/home# ls /home/dataset
VOCdevkit  pbr  pretrained_backbone  symnet
root@f6093b96bdc3:/home# ls /home/dataset/symnet
binary_code  models_GT_color
```
### 3. soft link
```
mkdir output
mkdir /home/Symnet/datasets
mkdir /home/Symnet/datasets/BOP_DATASETS
ln -s /home/dataset/pbr/ycbv/ /home/Symnet/datasets/BOP_DATASETS/
ln -s /home/dataset/pbr/tless/ /home/Symnet/datasets/BOP_DATASETS/
ln -s /home/dataset/symnet/binary_code/ /home/Symnet/datasets/.
ln -s /home/dataset/symnet/models_GT_color/ /home/Symnet/datasets/.
ln -s /home/dataset/symnet/detections/ /home/Symnet/datasets/.
ln -s /home/dataset/VOCdevkit/ /home/Symnet/datasets/.
ln -s /home/dataset/pretrained_backbone/ /home/Symnet/
mkdir pretrained_backbone
ln -s /home/dataset/pretrained_backbone/resnet/resnet34-333f7ec4.pth /home/Symnet/pretrained_backbone/
```

### 4. modify ```MetaInfo.py```
all path of datasets is defined in core/symn/MetaInfo.py, check it or change it.
```
# untrack the file if modified locally
git update-index --assume-unchanged "core/symn/MetaInfo.py"
```
