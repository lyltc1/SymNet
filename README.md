# SymNet
This repo provides the PyTorch implementation of our SymNet work:
”SymNet: Symmetry-aware Surface Encoding for End-to-end Instance-level Object Pose Estimation”. Not published yet.

## Environment
- CUDA >= 11.1
- torch >= 1.13.1 and torchvision >= 0.14.1

Setting up the environment can be tedious, so we've provided a Dockerfile to simplify the process. Please refer to the [README](./docker/README.md) in the Docker directory for more information.

## Datasets
1. Download the dataset TLESS from the [`BOP benchmark`](https://bop.felk.cvut.cz/datasets/). 

2. Download [VOC 2012 Train/Validation Data(1.9GB)](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) for background images.

3. Download required `XX_GT` folders of zebrapose from [`owncloud`](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW).
Download `tless/train_pbr_GT.zip`, `tless/test_primesense_bop_GT.zip` and `tless/train_real_GT.zip`.

4. Download pretrained_backbone from [`owncloud`](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW), note the path should be modified.

The structure of this project should look like below after using soft links:
```
# recommend using soft links (ln -sf)
Symnet
├── asserts
├── bop_toolkit
├── datasets
    ├── BOP_DATASETS
        ├──tless
            ├── models
            ├── models_cad
            ├── models_eval
            ├── train_primesense
            ├── train_pbr
            ├── test_primesense
            ├── test_targets_bop19.json
    ├── zebrapose_code
        ├── tless
            ├── train_pbr_GT (unzip from `train_pbr_GT.zip`)
                ├── 000000
                    ├── 000000_000000.png
                    ├── ...
            ├── test_primesense_GT (unzip from `test_primesense_bop_GT.zip`)
            ├── train_primesense_GT (unzip from `train_real_GT.zip`)
    ├── symnet_code (not needed for sandeep's ablation study)
        ├── tless (not needed for sandeep's ablation study)
            ├── train_pbr_GT (unzip from `train_pbr_GT.zip`) (not needed for sandeep's ablation study)
    ├── detections
            ├── gdrnppdet-pbr/
                ├── gdrnppdet-pbr_tless-test_bed8...json
            ├── gdrnppdet-pbrreal/
            ├── zebrapose_detections/
                ├── tless
                    ├── tless_bop_pbr_only.json
    ├── VOCdevkit
            ├── VOC2012/
                ├── JPEGImages/
                ├── SegmentationClass/
                ├── SegmentationObject/
├──pretrained_backbone/
    └── resnet34-333f7ec4.pth
├── docker
├── ...
└── ...
```


<!-- 

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
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 1

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
``` -->
