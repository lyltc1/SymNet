# SYM-Net
This repo provides the PyTorch implementation of the work:

**SYM-Net: Symmetry aware Net.

## Overview


## Requirements
* Ubuntu 16.04/18.04/20.04/2204, CUDA, python >= 3.6, PyTorch >= 1.6, torchvision

## Installation
One way is to set up the environment with docker.See [this](./docker/README.md)
Another way is to install the following parts.

* Install `detectron2` from [detectron2](https://github.com/facebookresearch/detectron2).
* `sh scripts/install_deps.sh`
* Download [`bop_toolkit`](https://github.com/thodan/bop_toolkit) in project root. 
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

* Download the 6D pose datasets (LM, LM-O, YCB-V) from the
[BOP website](https://bop.felk.cvut.cz/datasets/).

* Download[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.

* Please also download the `binary_code` and `models_GT_color` and `detections`from here.

* Download the [`pretrained resnet`](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW), 
save them under `./pretrained_backbone`.

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
`python core/symn/run_evaluate.py <config_path><ckpt_path> --debug (other args)`

Example:
```
python core/symn/run_evaluate.py \
--config-file output/SymNet_tless_obj4_20221225_171440/symn_tless_obj04_R_allo_sym.py \
--ckpt output/SymNet_tless_obj4_20221225_171440/epoch=282-step=113482.ckpt \
--debug
```
