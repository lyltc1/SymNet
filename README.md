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

The structure of this project should look like below after using soft links, the procjet inside docker should looks like:
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

## Training
Specify the config-file and the object need to be trained, also the gpus to be used if needed.
Train in one gpu：
```python
# train tless-obj01 in one gpu
python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --obj_id 4
```
Train in mulit-gpu：
```python
# train tless-obj04 in six gpus
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --gpus 0 1 2 3 4 5 --obj_id 4
# train ycbv-obj01 in eight gpus, train 10bits SymCode in pbr setting
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python core/symn/run_train.py --config-file configs/symn/ycbv/symn_ycbv_config_bit10_pbr.py --gpus 0 1 2 3 4 5 6 7 --obj_id 1
```
Some more args explained:
```python
--debug True  # Train in smaller batch size set in code
```

## Evaluation
the output of training is a fold with time saved in `SymNet/output/`
```python
python core/symn/run_evaluate.py --eval_folder output/SymNet_tless_obj4_20221225_171440
```
More args explained:
```python
```

<!-- 
# untrack the file if modified locally
git update-index --assume-unchanged "core/symn/MetaInfo.py"
``` -->
