# SymNet
This repo provides the PyTorch implementation of our SymNet work:
”SymNet: Symmetry-aware Surface Encoding for End-to-end Instance-level Object Pose Estimation”. Not published yet.

## Environment
- CUDA >= 11.1
- torch >= 1.13.1 and torchvision >= 0.14.1

Setting up the environment can be tedious, so we've provided a [Dockerfile](./docker/Dockerfile) to simplify the process. The images contains the code already.

### Pull or Build 
Option1: Just pull the whole image from DockerHub.
```bash
docker pull lyltc1/symnet:mmcv2
```
Option2: Build the image by yourself.
Note: There are some mirror settings which need to be adapted. The default setting is for usage in China. Just remove the mirror setting if needed.
```bash
cd docker
docker build -t lyltc1/symnet:mmcv2 .
```
### Run Docker
Pay attention to the dataset and output volume.
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
--gpus all --shm-size 12G --device=/dev/dri --group-add video \
--volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" --name SymNet_mmcv2 \
-v path/to/dataset/:/home/dataset:ro \
-v path/to/output/:/home/SymNet/output:rw \
lyltc1/symnet:mmcv2
```
example:
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
--gpus all --shm-size 12G --device=/dev/dri --group-add video \
-p 8031:22 \
--volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" --name SymNet_mmcv2 \
-v /home/lyl/dataset/:/home/dataset:ro \
-v /home/lyl/git/output/:/home/SymNet/output:rw \
lyltc1/symnet:mmcv2
```

Note: if you have different folders to contain all the data needed,
you need to specify all when you run docker with ```-v```, like 

```
-v path/to/dataset_part1/:/home/dataset1:ro
-v path/to/dataset_part2/:/home/dataset2:ro
```

### UpdateCode
This is important since the code is under development.

```
cd /home/SymNet
git pull
```

## Prepare Datasets
1. Download the dataset TLESS from the [BOP benchmark](https://bop.felk.cvut.cz/datasets/). 

2. Download [VOC 2012 Train/Validation Data(1.9GB)](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) for background images.

3. Download required `XX_GT` folders of zebrapose from [owncloud](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW).
Download `tless/train_pbr_GT.zip`, `tless/test_primesense_bop_GT.zip` and `tless/train_real_GT.zip`.

4. Download pretrained_backbone from [owncloud](https://cloud.dfki.de/owncloud/index.php/s/zT7z7c3e666mJTW), note the path should be modified.

5. Download detections currently from Skype for Sandeep.

The structure of this project should look like below after using soft links, the procjet inside docker ```/home/SymNet``` should looks like:
```
# recommend using soft links (ln -sf)
/home/SymNet
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
### Training For sandeep
The experiments is using ZebraCode while with the SymNet architecture, 
the training is perfromed in one gpu for one object. So need to run 30 times.

The experiments can be run by following command:
```
# --obj_id can be replaced from --obj_id 1 to --obj_id 30
python core/symn/run_train.py --config-file configs/symn/symn_tless_config_pbr_bit16_ZebraCode.py --obj_id xxx
```
like:
```
python core/symn/run_train.py --config-file configs/symn/symn_tless_config_pbr_bit16_ZebraCode.py --obj_id 1
```

<!-- Specify the config-file and the object need to be trained, also the gpus to be used if needed.
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
--small_dataset  # Train in smaller dataset for debug
```

## Evaluation
the output of training is a fold with time saved in `SymNet/output/`
```python
python core/symn/run_evaluate.py --eval_folder output/SymNet_tless_obj4_20221225_171440
```
More args explained:
```python
``` -->

## Problems and Solve
1. python can not be found
```shell
/opt/conda/bin/conda init
```
2. cannot import cv2
```shell
pip install opencv-python-headless
# or
/opt/conda/bin/python -m pip install opencv-python-headless
```

<!-- Datasets should be prepared in ```path/to/dataset/```, so that it can be found in container. 

For me, ```path/to/dataset/``` is ```/home/lyl/dataset/```, the structure of dataset, aftar decompression:
```
/home/lyl/dataset/
    ├── pbr
        ├── tless
            ├── models
            ├── train_pbr
            ├── test_primesense
            ├── test_targets_bop19.json
    ├── VOCdevkit
        ├── VOC2012/
            ├── JPEGImages/
            ├── SegmentationClass/
            ├── SegmentationObject/
    ├── zebrapose
        ├── zebrapose_code/
            ├── tless
                ├── train_pbr_GT
                ├── train_primesense_GT
                ├── test_primesense_GT
    ├── symnet
        ├── detections
            ├── gdrnppdet-pbr/
                ├── gdrnppdet-pbr_tless-test_bed8...json
            ├── gdrnppdet-pbrreal/
            ├── zebrapose_detections/
                ├── tless
                    ├── tless_bop_pbr_only.json
    ├── pretrained_backbone
        ├──resnet
            ├──resnet34-333f7ec4.pth
``` -->
<!-- Here are some command to make the softlink to the project ```/home/SymNet/```
1. link pretrained_backbone
```
ln -s /home/dataset/pretrained_backbone/resnet/resnet34-333f7ec4.pth \
/home/SymNet/pretrained_backbone/
```
2. link bop dataset
```
ln -s /home/dataset/pbr/tless/ /home/SymNet/datasets/BOP_DATASETS/
``` 
3. link detections
```
ln -s /home/dataset/symnet/detections/* /home/SymNet/datasets/detections/
```
4. link VOC
```
ln -s /home/dataset/VOCdevkit/* /home/SymNet/datasets/VOCdevkit/
```
5. link zebrapose_code
```
ln -s /home/dataset/zebrapose/zebrapose_code/tless/ /home/SymNet/datasets/zebrapose_code/
```
6. link symnet_code (not needed for Sandeep)
```
ln -s /home/dataset/symnet/binary_code/tless/ /home/SymNet/datasets/symnet_code/
```
7. All the above path is defined in ```core/symn/MetaInfo.py```. If there exists some path error, check it or change it.


#### SoftLink for another dataset (not needed for Sandeep)
Use icbin as an example:

1. link bop dataset (not needed for Sandeep)
```
ln -s /home/dataset/pbr/icbin/ /home/SymNet/datasets/BOP_DATASETS/
``` 
2. link symnet_code (not needed for Sandeep)
```
ln -s /home/dataset/symnet/binary_code/icbin/ /home/SymNet/datasets/symnet_code/
```
3. (optional)link zebrapose_code (not needed for Sandeep)
```
ln -s /home/dataset/zebrapose/zebrapose_code/icbin/ /home/SymNet/datasets/zebrapose_code/
``` -->
