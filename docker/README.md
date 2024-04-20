# README

## Pull or Build 
Option1: Just pull the whole image for DockerHub.
```bash
docker pull lyltc1/SymNet:mmcv2
```
Option2: Build the image by yourself.
Note: There are some mirror settings which need to be adapted. The default setting is for usage in China. Just remove the mirror setting if needed.
```bash
docker build -t lyltc1/symnet:mmcv2 .
```
## Run Docker
Pay attention to the dataset and output volume.
```
docker run -it --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all \
--gpus all --shm-size 12G --device=/dev/dri --group-add video \
--volume=/tmp/.X11-unix:/tmp/.X11-unix --env="DISPLAY=$DISPLAY" \
--env="QT_X11_NO_MITSHM=1" --name SymNet_mmcv2 \
-v path/to/dataset/:/home/dataset:ro \
-v path/to/output/:/home/SymNet/output:rw \
lyltc1/symnet:mmcv2 /bin/bash
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
lyltc1/symnet:mmcv2 /bin/bash
```

## UpdateCode
This is important since the code is under development.
```
cd /home/SymNet
git pull
```

## SoftLink
Note this part is to link the dataset and output, it's depend on your volume.
For me, the structure of dataset:
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
```
Make the softlink to the project
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

## SoftLink for another dataset (not needed for Sandeep)
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
```
