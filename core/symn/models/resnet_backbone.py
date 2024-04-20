import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from mmengine.model import normal_init, constant_init


# Specification
resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], "resnet18"),
    34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], "resnet34"),
    50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], "resnet50"),
    101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], "resnet101"),
    152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], "resnet152"),
}


class ResNetBackboneNetForCDPN(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False, concat=False):
        self.freeze = freeze
        self.concat = concat
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x.shape [bsz, 3, 256, 256]
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x_f128 = self.relu(x)
                x_low_feature = self.maxpool(x_f128)
                x_f64 = self.layer1(x_low_feature)
                x_f32 = self.layer2(x_f64)
                x_f16 = self.layer3(x_f32)
                x_f8 = self.layer4(x_f16)
                if self.concat:
                    return x_f8.detach(), x_f16.detach(), x_f32.detach(), x_f64.detach()
                else:
                    return x_f8.detach()
        else:
            x = self.conv1(
                x
            )  # x.shape [bsz, 3, 256, 256] -> x.shape [bsz, 64, 128, 128]
            x = self.bn1(x)
            x_f128 = self.relu(x)
            x_low_feature = self.maxpool(x_f128)  # x.shape [bsz, 64, 64, 64]
            x_f64 = self.layer1(x_low_feature)  # x.shape [bsz, 64, 64, 64]
            x_f32 = self.layer2(x_f64)  # x.shape [bsz, 128, 32, 32]
            x_f16 = self.layer3(x_f32)  # x.shape [bsz, 256, 16, 16]
            x_f8 = self.layer4(x_f16)  # x.shape [bsz, 512, 8, 8]
            if self.concat:
                return x_f8, x_f16, x_f32, x_f64
            else:
                return x_f8


class MyBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(MyBasicBlock, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(
            in_channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.downsample(x)
        out = F.relu(out)
        return out


def my_make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (
        num_blocks - 1
    )  # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])
    blocks = []
    for stride in strides:
        blocks.append(
            block(
                in_channels=in_channels,
                channels=channels,
                stride=stride,
                dilation=dilation,
            )
        )
        in_channels = block.expansion * channels
    layer = nn.Sequential(*blocks)
    return layer


class ResNetBackboneNetForASPP(nn.Module):
    def __init__(self, block, layers, in_channel=3, freeze=False, concat=False):
        self.freeze = freeze
        self.concat = concat
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.new_layer3 = my_make_layer(
            MyBasicBlock,
            in_channels=128,
            channels=256,
            num_blocks=6,
            stride=1,
            dilation=2,
        )
        self.new_layer4 = my_make_layer(
            MyBasicBlock,
            in_channels=256,
            channels=512,
            num_blocks=3,
            stride=1,
            dilation=4,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):  # x.shape [bsz, 3, 256, 256]
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x_f128 = self.relu(x)
                x_low_feature = self.maxpool(x_f128)
                x_f64 = self.layer1(x_low_feature)
                x_f32 = self.layer2(x_f64)
                x_f32 = self.new_layer3(x_f32)
                x_f32 = self.new_layer4(x_f32)
                if self.concat:
                    return x_f32.detach(), x_f64.detach(), x_f128.detach()
                else:
                    return x_f32.detach()
        else:
            x = self.conv1(
                x
            )  # x.shape [bsz, 3, 256, 256] -> x.shape [bsz, 64, 128, 128]
            x = self.bn1(x)
            x_f128 = self.relu(x)
            x_low_feature = self.maxpool(x_f128)  # x.shape [bsz, 64, 64, 64]
            x_f64 = self.layer1(x_low_feature)  # x.shape [bsz, 64, 64, 64]
            x_f32 = self.layer2(x_f64)  # x.shape [bsz, 128, 32, 32]
            x_f32 = self.new_layer3(x_f32)
            x_f32 = self.new_layer4(x_f32)
            if self.concat:
                return x_f32, x_f64, x_f128
            else:
                return x_f32
