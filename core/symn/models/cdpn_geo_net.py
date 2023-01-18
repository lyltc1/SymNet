import torch.nn as nn
import torch
from .resnet_backbone import resnet_spec
from detectron2.layers.batch_norm import BatchNorm2d
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init


class CDPNGeoNet(nn.Module):
    def __init__(self, cfg, in_channels, num_classes=1):
        super().__init__()
        self.concat = cfg.MODEL.BACKBONE.CONCAT
        num_filters = 256
        self.features = nn.ModuleList()
        self.features.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3,
                                                stride=2, padding=1, output_padding=1, bias=False))
        self.features.append(BatchNorm2d(num_filters))
        self.features.append(nn.ReLU(inplace=True))
        if self.concat:
            _, _, channels, _ = resnet_spec[cfg.MODEL.BACKBONE.NUM_LAYERS]
        for i in range(3):
            if self.concat or i >= 1:
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
            if self.concat:
                self.features.append(
                    nn.Conv2d(num_filters + channels[-2 - i], num_filters,
                              kernel_size=3, stride=1, padding=1, bias=False))
            else:
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))
            self.features.append(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.features.append(BatchNorm2d(num_filters))
            self.features.append(nn.ReLU(inplace=True))

        self.visib_mask_output_dim = 1 * num_classes
        self.amodal_mask_output_dim = 1 * num_classes
        self.binary_code_output_dim = 16 * num_classes

        self.features.append(
            nn.Conv2d(
                num_filters,
                self.visib_mask_output_dim + self.amodal_mask_output_dim + self.binary_code_output_dim,
                kernel_size=1,
                padding=0,
                bias=True,
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)


    def forward(self, x, x_f16=None, x_f32=None, x_f64=None):
        if self.concat:
            for i, l in enumerate(self.features):
                if i == 3:
                    x = torch.cat([x, x_f16], 1)
                elif i == 10:
                    x = torch.cat([x, x_f32], 1)
                elif i == 17:
                    x = torch.cat([x, x_f64], 1)
                x = l(x)
        else:
            for i, l in enumerate(self.features):
                x = l(x)
        visib_mask, amodal_mask, binary_code = torch.split(x, [self.visib_mask_output_dim, self.amodal_mask_output_dim, self.binary_code_output_dim], 1)
        return visib_mask, amodal_mask, binary_code
