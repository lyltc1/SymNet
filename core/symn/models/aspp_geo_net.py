import torch
import torch.nn as nn
import torch.nn.functional as F


def upsample(in_channels, num_filters, kernel_size, padding, output_padding):
    upsample_layer = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            num_filters,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=output_padding,
            bias=False,
        ),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(inplace=True),
        nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(inplace=True)
    )
    return upsample_layer


class ASPPGeoNet(nn.Module):
    def __init__(self, cfg, in_channels=128, num_classes=1):
        super().__init__()
        self.concat = cfg.MODEL.BACKBONE.CONCAT
        num_filters = 256

        self.conv_1x1_1 = nn.Conv2d(in_channels, num_filters, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(num_filters)

        self.conv_3x3_1 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(num_filters)

        self.conv_3x3_2 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(num_filters)

        self.conv_3x3_3 = nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(num_filters)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(in_channels, num_filters, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(num_filters)

        self.conv_1x1_3 = nn.Conv2d(num_filters*5, num_filters, kernel_size=1)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(num_filters)

        self.upsample_1 = upsample(num_filters, num_filters, 3, 1, 1)
        if self.concat:
            self.upsample_2 = upsample(num_filters + 64, num_filters, 3, 1, 1)
        else:
            self.upsample_2 = upsample(num_filters, num_filters, 3, 1, 1)

        self.visib_mask_output_dim = 1 * num_classes
        self.amodal_mask_output_dim = 1 * num_classes
        self.binary_code_output_dim = 16 * num_classes
        if self.concat:
            self.conv_1x1_4 = nn.Conv2d(num_filters + 64, self.visib_mask_output_dim + self.amodal_mask_output_dim
                                        + self.binary_code_output_dim, kernel_size=1, padding=0)
        else:
            self.conv_1x1_4 = nn.Conv2d(num_filters, self.visib_mask_output_dim + self.amodal_mask_output_dim +
                                        self.binary_code_output_dim, kernel_size=1, padding=0)

    def forward(self, x_high_feature, x_f64=None, x_f128=None):
        feature_map_h = x_high_feature.size()[2]
        feature_map_w = x_high_feature.size()[3]

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(x_high_feature)))
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(x_high_feature)))
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(x_high_feature)))
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(x_high_feature)))

        out_img = self.avg_pool(x_high_feature)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))
        out_img = F.interpolate(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))

        if self.concat:
            x = self.upsample_1(out)
            x = torch.cat([x, x_f64], 1)
            x = self.upsample_2(x)
            x = self.conv_1x1_4(torch.cat([x, x_f128], 1))
        else:
            x = self.upsample_1(out)
            x = self.upsample_2(x)
            x = self.conv_1x1_4(x)

        visib_mask, amodal_mask, binary_code = torch.split(x, [self.visib_mask_output_dim, self.amodal_mask_output_dim, self.binary_code_output_dim], 1)
        return visib_mask, amodal_mask, binary_code
