import torch
import torch.nn as nn
import torch.nn.functional as F
from .rsoftmax import rSoftMax


class SplitAttention(nn.Module):
    def __init__(self, in_channels, channels, kernel_size,
                 stride=1, padding=0,
                 radix=2, cardinality=1, reduction_factor=4):
        super().__init__()

        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels

        self.conv = nn.Conv2d(
            in_channels,
            channels * radix,
            kernel_size,
            stride,
            padding,
            groups=cardinality * radix,
            bias=False
        )

        self.bn = nn.BatchNorm2d(channels * radix)
        self.relu = nn.ReLU(inplace=True)

        inter_channels = max(in_channels * radix // reduction_factor, 32)

        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=cardinality)

        self.rsoftmax = rSoftMax(radix, cardinality)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        B, C, H, W = x.shape

        if self.radix > 1:
            splits = torch.split(x, C // self.radix, dim=1)
            gap = sum(splits)
        else:
            gap = x

        gap = F.adaptive_avg_pool2d(gap, 1)

        atten = self.fc1(gap)
        atten = self.bn1(atten)
        atten = self.relu(atten)
        atten = self.fc2(atten)

        atten = self.rsoftmax(atten.view(B, -1)).view(B, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, C // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splits)])
        else:
            out = atten * x

        return out
