import torch.nn as nn
from .splat_attention import SplitAttention


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels,
                 stride=1, downsample=None,
                 radix=2, cardinality=1):
        super().__init__()

        width = channels

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = SplitAttention(
            width, width,
            kernel_size=3,
            stride=stride,
            padding=1,
            radix=radix,
            cardinality=cardinality
        )

        self.conv3 = nn.Conv2d(width, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
