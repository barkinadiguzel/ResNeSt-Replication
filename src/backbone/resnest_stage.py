import torch.nn as nn
from src.layers.bottleneck_block import Bottleneck


class ResNeStStage(nn.Module):
    def __init__(self, in_channels, channels, blocks,
                 stride=1, radix=2, cardinality=1):
        super().__init__()

        downsample = None
        if stride != 1 or in_channels != channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels,
                          channels * Bottleneck.expansion,
                          1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * Bottleneck.expansion),
            )

        layers = []
        layers.append(
            Bottleneck(in_channels, channels,
                       stride, downsample,
                       radix, cardinality)
        )

        in_channels = channels * Bottleneck.expansion

        for _ in range(1, blocks):
            layers.append(
                Bottleneck(in_channels, channels,
                           radix=radix,
                           cardinality=cardinality)
            )

        self.stage = nn.Sequential(*layers)

    def forward(self, x):
        return self.stage(x)
