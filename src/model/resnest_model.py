import torch.nn as nn
from src.backbone.resnest_stage import ResNeStStage


class ResNeSt(nn.Module):
    def __init__(self, layers, num_classes=1000,
                 radix=2, cardinality=1):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = ResNeStStage(64, 64, layers[0],
                                   radix=radix,
                                   cardinality=cardinality)

        self.layer2 = ResNeStStage(256, 128, layers[1],
                                   stride=2,
                                   radix=radix,
                                   cardinality=cardinality)

        self.layer3 = ResNeStStage(512, 256, layers[2],
                                   stride=2,
                                   radix=radix,
                                   cardinality=cardinality)

        self.layer4 = ResNeStStage(1024, 512, layers[3],
                                   stride=2,
                                   radix=radix,
                                   cardinality=cardinality)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)

        return x
