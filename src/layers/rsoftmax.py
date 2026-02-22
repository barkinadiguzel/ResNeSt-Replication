import torch
import torch.nn as nn
import torch.nn.functional as F


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        if self.radix > 1:
            B = x.size(0)
            x = x.view(B, self.cardinality, self.radix, -1)
            x = F.softmax(x, dim=2)
            x = x.view(B, -1)
        else:
            x = torch.sigmoid(x)
        return x
