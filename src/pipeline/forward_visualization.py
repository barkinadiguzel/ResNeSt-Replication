import torch
import torch.nn.functional as F


def forward_visualization(x, radix=2, cardinality=2):
    B, C, H, W = x.shape
    print("Input:", x.shape)

    groups = radix * cardinality
    assert C % groups == 0, "C, radix*cardinality'ye bölünmeli"

    x = x.view(B, cardinality, radix, C // groups, H, W)
    print("After KR split:", x.shape)

    gap = x.mean(dim=(3, 4, 5))   
    print("After GAP:", gap.shape)

    attn = F.softmax(gap, dim=2)  
    print("Attention weights:", attn.shape)

    attn = attn.view(B, cardinality, radix, 1, 1, 1)

    weighted = x * attn
    print("After weighting:", weighted.shape)

    out = weighted.sum(dim=2)  #
    print("After sum over radix:", out.shape)

    out = out.view(B, C, H, W)
    print("Final output:", out.shape)

    return out


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    forward_visualization(x, radix=2, cardinality=2)
