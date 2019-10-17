from torch.nn.modules import Module
from torch import nn
import torch.nn.functional as F
import torch


class Cos(Module):
    def __init__(self, in_channels, out_channels):
        super(Cos, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels))
        self.out_channels = out_channels

    def forward(self, fea):
        fea_norm = F.normalize(fea, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        out = fea_norm.matmul(weight_norm.t())
        return out