from torch.nn.modules import Module
import torch


class Margin(Module):
    def __init__(self, device, margin_s, margin_m, num_classes):
        super(Margin, self).__init__()
        self.device = device
        self.margin_s = margin_s
        self.margin_m = margin_m
        self.num_classes = num_classes

    def forward(self, orin_out, labels):
        label_one_hot = torch.zeros((labels.size(0), self.num_classes),
                                    dtype=torch.float32, device=self.device)
        label_one_hot.scatter_(1, torch.unsqueeze(labels, 1), self.margin_m)
        out = orin_out - label_one_hot
        out = out * self.margin_s
        return out
