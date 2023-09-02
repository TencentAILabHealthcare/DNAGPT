# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/2/10
import torch
import torch.nn as nn


class MP(nn.Module):
    """Max pooling"""

    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    """Spatial pooling"""

    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool3d` and `AdaptiveMaxPool3d`."

    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or (1, 1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x),
                          self.ap(x)], dim=1)


class GlobalConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = AdaptiveConcatPool2d(1)

    def forward(self, x):
        x = self.pool(x)
        bs = x.size(0)
        x = x.view(bs, -1)

        return x


class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.ap(x)
        bs = x.size(0)
        x = x.view(bs, -1)

        return x
