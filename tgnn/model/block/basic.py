# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/10/10 17:19
from functools import partial
from typing import Any, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _ntuple
from torch import Tensor

from .attention import SEBlock
from ..utils import autopad, make_divisible
from ..comon import Conv, BnConv


class BasicStem(nn.Sequential):
    def __init__(self, in_channels, out_channels, depth=1, kernel_size=3, stride=1, ND=2):
        super().__init__()
        Conv = getattr(nn, f"Conv{ND}d")
        self.append(Conv(in_channels,
                         out_channels,
                         kernel_size,
                         stride=stride,
                         padding=autopad(kernel_size, None),
                         bias=False))
        for i in range(depth):
            self.append(BasicBlock(out_channels, out_channels, kernel_size, ND=ND))


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 ND=2):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size, stride=stride, ND=ND)
        self.conv2 = Conv(out_channels, out_channels, kernel_size, activation=None, ND=ND)
        self.downsample = None
        if stride != 1:
            self.downsample = Conv(out_channels, out_channels, 1,
                                   stride=stride, activation=None, ND=ND)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


BasicBlock1d = partial(BasicBlock, ND=1)
BasicBlock2d = partial(BasicBlock, ND=2)


class BasicBlockPre(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: int = 3,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 ND=2):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = BnConv(in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            dilation=dilation, ND=ND)
        self.conv2 = BnConv(out_channels, in_channels,
                            kernel_size=kernel_size,
                            dilation=dilation, ND=ND)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        x += identity

        return x


class Bottleneck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 dilation: int = 1,
                 base_width: int = 64,
                 ND=2):
        super().__init__()
        expansion: int = 4
        width = int(out_channels // expansion * (base_width / 64.0)) * groups
        self.conv1 = Conv(in_channels, width, 1, ND=ND)
        self.conv2 = Conv(width, width, kernel_size, stride, groups=groups,
                          dilation=dilation, ND=ND)
        self.conv3 = Conv(width, out_channels * self.expansion, 1, activation=None, ND=ND)
        self.downsample = None
        if stride != 1:
            self.downsample = Conv(out_channels, out_channels, 1,
                                   stride=stride, activation=None, ND=ND)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


Bottleneck1d = partial(Bottleneck, ND=1)
Bottleneck2d = partial(BasicBlock, ND=2)


class InvertedResidual(nn.Module):

    def __init__(self,
                 in_channels: int,
                 expanded_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 width_mult: float,
                 use_se: bool,
                 activation: Callable[..., nn.Module] = partial(nn.Hardswish, inplace=True),
                 se_layer: Callable[..., nn.Module] = partial(SEBlock, scale_activation=nn.Hardsigmoid),
                 ND=2
                 ):
        super().__init__()
        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")
        stride = 1 if dilation > 1 else stride
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.in_channels = self.adjust_channels(in_channels, width_mult)
        self.kernel_size = kernel_size
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.stride = stride
        self.dilation = dilation

        layers: List[nn.Module] = []
        # expand
        if self.expanded_channels != self.in_channels:
            layers.append(
                Conv(self.in_channels,
                     self.expanded_channels,
                     kernel_size=1,
                     activation=activation,
                     ND=ND)
            )
        # depthwise
        layers.append(
            Conv(
                self.expanded_channels,
                self.expanded_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                groups=self.expanded_channels,
                activation=activation,
                ND=ND
            )
        )
        if use_se:
            squeeze_channels = self.adjust_channels(expanded_channels, 0.25)
            layers.append(se_layer(self.expanded_channels, squeeze_channels))
            # project
            layers.append(
                Conv(self.expanded_channels, self.out_channels, kernel_size=1, activation=None, ND=ND)
            )
        self.block = nn.Sequential(*layers)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return make_divisible(channels * width_mult, 8)

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)

        if self.use_res_connect:
            result += input

        return result
