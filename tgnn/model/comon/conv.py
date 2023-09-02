# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/10/9 17:56
from functools import partial

import torch
import torch.nn as nn

from ..utils import autopad
from .activation import get_activation


class Conv(nn.Module):
    """Basic Conv
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bn=True,
                 activation="ReLU",
                 ND=2
                 ):
        super().__init__()
        padding = autopad(kernel_size, padding)
        bias = False if bn else True
        Conv = getattr(nn, f"Conv{ND}d")
        BatchNorm = getattr(nn, f"BatchNorm{ND}d")
        self.conv = Conv(in_channels, out_channels, kernel_size, stride,
                         padding=padding, dilation=dilation, groups=groups,
                         bias=bias)
        self.bn = BatchNorm(out_channels) if bn else None
        self.act = get_activation(activation) if activation is not None else None

    def forward(self, x):
        x = self.conv(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x


class BnConv(Conv):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 activation="ReLU",
                 ND=2
                 ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                         bn=True, activation=activation, ND=ND)

    def forward(self, x):
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)

        return self.conv(x)


Conv1d = partial(Conv, ND=1)
ConvBn1d = partial(Conv, ND=1, activation=None)
ConvBnRelu1d = partial(Conv, ND=1, activation='ReLU')

Conv2d = partial(Conv, ND=2)
ConvBn2d = partial(Conv, ND=2, activation=None)
ConvBnRelu2d = partial(Conv, ND=2, activation='ReLU')


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True):
        super().__init__()
        self.spatial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=autopad(kernel_size, padding),
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=bias
        )

    def forward(self, x):
        return self.point_conv(self.spatial_conv(x))


class SeparableConv(nn.Module):
    """
    Time-Channel Separable Convolution
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=None,
                 dilation=1, bias=False, bn=True,
                 activation="ReLU", ND=2):
        super().__init__()
        padding = autopad(kernel_size, padding)
        Conv = getattr(nn, f"Conv{ND}d")
        BatchNorm = getattr(nn, f"BatchNorm{ND}d")
        self.depthwise = Conv(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=bias, groups=in_channels
        )
        self.pointwise = Conv(
            in_channels, out_channels, kernel_size=1, stride=1,
            dilation=dilation, bias=bias, padding=0
        )
        self.bn = BatchNorm(out_channels) if bn else None
        self.act = get_activation(activation) if activation else None

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.bn is not None:
            x = self.bn(x)

        if self.act is not None:
            x = self.act(x)

        return x


SeparableConv1d = partial(SeparableConv, ND=1)
SeparableConv2d = partial(SeparableConv, ND=2)
