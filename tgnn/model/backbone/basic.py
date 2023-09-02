# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/12/19 14:36
import torch
import torch.nn as nn

from ..block import BasicBlock, BasicStem
from ..utils import autopad


class BasicBackbone(nn.Module):

    def __init__(self,
                 in_channels=1,
                 depths=(1, 3, 3, 3),
                 channels=(64, 128, 256, 512),
                 kernel_sizes=(3, 3, 3, 3),
                 strides=(1, 2, 2, 2),
                 ND=2):
        super(BasicBackbone, self).__init__()
        self.in_channels = in_channels
        assert len(depths) == len(channels), f"number of backone stages must be equal to number of depth list"
        self.channels = channels
        self.stages = nn.ModuleList()
        self.stages.append(BasicStem(in_channels, channels[0], kernel_sizes[0], strides[0], ND=ND))
        Conv = getattr(nn, f"Conv{ND}d")

        in_channels = channels[0]
        for out_channels, kernel_size, stride, depth in zip(channels[1:], kernel_sizes[1:], strides[1:], depths[1:]):
            stage = [
                Conv(in_channels, out_channels, kernel_size, stride=stride,
                     padding=autopad(kernel_size), bias=False), ]
            for i in range(depth):
                stage.append(BasicBlock(out_channels, out_channels, kernel_size, ND=ND))
            self.stages.append(nn.Sequential(*stage))
            in_channels = out_channels

    def forward(self, x, return_features=False):
        outputs = []
        for s in self.stages:
            x = s(x)
            outputs.append(x)

        if return_features:
            return outputs

        return x
