# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/5/5
import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self,
                 in_channels=4096,
                 out_channels=150,
                 pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        for scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        input_size = x.size()[2:]
        ppm_outs = []
        for pool_up in self:
            ppm_out = pool_up(x)
            ppm_out = F.interpolate(ppm_out, input_size,
                                    mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)

        return ppm_outs


class PSPHead(nn.Module):
    def __init__(self,
                 in_channels=4096,
                 mid_channels=512,
                 num_classes=2,
                 pool_scales=(1, 2, 3, 6),
                 dropout_ratio=0.1):
        super(PSPHead, self).__init__()
        self.pool_scales = pool_scales
        self.psp_modules = PPM(self.pool_scales, in_channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * mid_channels,
                      mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if self.dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def transform_inputs(self, inputs):
        x = inputs[0]
        input_size = x.size()[2:]
        features = [x]
        for feat in inputs[1:]:
            features.append(F.interpolate(feat,
                                          input_size,
                                          mode='bilinear',
                                          align_corners=False))
        inputs = torch.cat(features, dim=1)

        return inputs

    def forward(self, features):
        x = self.transform_inputs(features)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        feat = torch.cat(psp_outs, dim=1)
        feat = self.bottleneck(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)

        return output