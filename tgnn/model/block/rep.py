# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/11/8 12:02
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..comon import ConvBn1d, DropPath, Conv1d, get_activation
from ..block import SEBlock1d
from ..utils import fuse_bn, bn_to_conv_params, autopad


class RepConv1d(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 merged=False, use_se=False,
                 post_se=False,
                 squeeze=16,
                 activation='ReLU'):
        super(RepConv1d, self).__init__()
        self.merged = merged
        self.groups = groups
        self.in_channels = in_channels
        self.act = get_activation(activation)
        # Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
        self.pose_se = post_se
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock1d(out_channels, out_channels // squeeze)
        else:
            self.se = nn.Identity()

        if merged:
            padding = autopad(kernel_size)
            self.rbr_reparam = nn.Conv1d(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, groups=groups, bias=True)
        else:
            self.rbr_identity = None
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm1d(in_channels)

            self.rbr_dense = ConvBn1d(in_channels, out_channels, kernel_size, stride=stride, groups=groups)
            # note that kernel size and padding size
            self.rbr_1x1 = ConvBn1d(in_channels, out_channels, 1, stride=stride, groups=groups)

    def forward(self, inputs):
        if self.merged:
            out = self.rbr_reparam(inputs)
        else:
            if self.rbr_identity is None:
                id_out = 0
            else:
                id_out = self.rbr_identity(inputs)
            out = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out

        if self.pose_se:
            return self.se(self.act(out))
        else:
            return self.act(self.se(out))

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1])

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = fuse_bn(self.rbr_dense.conv, self.rbr_dense.bn)
        kernel1x1, bias1x1 = fuse_bn(self.rbr_1x1.conv, self.rbr_1x1.bn)
        kernelid, biasid = bn_to_conv_params(self.rbr_identity, self.kernel_size, self.groups)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def merge_kernel(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv1d(self.rbr_dense.conv.in_channels,
                                     self.rbr_dense.conv.out_channels,
                                     self.rbr_dense.conv.kernel_size,
                                     stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding,
                                     dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


RepPlusConv1d = partial(RepConv1d, post_se=True, squeeze=4)


class RepLKConv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 small_kernel,
                 merged=False
                 ):
        super(RepLKConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        groups = in_channels
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise,
        # you may configure padding as you wish, and change the padding of small_conv accordingly.
        if merged:
            self.lkb_reparam = nn.Conv1d(in_channels, out_channels, kernel_size, groups=groups, bias=True)
        else:
            self.lkb_origin = ConvBn1d(in_channels, out_channels, kernel_size, groups=groups)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = ConvBn1d(in_channels, out_channels, small_kernel, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            return self.lkb_reparam(inputs)

        out = self.lkb_origin(inputs)
        if hasattr(self, 'small_conv'):
            out += self.small_conv(inputs)
        return out

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            # add to the central part
            eq_k += F.pad(small_k, [(self.kernel_size - self.small_kernel) // 2] * 2)

        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(self.lkb_origin.conv.in_channels,
                                     self.lkb_origin.conv.out_channels,
                                     self.lkb_origin.conv.kernel_size,
                                     stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding,
                                     dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class ConvFFN1d(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.preffn_bn = nn.BatchNorm1d(in_channels)
        self.pw1 = ConvBn1d(in_channels, internal_channels, 1)
        self.nonlinear = nn.GELU()
        self.pw2 = ConvBn1d(internal_channels, out_channels, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)

        return x + self.drop_path(out)


class RepLKBlock1d(nn.Module):

    def __init__(self, in_channels, out_channels, block_lk_size, small_kernel, drop_path,
                 merged=False):
        super().__init__()
        self.prelkb_bn = nn.BatchNorm1d(in_channels)
        self.pw1 = Conv1d(in_channels, out_channels, 1)
        self.large_kernel = RepLKConv1d(out_channels, out_channels, block_lk_size,
                                        small_kernel=small_kernel,
                                        merged=merged)
        self.lk_nonlinear = nn.ReLU(inplace=True)
        self.pw2 = ConvBn1d(out_channels, in_channels, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        print('drop path:', self.drop_path)

    def forward(self, x):
        out = self.prelkb_bn(x)
        out = self.pw1(out)
        out = self.large_kernel(out)
        out = self.lk_nonlinear(out)
        out = self.pw2(out)

        return x + self.drop_path(out)
