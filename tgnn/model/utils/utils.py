# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/10/30 17:50
import math
import torch
import torch.nn as nn
from tgnn.utils import get_torch_version

if get_torch_version() >= [1, 10]:
    from torch.hub import load_state_dict_from_url
else:
    from torchvision.models.utils import load_state_dict_from_url


def load_filtered_state(model, state, verbose=True):
    """find layers which have same name in two networks
    and remove the layer whose size mismatch"""
    if "model" in state:
        state = state["model"]
    net_dict = model.state_dict()
    pretrain_dict = {}
    base_parameters_names = []
    for k, v in state.items():
        if k in net_dict.keys() and v.size() == net_dict[k].size():
            pretrain_dict[k] = v
            base_parameters_names.append(k)
        else:
            if verbose:
                has_key = k in net_dict.keys()
                if has_key:
                    print(
                        f"skip load size mismatch weight: {k}\t soure size: {v.size()} dest size: {net_dict[k].size()}")
                else:
                    print(f"skip missing state dict key: {k}")
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)

    return base_parameters_names


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum


def fuse_bn(conv, bn):
    kernel = conv.weight
    dim = kernel.dim() - 1

    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()

    shape = [-1, ] + [1, ] * dim
    t = (gamma / std).reshape(*shape)

    if conv.bias is not None:
        bias = conv.bias
    else:
        bias = running_mean.new_zeros(running_mean.shape)

    return kernel * t, beta + (bias - running_mean) * gamma / std


def bn_to_conv_params(bn, kernel_size, groups=1):
    running_mean = bn.running_mean
    dim = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d].index(type(bn)) + 1
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, ] * dim
    out_channels = bn.num_features
    in_channels = out_channels // groups

    kernel = torch.zeros((out_channels, in_channels) + kernel_size).to(running_mean)
    for i in range(out_channels):
        kernel[i, i % in_channels, 1, 1] = 1.0

    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()

    shape = [-1, ] + [1, ] * dim
    t = (gamma / std).reshape(**shape)

    return kernel * t, beta - running_mean * gamma / std


def fix_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def autopad(k, p=None, d=1):
    """pading to same

    Args:
        k: kernel
        p: padding
        d: dilation
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def is_parallel(model):
    """check if model is in parallel mode."""
    parallel_type = (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel,
    )
    return isinstance(model, parallel_type)


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor"""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int

    return math.ceil(x / divisor) * divisor
