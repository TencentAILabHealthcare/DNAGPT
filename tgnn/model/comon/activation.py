# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/3/31
from typing import Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from tgnn.utils.registry import Registry

ACTIVATION = Registry('activation')

if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SiLU(nn.Module):
    """Export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Hardswish(nn.Module):
    """
    Export-friendly version of nn.Hardswish()
    """

    @staticmethod
    def forward(x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


@ACTIVATION.register()
class GELU(nn.Module):
    def __init__(self, approximate: str = 'none') -> None:
        super(GELU, self).__init__()
        assert approximate in ['none', 'tanh', 'sigmoid']
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'sigmoid':
            return torch.sigmoid(1.702 * x) * x
        else:
            return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(self.approximate)


def get_activation(name: Union[str, nn.Module, Callable[..., nn.Module]] = "ReLU",
                   inplace=True, leak=0.1, **kwargs):
    if isinstance(name, nn.Module):
        return name

    elif isinstance(name, str):
        if name in ACTIVATION:
            module = ACTIVATION[name](**kwargs)
        elif name in ["SiLU", "ReLU", "SELU"]:
            module = getattr(nn, name)(inplace=inplace)
        elif name in ["LeakyReLU"]:
            module = getattr(nn, name)(leak, inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module

    return name(inplace=inplace)