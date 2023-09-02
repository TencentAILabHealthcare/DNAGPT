# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/2/2 12:07
import torch
import torch.nn as nn
import torch.nn.functional as F

import numbers
from torch.nn.parameter import Parameter
from typing import Union, List, Tuple

_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. See description of nn.LayerNorm"""
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs)) if bias else None
        else:
            assert not bias, f"when elementwise_affine = False, bias item is None"
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
               'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization

    Ref：
        1.Root Mean Square Layer Normalization
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, hidden_states):
        hidden_states = self._norm(hidden_states).type_as(hidden_states)

        return self.weight * hidden_states


NORM_LAYERS = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    LayerNorm,
    RMSNorm
)
