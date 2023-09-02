# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/8/11 12:12
from typing import List, Union, Tuple, Any
import sys
import warnings

import numpy as np
import torch
from torch import Tensor
from .type import is_numpy, is_tensor, is_sequence, is_tensor_or_array

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]


def size(x, dim=0):
    """
    Type agnostic size.
    """
    if hasattr(x, 'shape'):
        return x.shape[dim]
    elif dim == 0:
        assert is_sequence(x), f"x must be sequence, get {type(x)}"
        return len(x)

    raise TypeError


def unsqueeze(x, dim=0):
    assert is_tensor_or_array(x), f"input must be torch tensor or numpy array"
    if is_tensor(x):
        return x.unsqueeze(dim)
    elif is_numpy(x):
        return np.expand_dims(x, axis=dim)


def cat(xs, dim: int = 0):
    x0 = xs[0]
    for y in xs[1:]:
        assert type(x0) == type(y), f"only same type object can concatenate," \
                                    f"x: {type(x0)} y: {type(y)}"
    if isinstance(x0, (list, tuple)):
        xg = (x for l in xs for x in l)
        return type(x0)(xg)
    elif isinstance(x0, str):
        return ''.join(xs)
    elif isinstance(x0, dict):
        return {k: cat([x[k] for x in xs], dim) for k in x0.keys()}
    elif is_numpy(x0):
        return np.concatenate(xs, axis=dim)
    elif is_tensor(x0):
        return torch.cat(xs, dim=dim)
    else:
        raise f"can not support type: {type(x0)}"


def unfold(x, dimension, size, step):
    if is_tensor(x):
        return x.unfold(dimension, size, step)
    else:
        warnings.warn("numpy has no unfold op, convert to torch backend", Warning)
        x = torch.from_numpy(x)
        x = x.unfold(dimension, size, step)
        return x.numpy()


def take(x, indices, dim=0):
    if is_numpy(x):
        return np.take_along_axis(x, indices, dim=dim)
    if is_tensor(x):
        return torch.take_along_dim(x, indices, dim=dim)


def repeat(x, *size):
    if is_tensor(x):
        print(size)
        if len(size) < x.dim():
            warnings.warn("repeat size dim is less than tensor dim, size will left pad 1", Warning)
            size = [1, ] * (x.dim() - len(size)) + list(size)
        return x.repeat(*size)
    else:
        return np.tile(x, size)


def repeat_interleave(input: Union[torch.Tensor, np.ndarray], repeats, dim=None):
    if is_tensor(input):
        return torch.repeat_interleave(input, repeats, dim=dim)
    else:
        return input.repeat(repeats, axis=dim)


def stack(tensors: Union[Tuple[Tensor, ...], List[Tensor]], dim=0):
    if is_tensor(tensors):
        return torch.stack(tensors, dim=dim)
    else:
        return np.stack(tensors, axis=dim)


def dim(x):
    if is_numpy(x):
        return x.ndim
    elif is_tensor(x):
        return x.dim()
    else:
        return len(x.shape)


def numel(x):
    if isinstance(x, (list, tuple)):
        return len(x)
    elif is_numpy(x):
        return np.prod(x.shape)
    elif is_tensor(x):
        return x.numel()
    else:
        raise f"cannot calculate number of elements"


def where(condition, x=None, y=None):
    if is_numpy(condition):
        return np.where(condition, x, y)
    elif is_tensor(condition):
        return torch.where(condition, x, y)
    else:
        raise f"not support type: {type(condition)}"


def exp(x):
    if is_tensor(x):
        return torch.exp(x)
    else:
        return np.exp(x)


def log(x):
    if is_tensor(x):
        return torch.log(x)
    else:
        return np.log(x)


def clip(x, min, max=None):
    if is_tensor(x):
        return torch.clip(x, min, max)
    else:
        return np.clip(x, min, max)


def maximum(x1, x2):
    assert type(x1) == type(x2)
    if is_tensor(x1):
        return torch.maximum(x1, x2)
    else:
        return np.maximum(x1, x2)


def minimum(x1, x2):
    assert type(x1) == type(x2)
    if is_tensor(x1):
        return torch.minimum(x1, x2)
    else:
        return np.minimum(x1, x2)


def argsort(x, dim=-1, descending=False):
    if is_tensor(x):
        return torch.argsort(x, dim=dim, descending=descending)
    else:
        x = np.argsort(x, axis=dim)
        if descending:
            x = x[::-1]
        return x


def sigmoid(x):
    if is_tensor(x):
        return torch.sigmoid(x)
    else:
        return 1.0 / (1.0 + np.exp(-x))


def softmax(x, dim=0):
    if is_tensor(x):
        return torch.softmax(x, dim=dim)
    else:
        return np.exp(x) / np.sum(np.exp(x), axis=dim)


def meshgrid(*tensors, indexing="xy"):
    if is_tensor(tensors[0]):
        if _TORCH_VER >= [1, 10]:
            return torch.meshgrid(*tensors, indexing=indexing)
        else:
            return torch.meshgrid(*tensors)
    else:
        return np.meshgrid(*tensors, indexing=indexing)
