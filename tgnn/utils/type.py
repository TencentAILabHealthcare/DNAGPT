# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/3/3
from typing import Any
from itertools import repeat
import collections

import torch
from PIL import Image
import numpy as np

try:
    import accimage
except ImportError:
    accimage = None

HALF_DTYPES = ("float16", "bfloat16")


def is_amp_dtype(dtype):
    return dtype in ("float16")


def to_torch_dtype(dtype: str):
    tdtype = getattr(torch, dtype, None)
    assert tdtype is not None, f"no exist dtype: {dtype} in torch"

    return tdtype


def is_sequence(obj):
    return isinstance(obj, collections.abc.Sequence)


def is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_tensor(x):
    return isinstance(x, torch.Tensor)


def is_numpy(x: Any) -> bool:
    return isinstance(x, np.ndarray)


def is_tensor_or_array(x):
    return is_tensor(x) or is_numpy(x)


def is_numpy_img(img: Any) -> bool:
    return img.ndim in {2, 3}


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
