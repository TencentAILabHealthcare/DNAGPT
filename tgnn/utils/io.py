# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Aucthor: chenchenqin
# Data: 2021/6/9
import shutil
import os

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor(tensor,
              device='cuda',
              dtype=None,
              non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_tensor(t,
                                         device=device,
                                         dtype=dtype,
                                         non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_tensor(value,
                                       device=device,
                                       dtype=dtype,
                                       non_blocking=non_blocking)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device=device,
                         dtype=dtype,
                         non_blocking=non_blocking)
    elif isinstance(tensor, np.ndarray):
        return torch.tensor(tensor, dtype=dtype, device=device)
    else:
        return tensor


def to_cuda(tensor, non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_cuda(t, non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_cuda(value, non_blocking=non_blocking)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cuda(non_blocking=non_blocking)
    else:
        return tensor


def to_cpu(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_cpu(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_cpu(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    else:
        return tensor


def to_numpy(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_numpy(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_numpy(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor


def record_stream(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(record_stream(t))
        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = record_stream(value)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.record_stream(torch.cuda.current_stream())
    else:
        return tensor


def to_size(batch_x, target_size, mode='nearest'):
    if isinstance(batch_x, (list, tuple)):
        new_tensors = []
        for t in batch_x:
            new_tensors.append(to_size(t, target_size, mode))

        return new_tensors
    elif isinstance(batch_x, dict):
        new_dict = {}
        for name, value in batch_x.items():
            new_dict[name] = to_size(value, target_size, mode)

        return new_dict
    elif isinstance(batch_x, torch.Tensor):
        batch_x = F.interpolate(batch_x, target_size, mode=mode)

        return batch_x
    else:
        # TODO: add numpy array resize
        return batch_x


def clone(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(clone(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = clone(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.clone()
    else:
        return np.copy(tensor)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def cat_files(files, output, end="\n"):
    files = list(files)
    n_files = len(files)
    if n_files == 0:
        return

    with open(output, mode="wb") as out:
        for i, path in enumerate(files):
            with open(path, mode="rb") as f:
                shutil.copyfileobj(f, out)
                if i < n_files - 1 and end:
                    out.write(end.encode())
