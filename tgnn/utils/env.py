# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Aucthor: chenchenqin
# Data: 2021/3/8
import logging
import os
import random
import warnings
from datetime import datetime

import torch
import torch.distributed as dist

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

import numpy as np

__all__ = ["seed_all_rng"]

TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])

"""
PyTorch version as a tuple of 2 ints. Useful for comparison.
"""


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if xm is not None:
        state["xla_rng_state"] = xm.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if xm is not None:
        xm.set_rng_state(state["xla_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (os.getpid() +
                int(datetime.now().strftime("%S%f")) +
                int.from_bytes(os.urandom(2), "big")
                )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))

    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if xm is not None:
            xm.set_rng_state(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)


def init_device(device=(0,), batch_size=None):
    if isinstance(device, (list, tuple)):
        device = ','.join([str(gid) for gid in device])
        cpu = False
    else:
        cpu = device.lower() == "cpu"
        if cpu:
            device = -1

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    cuda = not cpu and torch.cuda.is_available()
    s = f"torch {torch.__version__}"
    if cuda:
        n = torch.cuda.device_count()
        if n > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(n)):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'
    print(s)
    device = torch.device('cuda:0' if cuda else 'cpu')

    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)
    else:
        torch.cuda.set_device(device)


def init_env(device=(0,), batch_size=None, seed=42):
    init_device(device, batch_size)
    seed_all_rng(seed)
    warnings.filterwarnings('ignore')
