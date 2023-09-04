# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Aucthor: chenchenqin
# Data: 2021/3/8
import os
import random
from datetime import datetime

import numpy as np
import torch

__all__ = ["seed_all_rng"]


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
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
