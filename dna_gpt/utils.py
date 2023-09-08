# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Aucthor: chenchenqin
# Data: 2021/3/8
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["seed_all_rng"]


def top_k_top_p_filter(logits, top_k: int = 0, top_p: float = 0.0):
    if top_k > 0:
        filter, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < filter[:, [-1]]] = float('-inf')

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')

    return logits


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
