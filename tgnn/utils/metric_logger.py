# -*- coding: utf-8 -*-
# Copyright (c) 2021, Tencent Inc. All rights reserved.
# Aucthor: chenchenqin
# Data: 2021/6/9
from collections import defaultdict
from collections import deque

import numpy as np
import torch


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=1, fmt=None):
        if fmt is None:
            fmt = self.default_format()
        self.fmt = fmt
        self.deque = deque(maxlen=window_size)
        self.his = []
        self.total = 0.0
        self.count = 0

    def default_format(self):
        return "{median:.4f} ({global_avg:.4f})"

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value
        self.his.append(value)

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        d = torch.tensor(list(self.deque))
        return d.max().item()

    @property
    def value(self):
        return self.deque[-1]

    @property
    def history(self):
        return np.array(self.his)

    @property
    def global_std(self):
        return np.std(self.his)

    @property
    def global_info(self):
        return f"MEAN:{self.global_avg:.4f} STD: {self.global_std:.4f}"

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            global_avg=self.global_avg,
            avg=self.avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def stastic_info(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.global_info}")

        return self.delimiter.join(loss_str)
