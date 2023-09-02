# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/11/23 17:31
from typing import List, Optional, Callable
import torch
import torch.nn as nn



class MLP(nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., torch.nn.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer wont be used. Default: ``None``
        activation (Callable[..., torch.nn.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer wont be used. Default: ``torch.nn.ReLU``
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: List[int],
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
            inplace: Optional[bool] = True,
            bias: bool = True,
            dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)


class Linear(nn.Module):
    def __init__(self, in_channels, out_channels,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation: Optional[Callable[..., nn.Module]] = nn.ReLU,
                 dropout: float = 0.0,
                 inplace: Optional[bool] = True):
        super().__init__()
        bias = norm_layer is None
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = activation(inplace=inplace)
        self.dropout = nn.Dropout(dropout, inplace=inplace) if dropout > 0 else None

    def forward(self, x):
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x
