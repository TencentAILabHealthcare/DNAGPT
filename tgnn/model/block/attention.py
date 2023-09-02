# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/10/25 20:00
import math
from typing import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange


class SEBlock(nn.Module):
    """
    Ref:
       1. https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html
    """

    def __init__(self,
                 in_channels: int,
                 squeeze_channels: int,
                 activation: Callable[..., torch.nn.Module] = partial(torch.nn.ReLU, inplace=True),
                 scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
                 ND=2
                 ):
        super().__init__()
        Conv = getattr(nn, f"Conv{ND}d")

        self.avgpool = getattr(nn, f"AdaptiveAvgPool1d{ND}d")(1)
        self.down = Conv(in_channels, squeeze_channels, 1)
        self.up = Conv(squeeze_channels, in_channels, 1)
        self.in_channels = in_channels
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.down(scale)
        scale = self.activation(scale)
        scale = self.up(scale)

        return self.scale_activation(scale)

    def forward(self, inputs):
        """
        Args:
            inputs: tensor[bs, cs, seq_len]
        """
        scale = self._scale(inputs)

        return scale * inputs


SEBlock1d = partial(SEBlock, ND=1)
SEBlock2d = partial(SEBlock, ND=2)


def get_positional_features_exponential(positions, features, seq_len, min_half_life=3.):
    max_range = math.log(seq_len) / math.log(2.)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.) / half_life * positions)


def get_positional_features_central_mask(positions, features, seq_len):
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1., x) - rate * x
    log_normalization = (torch.lgamma(concentration) - concentration * torch.log(rate))
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(positions, features, seq_len, stddev=None, start_mean=None, eps=1e-8):
    if stddev is None:
        stddev = seq_len / (2 * features)

    if start_mean is None:
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev ** 2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len, feature_size, device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    feature_functions = [
        get_positional_features_exponential,
        get_positional_features_central_mask,
        get_positional_features_gamma
    ]

    num_components = len(feature_functions) * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f'feature size is not divisible by number of components ({num_components})')

    num_basis_per_class = feature_size // num_components

    embeddings = []
    for fn in feature_functions:
        embeddings.append(fn(distances, num_basis_per_class, seq_len))

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)

    return x[..., :((t2 + 1) // 2)]


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_rel_pos_features,
            heads=8,
            dim_key=64,
            dim_value=64,
            dropout=0.,
            pos_dropout=0.
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding
        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        # dropouts
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        n, h, device = x.shape[-2], self.heads, x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        q = q * self.scale
        content_logits = einsum('b h i d, b h j d -> b h i j', q + self.rel_content_bias, k)
        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h=h)
        rel_logits = einsum('b h i d, h j d -> b h i j', q + self.rel_pos_bias, rel_k)
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class MultiheadAttention(nn.Module):
    """multi-head attention like torch.nn.MultiheadAttention
    """

    def __init__(self, embedding_dim, num_heads, bias=True, dropout=0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0
        # Q, K, V
        self.c_attn = nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        # regularization
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        self.res_dropout = nn.Dropout(self.dropout)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

    def forward(self, x, attn_mask=None):
        """
        Args；
            x: tensor[bs, seq_len, embedding_dim]
            attn_mask: 掩盖掉当前时刻之后所有位置的信息

        Returns:
            tensor[bs, seq_len, embedding_dim]
        """
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.embedding_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        # pytorch build-in attention, efficient attention using Flash Attention CUDA kernels
        if hasattr(F, 'scaled_dot_product_attention'):
            assert self.dropout == 0.0, "need dropout=0.0 for now, PyTorch team is working on fix in #92917"
            y = F.scaled_dot_product_attention(q, k, v,
                                               attn_mask=None,
                                               dropout_p=self.dropout,
                                               is_causal=True)
        else:
            # self-attention (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # causal mask to ensure that attention is only applied to the left in the input sequence
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(T, T, dtype=x.dtype, device=x.device)).view(1, 1, T, T)

            att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = self.res_dropout(y)

        return y
