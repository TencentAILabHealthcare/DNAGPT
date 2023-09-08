# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/2/2 12:07
import math
import numbers
from typing import Union, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ..utils import top_k_top_p_filter
_shape_t = Union[int, List[int], torch.Size]


class LayerNorm(nn.Module):
    """LayerNorm with optional bias. See description of nn.LayerNorm"""
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self,
                 normalized_shape: _shape_t,
                 eps: float = 1e-5,
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
            attn_mask: mask for masked attention

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
        y = self.c_proj(y)
        y = self.res_dropout(y)

        return y


class MLP(nn.Module):
    """feed forward"""

    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden_dim, bias=bias)
        self.gelu = nn.GELU('tanh')
        self.c_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    """Transformer block"""

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias=True):
        super().__init__()
        self.attn = MultiheadAttention(dim, num_heads, dropout=dropout, bias=bias)
        self.mlp = MLP(dim, hidden_dim=4 * dim, bias=bias, dropout=dropout)
        self.ln_1 = LayerNorm(dim, bias=bias)
        self.ln_2 = LayerNorm(dim, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    CONFIG = {
        # GPT-1
        'openai-gpt': dict(num_layers=12, num_heads=12, embedding_dim=768),  # 117M params
        # GPT-2 configs
        'gpt2': dict(num_layers=12, num_heads=12, embedding_dim=768),  # 124M params
        'gpt2-medium': dict(num_layers=24, num_heads=16, embedding_dim=1024),  # 350M params
        'gpt2-large': dict(num_layers=36, num_heads=20, embedding_dim=1280),  # 774M params
        'gpt2-xl': dict(num_layers=48, num_heads=25, embedding_dim=1600),  # 1558M params
        # Gophers
        'gopher-44m': dict(num_layers=8, num_heads=16, embedding_dim=512),
        'gpt-mini': dict(num_layers=6, num_heads=6, embedding_dim=192),
        'gpt-micro': dict(num_layers=4, num_heads=4, embedding_dim=128),
        'gpt-nano': dict(num_layers=3, num_heads=3, embedding_dim=48)
    }

    def __init__(self,
                 vocab_size,
                 max_len=1024,
                 num_layers=3,
                 num_heads=3,
                 embedding_dim=48,
                 dropout=0.0,
                 bias=True,
                 include_head=True
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.Block = Block
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.vocab_size, self.embedding_dim),
            wpe=nn.Embedding(self.max_len, self.embedding_dim),
            drop=nn.Dropout(self.dropout),
            h=nn.ModuleList(
                [Block(self.embedding_dim, self.num_heads, dropout=dropout, bias=bias) for _ in
                 range(self.num_layers)]),
            ln_f=LayerNorm(self.embedding_dim, bias=bias),
        ))
        self.lm_head = nn.Linear(self.embedding_dim, self.vocab_size, bias=bias) if include_head else None
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, input_ids):
        bs, seq_len = input_ids.shape
        tok_emb = self.transformer.wte(input_ids)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, tseq_len, n_embed)
        emb = tok_emb + pos_emb
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        # [bs, seq_len, n_embed]
        if self.lm_head is not None:
            x = self.lm_head(x)

        return x

    @torch.no_grad()
    def generate(self,
                 idx,
                 max_new_tokens,
                 temperature=1.0,
                 do_sample=False,
                 top_k=0,
                 top_p=0.,
                 stop_ids=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.max_len else idx[:, -self.max_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            # if top_k:
            #     v, _ = torch.topk(logits, top_k)
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)

            if stop_ids is not None and idx_next.item() in stop_ids:
                break

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx