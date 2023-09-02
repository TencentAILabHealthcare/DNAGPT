# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/12/14 19:15
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from tgnn.model.build import MODEL_REGISTRY
from tgnn.model.arch.gpt import GPT
from tgnn.model.comon import LayerNorm
from utils import KmerVocab


@MODEL_REGISTRY.register()
class DNAGPT(GPT):
    """ DNAGPT gene sequence model

    References:
        1) the official GPT-2 TensorFlow implementation released by OpenAI:
        https://github.com/openai/gpt-2/blob/master/src/model.py
        2) huggingface/transformers PyTorch implementation:
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """

    def __init__(self,
                 vocab_size=7,
                 kmer = 6,
                 special_token = None, 
                 max_len=1024,
                 num_layers=3,
                 num_heads=3,
                 embedding_dim=48,
                 dropout=0.1,
                 bias=True,
                 in_head=None,
                 out_head=None,
                 gen = None,
                 out_num=None,
                 tensor_len=None
                 ):
        super().__init__(vocab_size,
                         max_len,
                         num_layers,
                         num_heads,
                         embedding_dim,
                         dropout,
                         bias=bias,
                         include_head=False,
                         in_head=in_head, 
                         tensor_len=tensor_len)
        self.in_head = in_head
        self.out_head = out_head
        if out_head[1]:
            self.num_regression = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
                nn.SiLU(inplace=True),
                LayerNorm(self.embedding_dim, bias=bias),
                nn.Linear(self.embedding_dim, 1, bias=bias)
            )
        if out_head[0]:
            self.mlm_head = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
                nn.SiLU(inplace=True),
                LayerNorm(self.embedding_dim, bias=bias),
                nn.Linear(self.embedding_dim, vocab_size, bias=bias)
            )
        if not gen: 
            self.cls_head = nn.Linear(self.embedding_dim, out_num, bias=bias)

        self.vocab = KmerVocab(kmer, special_token, max_len)

    def forward(self, fake_data):

        x = super().forward(fake_data)  # [bs, seq_len, n_embed]
        return x
