# -*- coding: utf-8 -*-
# Copyright (c) 2022, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2022/12/14 19:15
import torch
import torch.nn as nn

from .gpt import GPT, LayerNorm


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
                 max_len=1024,
                 num_layers=3,
                 num_heads=3,
                 embedding_dim=48,
                 bias=True
                 ):
        super().__init__(vocab_size,
                         max_len,
                         num_layers,
                         num_heads,
                         embedding_dim,
                         bias=bias,
                         include_head=False)
        self.number_embedding = nn.Sequential(
            nn.Linear(1, self.embedding_dim, bias=bias),
            nn.SiLU(inplace=True),
            LayerNorm(self.embedding_dim, bias=bias),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias)
        )
        self.mlm_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
            nn.SiLU(inplace=True),
            LayerNorm(self.embedding_dim, bias=bias),
            nn.Linear(self.embedding_dim, vocab_size, bias=bias)
        )
        self.num_regression = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=bias),
            nn.SiLU(inplace=True),
            LayerNorm(self.embedding_dim, bias=bias),
            nn.Linear(self.embedding_dim, 1, bias=bias)
        )

    def _embedding_impl(self,
                        tokens,
                        numbers=None,
                        number_block=None):
        bs, _ = tokens.size()
        token_emb = self.transformer.wte(tokens)
        if numbers is not None and number_block is not None:
            numbers = numbers.to(token_emb.dtype)
            num_emb = self.number_embedding(numbers.unsqueeze(-1))
            output_tokens = []
            for j in range(bs):
                split_token = torch.split(token_emb[j], number_block[j].tolist(), dim=0)
                num_splits = len(split_token)
                stored_tokens = split_token[0]
                for i in range(num_splits - 1):
                    stored_tokens = torch.cat((stored_tokens, num_emb.unsqueeze(0)[j], split_token[i + 1]), dim=0)

                output_tokens.append(stored_tokens)

            token_emb = torch.stack(output_tokens, dim=0)

        seq_len = token_emb.shape[1]
        pos = torch.arange(0, seq_len,
                           dtype=torch.long,
                           device=token_emb.device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        emb = token_emb + pos_emb
        return emb

    def _transformer_impl(self, embeddings):
        x = self.transformer.drop(embeddings)
        for block in self.transformer.h:
            x = block(x)

        return self.transformer.ln_f(x)

    def _head_impl(self, hiddens, number_loc=None):
        mlm = self.mlm_head(hiddens)
        if number_loc is None:
            return mlm

        bs, seq_len = hiddens.shape[:2]
        total_num = []
        for i in range(bs):
            bs_num = []
            for j in range(number_loc.shape[1]):
                bs_num.append(hiddens[i, number_loc[i, j]])
            bs_num = torch.stack(bs_num, dim=0)
            total_num.append(bs_num)
        total_num = torch.stack(total_num, dim=0)  # [bs, ]
        num = self.num_regression(total_num)
        return num, mlm

    def forward(self,
                token_ids,
                numbers=None,
                number_loc=None,
                number_block=None):
        x = self._embedding_impl(token_ids, numbers, number_block)
        x = self._transformer_impl(x)
        return self._head_impl(x, number_loc)

    @classmethod
    def from_name(cls, name, vocab_size):
        model_cfgs = {
            'dna_gpt0.1b_h': dict(vocab_size=vocab_size,
                                  max_len=4096,
                                  num_layers=12,
                                  num_heads=12,
                                  embedding_dim=768,
                                  bias=False),
            'dna_gpt0.1b_m': dict(vocab_size=vocab_size,
                                  max_len=512,
                                  num_layers=12,
                                  num_heads=12,
                                  embedding_dim=768,
                                  bias=False),
            'dna_gpt3b_m': dict(vocab_size=vocab_size,
                                max_len=512,
                                num_layers=60,
                                num_heads=64,
                                embedding_dim=2048,
                                bias=False)
        }
        assert name in model_cfgs, f"unkown model name, only suport: {list(model_cfgs.keys())}"
        cfg = model_cfgs[name]
        return cls(**cfg)