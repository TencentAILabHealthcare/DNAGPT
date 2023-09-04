# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/9/4 12:01
import itertools as it
import re
from typing import Optional, Union

import numpy as np
import torch


class KmerTokenizer:

    def __init__(self,
                 k=3,
                 reserved_tokens=None,
                 dynamic_kmer=True):
        self.k = k
        if reserved_tokens is None:
            reserved_tokens = []
        assert len(reserved_tokens) == len(set(reserved_tokens)), "duplicated token in list"
        self.reserved_tokens = [f"<{t}>" for t in reserved_tokens]
        # N is unkown base in reference seqence
        self.bases = 'NAGCT'
        self.kmers = self.get_base_kmers(self.k, dynamic_kmer=dynamic_kmer)
        self.idx_to_token = self.reserved_tokens + self.kmers
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        self.pad_id = self.token_to_idx['<P>']
        self.unk_id = 0

    def get_base_kmers(self, k, dynamic_kmer=True):
        kmers = []
        start = 1 if dynamic_kmer else k
        for i in range(start, k + 1):
            kmers += list(it.product(self.bases, repeat=i))
        return [''.join(m) for m in kmers]

    def __len__(self):
        return len(self.idx_to_token)

    def piece_to_id(self, token: str):
        return self.token_to_idx.get(token, self.unk_id)

    def id_to_piece(self, token_id):
        assert 0 <= token_id < len(self), f"out of range of token id {token_id}, max {len(self)}"
        return self.idx_to_token[token_id]

    def _encode(self, token):
        if not isinstance(token, (list, tuple, np.ndarray)):
            return self.piece_to_id(token)
        return [self._encode(t) for t in token]

    def tokenize(self, text):
        tokens = re.split(r"[<>]", text)
        new_tokens = []
        n = self.k
        for t in tokens:
            if not t:
                continue

            if f"<{t}>" in self.reserved_tokens:
                new_tokens.append(f"<{t}>")
            else:
                seq = t
                # split kmers
                chunks = [seq[i:i + n] for i in range(0, len(seq), n)]
                new_tokens += chunks
        return new_tokens

    def encode(self,
               text,
               max_len: int = -1,
               pad: bool = False,
               device: Optional[torch.device] = None,
               to_tensor=True):
        pieces = self.tokenize(text)
        tokens = self._encode(pieces)
        if max_len > 0:
            tokens = tokens[:max_len]

        if pad and len(tokens) < max_len:
            tokens += [self.pad_id] * (max_len - len(tokens))

        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def decode(self, token_ids: Union[torch.Tensor, np.ndarray]) -> str:
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()

        seq = "".join([self.id_to_piece(tid) for tid in token_ids])
        return seq