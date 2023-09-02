import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..build import MODEL_REGISTRY
from ..block import MultiheadAttention
from ..comon import LayerNorm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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


@MODEL_REGISTRY.register()
class GPT(nn.Module):
    CONFIG = {
        # names follow the huggingface naming conventions
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
                 include_head=True,
                 in_head=None,
                 tensor_len=None
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_len = max_len
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.Block = Block
        self.in_head = in_head
        self.tensor_len = tensor_len
            
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
        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def forward(self, fake_data):
        bs, seq_len = fake_data.shape
        tok_emb = self.transformer.wte(fake_data)
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=fake_data.device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.transformer.wpe(pos)     # position embeddings of shape (1, tseq_len, n_embed)
        emb = tok_emb + pos_emb
        x = self.transformer.drop(emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # [bs, seq_len, n_embed]
        if self.lm_head is not None:
            x = self.lm_head(x)  # logits

        return x
    

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, LayerNorm)):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def split_params(self, cfg):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        if self.lm_head is not None:
            decay.remove('lm_head.weight')
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)
        # create the pytorch optimizer object
        return [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0}
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
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
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
