# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/9/4 12:01
import argparse
import os

import torch

from dna_gpt.model import DNAGPT
from dna_gpt.tokenizer import KmerTokenizer
from dna_gpt.utils import seed_all_rng


def get_model(model_name):
    special_tokens = (['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] +
                      ["+", '-', '*', '/', '=', "&", "|", "!"] +
                      ['M', 'B'] + ['P'] + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] + ['W', 'Y', 'X', 'Z'])
    if model_name in ('dna_gpt0.1b_m',):
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
    else:
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)

    vocab_size = len(tokenizer)
    model = DNAGPT.from_name(model_name, vocab_size)
    return model, tokenizer


def load_model(model, weight_path, device=None, dtype=None):
    state = torch.load(weight_path, map_location="cpu")
    if 'model' in state.keys():
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)
    print(f"loadding model weights from {weight_path}")
    model.to(device=device, dtype=dtype)
    model = model.eval()
    return model


def main(args):
    torch.set_grad_enabled(False)
    seed_all_rng(args.seed)
    model_name = args.name
    weight_path = args.weight or f"checkpoints/{model_name}.pth"
    assert os.path.exists(weight_path), f"not exist checkpoint: {weight_path}"
    num_samples = args.num_samples
    prompt = args.input
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = args.dtype or 'float16'
    dtype = getattr(torch, dtype)
    model, tokenizer = get_model(model_name)
    max_len = min(args.max_len, model.max_len)
    print(f"max length is {max_len}")
    model = load_model(model, weight_path, device=device, dtype=dtype)
    prompt_ids = tokenizer.encode(prompt, max_len=max_len, device=device)
    print(f"prompt token ids: {prompt_ids.tolist()}")
    max_new_tokens = max_len - len(prompt_ids)
    x = prompt_ids[None, :]
    temperature = args.temperature
    top_k = args.topk
    for k in range(num_samples):
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k,
                           do_sample=True,
                           stop_ids=(tokenizer.unk_id, tokenizer.pad_id))
        output = tokenizer.decode(y[0].tolist())
        output = output[len(prompt):]
        print(f"\033[31m{prompt}\033[0m{output}")
        print('-' * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample dna gpt model generation')
    parser.add_argument('--input', '-i', default='<R>TGAACAGAACGGCATGTA', help='input prompt text')
    parser.add_argument('--name', '-n', default='dna_gpt0.1_m', help='type of the model')
    parser.add_argument('--num_samples', '-ns', type=int, default=10, help='path to the weight')
    parser.add_argument('--max_len', '-ml', type=int, default=128, help='path to the weight')
    parser.add_argument('--temperature', type=float, default=1.0, help='path to the weight')
    parser.add_argument('--topk', type=int, default=196, help='path to the weight')
    parser.add_argument('--seed', default=42, type=int, help='random seed for sampling')
    parser.add_argument('--weight', '-w', default=None, help='path to the weight')
    parser.add_argument('--device', default=None, help='device of the model, cuda or cpu')
    parser.add_argument('--dtype', default=None, help='dtype of the model weights')
    args = parser.parse_args()
    main(args)