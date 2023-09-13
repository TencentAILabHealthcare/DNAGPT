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
    if model_name in ('dna_gpt0.1b_h',):
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=False)
    else:
        tokenizer = KmerTokenizer(6, special_tokens, dynamic_kmer=True)

    vocab_size = len(tokenizer)
    model = DNAGPT.from_name(model_name, vocab_size)
    return model, tokenizer


def load_model(model, weight_path, device=None, dtype=None):
    state = torch.load(weight_path, map_location="cpu")
    if 'model' in state.keys():
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    print(f"loading model weights from {weight_path}")
    model.to(device=device, dtype=dtype)
    model = model.eval()
    return model


def generate(model,
             tokenizer,
             prompt,
             max_len=256,
             num_samples=1,
             temperature=1.0,
             top_k=0,
             top_p=0):
    print(f"max length is {max_len}")
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, max_len=max_len, device=device)
    print(f"prompt token ids: {prompt_ids.tolist()}")
    max_new_tokens = max_len - len(prompt_ids)
    for k in range(num_samples):
        x = prompt_ids[None, :]
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k,
                           top_p=top_p,
                           do_sample=True,
                           stop_ids=(tokenizer.unk_id, tokenizer.pad_id))
        output = tokenizer.decode(y[0].tolist())
        output = output[len(prompt):]
        print(f"\033[31m{prompt}\033[0m{output}")
        print('-' * 50)


def regression(model, tokenizer, prompt, numbers, max_len=256):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    prompt_ids = tokenizer.encode(prompt, max_len=max_len, device=device)
    prompt_len = len(prompt_ids)
    x = prompt_ids[None]
    num_len = len(numbers)
    print(f"inputs numbers: {numbers}")
    token_length = torch.tensor(prompt_len + num_len - 1, device=device, dtype=torch.long)[None, None]
    number_block = torch.full((x.shape[0],), x.shape[1] - 2)
    num = torch.tensor(numbers, device=device, dtype=dtype)
    y = model(x, num, token_length, number_block=number_block)
    print("The predicted expression level is:", f"\033[0;31m{y[0][0, 0, 0].tolist()}\033[0m")


def classification(model, tokenizer, prompt, max_len=256):
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, max_len=max_len, device=device)[None]
    output = model(prompt_ids)[:, -1]
    y = torch.argmax(output, dim=-1)
    output = tokenizer.decode([y[0]])
    if output == 'N':
        print("\033[0;32mThe input is a real GSR sequence!\033[0m")
    elif output == 'A':
        print("\033[0;31mThe input is a fake GSR sequence!\033[0m")


def main(args):
    torch.set_grad_enabled(False)
    seed_all_rng(args.seed)
    model_name = args.name
    weight_path = args.weight or f"checkpoints/{model_name}.pth"
    assert os.path.exists(weight_path), f"not exist checkpoint: {weight_path}"
    prompt = args.input
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = args.dtype or 'float16'
    dtype = getattr(torch, dtype)
    model, tokenizer = get_model(model_name)
    model = load_model(model, weight_path, device=device, dtype=dtype)
    max_len = min(args.max_len, model.max_len)
    if args.task == "generation":
        generate(model, tokenizer, prompt,
                 max_len=max_len,
                 num_samples=args.num_samples,
                 temperature=args.temperature,
                 top_k=args.topk,
                 top_p=args.topp)
    elif args.task == "regression":
        numbers = [float(n) for n in args.numbers.split()]
        regression(model, tokenizer, prompt, numbers, max_len=max_len)
    elif args.task == "classification":
        classification(model, tokenizer, prompt, max_len=max_len)
    else:
        raise f"no task type: {args.task}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample dna gpt model generation')
    parser.add_argument('--task', default="generation", help='dtype of the model weights')
    parser.add_argument('--input', '-i', default='<R>CTGTATACCACAGA', help='input prompt text')
    parser.add_argument('--numbers', help='input number list')
    parser.add_argument('--name', '-n', default='dna_gpt0.1b_m', help='type of the model')
    parser.add_argument('--num_samples', '-ns', type=int, default=10, help='number samples of generation')
    parser.add_argument('--max_len', '-ml', type=int, default=256, help='max length of input token ids')
    parser.add_argument('--temperature', type=float, default=1.0, help='sample temperature')
    parser.add_argument('--topk', type=int, default=0, help='sample topk')
    parser.add_argument('--topp', type=float, default=0.95, help='sample topp')
    parser.add_argument('--seed', default=40, type=int, help='random seed for sampling')
    parser.add_argument('--weight', '-w', default=None, help='path to the weight')
    parser.add_argument('--device', default=None, help='device of the model, cuda or cpu')
    parser.add_argument('--dtype', default=None, help='dtype of the model weights')
    args = parser.parse_args()
    main(args)