import sys
sys.path.append(".")

import torch

from dnagpt.gpt import DNAGPT
from utils import KmerVocab, seq2chunk
import argparse

model_names = ['dnagpt_m', 'dnagpt_s_512', 'dnagpt_b_512'] 
special_token = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] + ["+", '-', '*', '/', '=', "&", "|", "!"]+ ['M', 'B'] + ['P'] + ['R', 'I', 'K', 'L', 'O', 'Q', 'S', 'U', 'V'] + ['W', 'Y', 'X', 'Z']  

def get_model(model_name): 
    if model_name == 'dnagpt_m':
        vocab_size = len(KmerVocab(6, special_token, 4096))
        model = DNAGPT(vocab_size=vocab_size,
                        kmer = 6,
                        special_token = special_token, 
                        max_len=4096,
                        num_layers=12,
                        num_heads=12,
                        embedding_dim=768,
                        dropout=0.0,
                        bias=False,
                        in_head=[1,0,0],
                        out_head=[0,0],
                        gen = True,
                        out_num=None,
                        tensor_len=None)
        tokenizer = KmerVocab(6, special_token, 4096)
    
    if model_name == 'dnagpt_s_512':
        vocab_size = len(KmerVocab(6, special_token, 512))
        model = DNAGPT(vocab_size=vocab_size,
                        kmer = 6,
                        special_token = special_token, 
                        max_len=512,
                        num_layers=12,
                        num_heads=12,
                        embedding_dim=768,
                        dropout=0.0,
                        bias=False,
                        in_head=[1,0,0],
                        out_head=[0,0],
                        gen = True,
                        out_num=None,
                        tensor_len=None)
        tokenizer = KmerVocab(6, special_token, 512)
    
    if model_name == 'dnagpt_b_512':
        vocab_size = len(KmerVocab(6, special_token, 512))
        model = DNAGPT(vocab_size=vocab_size,
                        kmer = 6,
                        special_token = special_token, 
                        max_len=512,
                        num_layers=60,
                        num_heads=64,
                        embedding_dim=2048,
                        dropout=0.0,
                        bias=False,
                        in_head=[1,0,0],
                        out_head=[0,0],
                        gen = True,
                        out_num=None,
                        tensor_len=None)
        tokenizer = KmerVocab(6, special_token, 512)
    
    return model, tokenizer

def load_model(model, weight_path):
    state = torch.load(weight_path, map_location= torch.device('cuda'))
    if 'model' in state.keys():
        model.load_state_dict(state['model'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model 
    
        
def padding(dna, length, tokenizer):
    extend_token = tokenizer['P']
    if len(dna) <= length:
        ext = [extend_token for i in range(length-len(dna))]
        dna.extend(ext)
        return dna
    else:
        print('The input token length exceed the model capacity!!')
        
def test(input, model_name, weight_path, spec_token, cls_token, sequence_length):
    model, tokenizer = get_model(model_name)
    load_model(model, weight_path)
    input_bin = tokenizer[seq2chunk(input, 6)]
    input_bin = [tokenizer[spec_token]] + input_bin + [tokenizer[cls_token]]
    true_len = len(input_bin)
    padded_dna = padding(input_bin, sequence_length, tokenizer)
    dna_tensor = torch.Tensor(padded_dna).unsqueeze(0).long()
    x = model(dna_tensor)[:,true_len-1]
    
    return x


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description = 'test')
    
    parser.add_argument('--input_dna', type=str, help ='input dna')
    parser.add_argument('--weight_path', type=str, help ='path to the weight')
    parser.add_argument('--model_name', type=str, help ='type of the model')
    parser.add_argument('--spec_token', type=str, default='R' ,help ='special token for specific species')
    
    args = parser.parse_args()
    cls_token = 'B'
    
    
    if args.model_name == 'dnagpt_m':
        sequence_length = 4096
    
    elif args.model_name == 'dnagpt_s_512':
        sequence_length = 512
    
    elif args.model_name == 'dnagpt_b_512':
        sequence_length = 512
        
    x = test(args.input_dna, args.model_name, args.weight_path, args.spec_token, cls_token, sequence_length)
    
    print(x.shape)
        

        