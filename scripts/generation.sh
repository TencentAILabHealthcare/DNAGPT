#!/bin/bash
# example for generation
cd ..
 gpt 0.1b human genomes pretrain model
python test.py --name 'dna_gpt0.1b_h' --task 'generation' --weight 'checkpoints/dna_gpt0.1b_h.pth' --input '<R>CTGTATACCACAGA' --max_len 256 --num_samples 10

# gpt 0.1b multi-organism pretrain model
python test.py --name 'dna_gpt0.1b_m' --task 'generation' --weight 'checkpoints/dna_gpt0.1b_m.pth' --input '<R>CTGTATACCACAGA' --max_len 256 --num_samples 10

# gpt 3b multi-organism pretrain model
python test.py --name 'dna_gpt3b_m' --task 'generation' --weight 'checkpoints/dna_gpt3b_m.pth' --input '<R>CTGTATACCACAGA' --max_len 256 --num_samples 10