import numpy as np
import itertools as it


class KmerVocab: 

    def __init__(self, k=3, reserved_tokens=None, seq_length=None):
        if reserved_tokens is None:
            reserved_tokens = []

        self.k = k
        # N is unkown base in reference seqence
        self.bases = 'NAGCT'
        kmers = []
        for i in it.product(self.bases, repeat=self.k):
            kmers.append(list(i))
        if seq_length == 4096:
            self.kmers = [''.join(m) for m in kmers]
        elif seq_length == 512:
            self.kmers = self.get_base_kmers(self.k)
        self.idx_to_kmer = reserved_tokens + self.kmers
        self.kmer_to_idx = {token: idx for idx, token in enumerate(self.idx_to_kmer)}

    def __len__(self):
        return len(self.idx_to_kmer)

    def __getitem__(self, kmer):
        if not isinstance(kmer, (list, tuple, np.ndarray)):
            assert len(kmer) <= self.k, f"get a {len(kmer)}mer, expect {self.k}mer"
            return self.kmer_to_idx.get(kmer, self.unk)
        
        return [self.kmer_to_idx.get(token_bag, self.unk) for token_bag in kmer]
    
    def get_base_kmers(self, k):
        kmers = []
        for i in range(1, k + 1):
            kmers += list(it.product(self.bases, repeat=i))
        return [''.join(m) for m in kmers]

    def to_kmers(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_kmer[indices]
        return [self.idx_to_kmer[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0
    
    
def seq2chunk(seq, kmers):

    seq = np.array(list(seq))
    raw_seq = seq
    if len(seq) % kmers != 0:
        chunk = (len(seq)-len(seq) % kmers) // kmers
        seq = np.split(seq[:-(len(raw_seq) % kmers)], chunk)
        seq.append(raw_seq[-(len(seq) % kmers):])    
    else:
        seq = np.split(seq, len(seq) // kmers)
    seq = ["".join(chunk) for chunk in seq]

    return seq

   