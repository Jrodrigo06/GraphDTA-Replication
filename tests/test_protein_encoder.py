import torch
from protein.protein_encoder import ProteinEncoder
from utils.tokenizer import tokenizer  

def test_protein_encoder():
    seqs = ["ACDE", "VVVVV", "M"]
    batch = torch.stack([tokenizer(seq) for seq in seqs])

    enc = ProteinEncoder()
    out = enc(batch)
    assert out.shape == (len(seqs), 128)