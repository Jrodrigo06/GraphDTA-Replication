# utils/tokenizer.py
import torch


AMINO_ACIDS = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X' : 21  
}

VOCAB_SIZE = 22

# Tokenizer function to a sequence of amino acids into a one-hot encoded tensor
def tokenizer(protein: str, max_length: int = 1000):
    '''
    Tokenizes a protein sequence into a one-hot encoded tensor.
    
    Args:
        protein (str): Protein sequence.
        max_length (int): Maximum length of the sequence. Sequences longer than this will be truncated, and shorter ones will be padded.
        Returns:
            torch.Tensor: One-hot encoded tensor (long) of shape (max_length, VOCAB_SIZE).'''
    protein = protein.upper()

    indices = [AMINO_ACIDS.get(aa, 21) for aa in protein[:max_length]]

    #Padding for short sequences
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))
    
    tensor = torch.tensor(indices, dtype=torch.long) 

    one_hot_encoded = torch.nn.functional.one_hot(tensor, num_classes=VOCAB_SIZE).float()
    
    return one_hot_encoded
