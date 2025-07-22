from pydantic import BaseModel, Field

AMINO_ACIDS = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21  
}

# Tokenizer function to a sequence of amino acids into a number
def tokenizer(protein: str, max_length=1000):
    protein = protein.upper()

    encoded = [AMINO_ACIDS.get(aa, 21) for aa in protein[:max_length]]
    
    if len(encoded) < max_length:
        encoded += [0] * (max_length - len(encoded))
    
    return encoded[:max_length]