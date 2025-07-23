from utils.tokenizer import tokenizer

# Class to test the tokenizer function
class TestTokenizer:

    #Method to test the tokenizer function
    def test_basic_tokenizer(self):

        protein = "ACDEFGHIKLMNPQRSTVWY"
        encoded = tokenizer(protein)
        assert encoded.shape == (1000, 22) # type: ignore
        expected_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for i, idx in enumerate(expected_indices):
            assert encoded[i][idx] == 1.0
            assert encoded[i].sum() == 1.0



    # Method to test tokenizer with a protein longer than max_length
    def test_tokenizer_with_long_protein(self):
        long_protein = "A" * 1500
        encoded = tokenizer(long_protein)
        assert encoded.shape == (1000, 22)

    # Method to test tokenizer with a protein shorter than max_length
    def test_tokenizer_with_short_protein(self):
        short_protein = "ACDE"
        encoded = tokenizer(short_protein)
        assert encoded.shape == (1000, 22)
        assert encoded[0][1] == 1.0  
        assert encoded[1][2] == 1.0  
        assert encoded[2][3] == 1.0 
        assert encoded[3][4] == 1.0 

if __name__ == "__main__":
    test = TestTokenizer()
    test.test_basic_tokenizer()
    test.test_tokenizer_with_long_protein()
    test.test_tokenizer_with_short_protein()
    print("All tests passed for tokenizer.")