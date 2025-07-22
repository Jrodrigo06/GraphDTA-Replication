from utils.tokenizer import tokenizer

# Class to test the tokenizer function
class TestTokenizer:

    #Method to test the tokenizer function
    def test_basic_tokenizer(self):

        protein = "ACDEFGHIKLMNPQRSTVWY"
        encoded = tokenizer(protein)
        assert len(encoded) == 1000
        assert encoded[:20] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Method to test tokenizer with a protein longer than max_length
    def test_tokenizer_with_long_protein(self):
        long_protein = "A" * 1500
        encoded_long = tokenizer(long_protein)
        assert len(encoded_long) == 1000
        assert encoded_long[:20] == [1] * 20

    # Method to test tokenizer with a protein shorter than max_length
    def test_tokenizer_with_short_protein(self):
        short_protein = "ACDE"
        encoded_short = tokenizer(short_protein)
        assert len(encoded_short) == 1000
        assert encoded_short[:4] == [1, 2, 3, 4]

if __name__ == "__main__":
    test = TestTokenizer()
    test.test_basic_tokenizer()
    test.test_tokenizer_with_long_protein()
    test.test_tokenizer_with_short_protein()
    print("All tests passed for tokenizer.")