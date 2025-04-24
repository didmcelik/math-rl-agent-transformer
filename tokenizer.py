
# tokenizer.py – minimal tokenizer for symbolic math
class MathTokenizer:
    """
    A minimal tokenizer for symbolic math reinforcement learning.
    No special tokens used.
    Vocabulary: digits, x, +, -, =
    """
    def __init__(self):
        self.vocab = ['x', '+', '-', '=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, expression):
        return [self.token_to_id.get(t, self.token_to_id['0']) for t in expression]

    def decode(self, ids):
        return "".join(self.id_to_token.get(idx, '0') for idx in ids)

    def vocab_size(self):
        return len(self.vocab)
