class Vocab:
    def __init__(self):
        self.token2idx = {}
        self.idx2token = {}
        self.build_vocab()

    def build_vocab(self):
        base_tokens = ["multiply", "by", "output", "carry", "final", "result", "do", "nothing"]
        numbers = [str(i) for i in range(100)]
        tokens = base_tokens + numbers
        for idx, token in enumerate(tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def encode(self, text):
        tokens = text.lower().split()
        return [self.token2idx.get(token, 0) for token in tokens]

    def decode(self, indices):
        return " ".join([self.idx2token.get(idx, "<unk>") for idx in indices])

    def __len__(self):
        return len(self.token2idx)
