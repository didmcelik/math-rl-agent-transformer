
# tokenizer_old.py – simplified for basic math RL training
class MathTokenizer:
    """
    A minimal tokenizer for symbolic math expressions.
    Only supports digits, x, +, -, = for early-stage RL training.
    """
    def __init__(self):
        self.special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        self.symbols = ['x', '+', '-', '=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.vocab = self.special_tokens + self.symbols

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def encode(self, expression, add_special_tokens=True):
        tokens = list(expression)
        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id['<bos>'])

        for tok in tokens:
            ids.append(self.token_to_id.get(tok, self.token_to_id['<unk>']))

        if add_special_tokens:
            ids.append(self.token_to_id['<eos>'])

        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for idx in ids:
            token = self.id_to_token.get(idx, '<unk>')
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return "".join(tokens)

    def vocab_size(self):
        return len(self.vocab)
