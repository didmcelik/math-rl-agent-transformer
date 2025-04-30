import torch

vocab = ["multiply", "add", "carry", "and", "combine", "do", "nothing", "subtract",
         "output", "ones", "digit", "tens", "hundreds"
         ] + [str(i) for i in range(10)]
vocab = [w.lower() for w in vocab]
vocab_size = len(vocab)
vocab_dict = {word: idx for idx, word in enumerate(vocab)}


def preprocess(text):
    text = text.lower()
    for symbol in [":", "*", "+", "-", "="]:
        text = text.replace(symbol, f" {symbol} ")
    return text.split()


def encode_text(text):
    tokens = preprocess(text)
    vec = torch.zeros(vocab_size)
    for token in tokens:
        if token in vocab_dict:
            vec[vocab_dict[token]] += 1.0
    return vec


def encode_state_text(problem_text, chain_text):
    return torch.cat([encode_text(problem_text), encode_text(chain_text)])
