# model.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Implements standard positional encoding using sine and cosine functions.
    Adds position information to token embeddings.
    """

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class TinyTransformer(nn.Module):
    """
    A minimal transformer encoder-based model for symbolic token generation.
    Used as a policy network in math RL environments.
    """

    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        """
        Args:
            src: Tensor of shape (batch_size, seq_len)
        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(src)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)  # add positional encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        logits = self.fc_out(x)  # (batch, seq_len, vocab_size)
        return logits
