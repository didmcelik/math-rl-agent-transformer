import torch
import torch.nn as nn

class TransformerPolicy(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, batch_first=False),
            num_layers
        )


    def forward(self, src):
        embedded = self.embedding(src)  # [seq_len, batch_size, d_model]
        output = self.transformer(embedded)  # [seq_len, batch_size, d_model]
        return output
