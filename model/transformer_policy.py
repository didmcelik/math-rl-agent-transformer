import torch.nn as nn
import torch

class TransformerPolicy(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=128,
                batch_first = False
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)  # Skor için 1 tane çıktı

    def forward(self, src):
        """
        src: (seq_len, batch_size) tensor
        """
        embedded = self.embedding(src)  # (seq_len, batch_size, d_model)
        output = self.transformer(embedded)  # (seq_len, batch_size, d_model)

        # Mean pooling: tüm tokenların ortalamasını al
        pooled = output.mean(dim=0)  # (batch_size, d_model)

        score = self.fc_out(pooled)  # (batch_size, 1)
        score = score.squeeze(-1)  # (batch_size)

        return score
