import torch
import torch.nn as nn


class TransformerPolicy(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, num_layers=2):
        super(TransformerPolicy, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.embedding(x)
        # Add sequence dimension (batch_size, seq_len=1, hidden_size)
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value

#
# class TransformerPolicy(nn.Module):
#     def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=d_model,
#                 nhead=nhead,
#                 dim_feedforward=128,
#                 batch_first = False
#             ),
#             num_layers=num_layers
#         )
#         self.fc_out = nn.Linear(d_model, 1)  # Skor için 1 tane çıktı
#
#     def forward(self, src):
#         """
#         src: (seq_len, batch_size) tensor
#         """
#         embedded = self.embedding(src)  # (seq_len, batch_size, d_model)
#         output = self.transformer(embedded)  # (seq_len, batch_size, d_model)
#
#         # Mean pooling: tüm tokenların ortalamasını al
#         pooled = output.mean(dim=0)  # (batch_size, d_model)
#
#         score = self.fc_out(pooled)  # (batch_size, 1)
#         score = score.squeeze(-1)  # (batch_size)
#
#         return score
