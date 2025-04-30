import torch
import torch.optim as optim
from model.transformer_policy import TransformerPolicy
import torch.nn as nn




class RLAgentTransformer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.model = TransformerPolicy(vocab_size=len(vocab))
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.episode_log_probs = []
        self.cumulative_loss = 0.0

    def choose_action(self, problem_text, chain_text, allowed_actions):
        input_text = (problem_text + " " + chain_text).strip()

        input_list = []
        for action_text in allowed_actions:
            full_input = input_text + " " + action_text
            input_ids = torch.tensor(self.vocab.encode(full_input), dtype=torch.long)
            input_list.append(input_ids)

        # Şimdi hepsini padding ile eşitle
        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            input_list, batch_first=False, padding_value=self.vocab.token2idx["<pad>"]
        )  # (seq_len, batch_size)

        # Model artık tüm action'ları birlikte işliyor
        scores = self.model(padded_inputs)  # (batch_size,)

        probs = torch.softmax(scores, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_index = dist.sample()

        self.episode_log_probs.append(dist.log_prob(action_index))

        return action_index.item()

    def accumulate_loss(self, reward):
        if self.episode_log_probs:
            self.cumulative_loss += -self.episode_log_probs[-1] * reward

    def finalize_episode(self, final_reward):
        bonus = sum([-lp * final_reward for lp in self.episode_log_probs])
        self.cumulative_loss += bonus
        self.optimizer.zero_grad()
        self.cumulative_loss.backward()
        self.optimizer.step()
        self.episode_log_probs = []
        self.cumulative_loss = 0.0