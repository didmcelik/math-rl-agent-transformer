import torch
import torch.optim as optim
from model.transformer_policy import TransformerPolicy

class RLAgent:
    def __init__(self, vocab):
        self.vocab = vocab
        self.model = TransformerPolicy(vocab_size=len(vocab))
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def choose_action(self, problem_text, chain_text):
        input_text = (problem_text + " " + chain_text).strip()
        input_ids = torch.tensor(self.vocab.encode(input_text)).unsqueeze(1)
        logits = self.model(input_ids)
        last_logits = logits[-1, 0, :]
        probs = torch.softmax(last_logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        action_token = self.vocab.idx2token[action_idx.item()]
        return action_token

    def train_step(self, log_probs, rewards):
        loss = 0
        for log_prob, reward in zip(log_probs, rewards):
            loss -= log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
