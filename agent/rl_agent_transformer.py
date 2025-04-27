import torch
import torch.optim as optim
from model.transformer_policy import TransformerPolicy
from utils.vocab import Vocab

class RLAgentTransformer:
    def __init__(self, vocab, d_model=128, nhead=4, num_layers=2, lr=1e-4):
        self.vocab = vocab
        self.transformer = TransformerPolicy(vocab_size=len(vocab), d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=lr)
        self.episode_log_probs = []
        self.cumulative_loss = 0.0

    def encode(self, text):
        return torch.tensor(self.vocab.encode(text)).unsqueeze(1)

    def choose_action(self, problem_text, chain_text, allowed_actions):
        input_tensor = torch.tensor(self.vocab.encode(problem_text + " " + chain_text)).unsqueeze(1)
        logits = self.transformer(input_tensor)
        final_hidden = logits[-1, 0, :]  # Last token output

        scores = []
        for action_text in allowed_actions:
            action_ids = torch.tensor(self.vocab.encode(action_text)).unsqueeze(1)
            embedded = self.transformer.embedding(action_ids)
            if embedded.dim() == 3:
                embedded = embedded.squeeze(1)
            action_representation = embedded.mean(dim=0)
            score = torch.dot(final_hidden, action_representation)
            scores.append(score)

        scores_tensor = torch.stack(scores)
        probs = torch.softmax(scores_tensor, dim=0)
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
