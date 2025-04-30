import torch
import torch.nn as nn
import torch.optim as optim
import random
import tkinter as tk

#############################################
# Vocabulary and Encoding
#############################################

vocab = [
    "multiply", "carry", "and", "combine", "do", "nothing", "subtract",
    "output", "ones", "digit", "tens", "hundreds"
] + [str(i) for i in range(10)]
vocab = [w.lower() for w in vocab]
vocab_dict = {word: idx for idx, word in enumerate(vocab)}


def preprocess(text):
    text = text.lower()
    for symbol in [":", "*", "+", "-", "="]:
        text = text.replace(symbol, f" {symbol} ")
    return text.split()


def encode_text(text):
    tokens = preprocess(text)
    ids = [vocab_dict[token] for token in tokens if token in vocab_dict]
    return torch.tensor(ids, dtype=torch.long)

#############################################
# Multiplication Environment
#############################################

class MultiplicationEnvThree:
    def __init__(self):
        self.reset()

    def reset(self):
        A = random.randint(100, 999)
        M = random.randint(1, 9)
        self.correct_answer = A * M
        self.problem_text = f"mul: multiply {A} x {M}"

        ones = A % 10
        tens = (A // 10) % 10
        hundreds = A // 100

        P1 = ones * M
        d1 = P1 % 10
        carry1 = P1 // 10

        P2 = tens * M + carry1
        d2 = P2 % 10
        carry2 = P2 // 10

        P3 = hundreds * M + carry2

        self.correct_steps = [
            f"multiply {ones} by {M}",
            f"multiply {tens} by {M} with carry {carry1}", # todo add carry
            f"multiply {hundreds} by {M} with carry {carry2}",
            f"output ones digit: {d1}",
            f"output tens digit: {d2}",
            f"output hundreds digits: {P3}"
        ]
        self.allowed_actions = self.correct_steps + ["dummy"]
        self.chain = []
        return self.problem_text, ""

    def step(self, action_idx):
        action = self.allowed_actions[action_idx.item()]  # 🔥 item() eklendi
        self.chain.append(action)
        done = len(self.chain) >= len(self.correct_steps)
        reward = 5.0 if (len(self.chain) <= len(self.correct_steps) and self.chain[-1] == self.correct_steps[
            len(self.chain) - 1]) else -2.0
        return (self.problem_text, " ".join(self.chain)), reward, done


#############################################
# Transformer Policy Network
#############################################

class TransformerPolicy(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=False),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def encode_sequence(self, token_ids):
        x = self.embedding(token_ids)
        x = self.transformer(x)
        pooled = x.mean(dim=0)
        return pooled

    def forward(self, state_tokens, candidate_actions):
        state_emb = self.encode_sequence(state_tokens)
        scores = []
        for action_tokens in candidate_actions:
            action_emb = self.encode_sequence(action_tokens)
            combined = state_emb + action_emb
            score = self.fc_out(combined).squeeze(-1)
            scores.append(score)
        scores = torch.stack(scores)
        return scores

#############################################
# RL Agent
#############################################

class RLAgent:
    def __init__(self, vocab_size, lr=1e-4):
        self.policy = TransformerPolicy(vocab_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_probs = []
        self.loss = 0

    def choose_action(self, state, allowed_actions):
        problem_text, chain_text = state
        state_tokens = encode_text(problem_text + " " + chain_text)
        action_tokens = [encode_text(a) for a in allowed_actions]
        state_tokens = state_tokens.unsqueeze(1)  # (seq_len, batch=1)
        action_tokens = [a.unsqueeze(1) for a in action_tokens]

        scores = self.policy(state_tokens, action_tokens)  # (7,) veya (7,1)

        if scores.dim() == 2:
            scores = scores.squeeze(-1)  # 🔥 Burayı ekliyoruz: (7,1) -> (7,)

        # NaN oluşmasını önlemek için:
        scores = scores - scores.max()
        probs = torch.softmax(scores, dim=0)

        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.size(0)  # 🔥 Eğer hala NaN varsa uniform dağılım!
            probs.requires_grad_(True)  # 🔥 yeniden requires_grad aç
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action

    def accumulate_loss(self, reward):
        if self.log_probs:
            self.loss += -self.log_probs[-1] * reward

    def finalize_episode(self, final_reward):
        bonus = sum([-lp * final_reward for lp in self.log_probs])
        self.loss += bonus
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.log_probs = []
        self.loss = 0

#############################################
# Training
#############################################

def train_agent(num_episodes=5000):
    env = MultiplicationEnvThree()
    agent = RLAgent(vocab_size=len(vocab))

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state, env.allowed_actions)
            next_state, reward, done = env.step(action)
            agent.accumulate_loss(reward)
            total_reward += reward
            state = next_state

        agent.finalize_episode(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Reward: {total_reward}")

    return agent

#############################################
# Main
#############################################

if __name__ == "__main__":
    print("Training RL agent for three-digit multiplication...")
    trained_agent = train_agent(num_episodes=20000)
    print("Training complete.")
