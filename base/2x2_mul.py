
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class SimpleMultiplicationEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # 2 basamaktan oluşan bir sayı ve çarpan seçiyoruz
        self.number1 = random.randint(10, 99)  # 2 basamaktan oluşan bir sayı
        self.number2 = random.randint(10, 99)  # 2. sayı da 2 basamaktan
        self.correct_answer = self.number1 * self.number2
        self.steps = []
        self.current_step = 0

        # Initialize all result variables with 0
        self.ones_result = 0
        self.tens_result = 0
        self.hundreds_result = 0

        return self._get_state()

    def _get_state(self):
        return np.array([
            self.number1 // 10,  # num1 onlar
            self.number1 % 10,   # num1 birler
            self.number2 // 10,  # num2 onlar
            self.number2 % 10,   # num2 birler
            self.current_step    # Mevcut adım
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        info = {}

        if action == 0 and self.current_step == 0:
            # Birler basamağı çarpımı
            ones_mult = (self.number1 % 10) * (self.number2 % 10)
            self.ones_result = ones_mult
            reward = 0.3
            self.steps.append(f"Birler: {self.number1 % 10} × {self.number2 % 10} = {ones_mult}")

        elif action == 1 and self.current_step == 1 and self.ones_result is not None:
            # Onlar basamağı çarpımı
            tens = (self.number1 // 10) % 10
            self.tens_result = tens * (self.number2 % 10)
            reward = 0.3
            self.steps.append(f"Onlar: {tens} × {self.number2 % 10} = {self.tens_result}")

        elif action == 2 and self.current_step == 2 and self.tens_result is not None:
            # İkinci satır birler basamağı çarpımı
            ones_mult2 = (self.number1 % 10) * (self.number2 // 10)
            self.ones_result += ones_mult2 * 10  # Çarpım sonucu 10 ile kaydırılarak eklenir
            reward = 0.3
            self.steps.append(f"Birler2: {self.number1 % 10} × {self.number2 // 10} = {ones_mult2 * 10}")

        elif action == 3 and self.current_step == 3:
            # İkinci satır onlar basamağı çarpımı
            tens_mult2 = (self.number1 // 10) * (self.number2 // 10)
            self.tens_result += tens_mult2 * 100  # Çarpım sonucu 100 ile kaydırılarak eklenir
            reward = 0.3
            self.steps.append(f"Onlar2: {self.number1 // 10} × {self.number2 // 10} = {tens_mult2 * 100}")

        elif action == 4 and self.current_step == 4:
            # Toplama işlemi
            if None not in [self.ones_result, self.tens_result]:
                answer = self.ones_result + self.tens_result
                if answer == self.correct_answer:
                    reward = 1.0
                    info['message'] = "Doğru cevap!"
                else:
                    reward = -0.5
                    info['message'] = f"Yanlış! Doğrusu: {self.correct_answer}"
            else:
                reward = -1.0
                info['message'] = "Eksik adımlar!"

        else:
            reward = -0.2  # Geçersiz adım

        self.current_step += 1
        return self._get_state(), reward, done, info


# Transformer modelinin tanımı
class MathTransformer(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, nhead=4, num_layers=2, output_dim=7):
        super(MathTransformer, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension (batch, seq_len=1, features)
        x = self.transformer(x)
        x = x.squeeze(1)  # Remove sequence dimension
        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value


# Ajan sınıfı
class TransformerRLAgent:
    def __init__(self):
        self.model = MathTransformer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.saved_actions = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        probs, value = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_actions.append((m.log_prob(action), value.squeeze()))
        return action.item()

    def update_policy(self):
        if not self.rewards:
            return

        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # Calculate losses
        policy_loss = []
        value_loss = []
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(value, R))

        # Total loss
        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset buffers
        self.saved_actions = []
        self.rewards = []


# Eğitim fonksiyonu
def train(episodes=500):
    env = SimpleMultiplicationEnv()
    agent = TransformerRLAgent()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            agent.rewards.append(reward)
            total_reward += reward

        agent.update_policy()

        if episode % 50 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")
            print(f"Problem: {env.number1} x {env.number2}")
            for step in env.steps:
                print(f"- {step}")
            print(info.get('message', ''))
            print()

    return agent


# Test fonksiyonu
def test(agent, num_tests=5):
    env = SimpleMultiplicationEnv()

    for _ in range(num_tests):
        state = env.reset()
        done = False
        print(f"\nTest: {env.number1} x {env.number2}")

        while not done:
            action = agent.select_action(state)
            state, _, done, info = env.step(action)
            if env.steps:
                print(f"Step {env.current_step}: {env.steps[-1]}")

        print(info.get('message', ''))
        print(f"Correct: {env.correct_answer}")


if __name__ == "__main__":
    print("Transformer Tabanlı Matematik RL Modeli Eğitimi")
    print("----------------------------------------------")
    agent = train(episodes=20000)

    print("\nModel Testi")
    print("-----------")
    test(agent)
