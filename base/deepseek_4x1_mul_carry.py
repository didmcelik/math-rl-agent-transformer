import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MultiplicationEnvironment:
    def __init__(self, digits=3):
        self.digits = digits
        self.reset()

    def reset(self):
        if self.digits == 3:
            self.number = random.randint(100, 999)
        else:
            self.number = random.randint(1000, 9999)
        self.multiplier = random.randint(1, 9)
        self.correct_answer = self.number * self.multiplier
        self.steps = []
        self.current_step = 0

        self.ones_result = None
        self.tens_result = None
        self.hundreds_result = None
        self.thousands_result = None
        self.carry = 0

        return self._get_state()

    def _get_state(self):
        return np.array([
            (self.number // 1000) % 10,
            (self.number // 100) % 10,
            (self.number // 10) % 10,
            self.number % 10,
            self.multiplier,
            self.current_step,
            self.carry,
            0.0 if self.ones_result is None else self.ones_result,
            0.0 if self.tens_result is None else self.tens_result,
            0.0 if self.hundreds_result is None else self.hundreds_result,
            0.0 if self.thousands_result is None else self.thousands_result
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        done = False
        info = {}

        if action == 0 and self.current_step == 0:
            ones = self.number % 10
            raw_result = ones * self.multiplier
            expected_digit = raw_result % 10
            expected_carry = raw_result // 10

            self.ones_result = raw_result
            digit = raw_result % 10
            self.carry = raw_result // 10

            if digit == expected_digit and self.carry == expected_carry:
                reward = 0.3
            else:
                reward = -0.5

            self.steps.append(f"Ones: {ones} x {self.multiplier} = {digit} carry {self.carry}")

        elif action == 1 and self.current_step == 1 and self.ones_result is not None:
            tens = (self.number // 10) % 10
            raw_result = tens * self.multiplier + (self.ones_result // 10)
            expected_digit = raw_result % 10
            expected_carry = raw_result // 10

            self.tens_result = raw_result
            digit = raw_result % 10
            self.carry = raw_result // 10

            if digit == expected_digit and self.carry == expected_carry:
                reward = 0.3
            else:
                reward = -0.5

            self.steps.append(f"Tens: {tens} x {self.multiplier} + {self.ones_result // 10} = {digit} carry {self.carry}")

        elif action == 2 and self.current_step == 2 and self.tens_result is not None:
            hundreds = (self.number // 100) % 10
            raw_result = hundreds * self.multiplier + (self.tens_result // 10)
            expected_digit = raw_result % 10
            expected_carry = raw_result // 10

            self.hundreds_result = raw_result
            digit = raw_result % 10
            self.carry = raw_result // 10

            if digit == expected_digit and self.carry == expected_carry:
                reward = 0.3
            else:
                reward = -0.5

            self.steps.append(f"Hundreds: {hundreds} x {self.multiplier} + {self.tens_result // 10} = {digit} carry {self.carry}")

        elif action == 3 and self.current_step == 3 and self.hundreds_result is not None:
            thousands = (self.number // 1000) % 10
            raw_result = thousands * self.multiplier + (self.hundreds_result // 10)
            expected_digit = raw_result % 10
            expected_carry = raw_result // 10

            self.thousands_result = raw_result
            digit = raw_result % 10
            self.carry = raw_result // 10

            if digit == expected_digit and self.carry == expected_carry:
                reward = 0.3
            else:
                reward = -0.5

            self.steps.append(f"Thousands: {thousands} x {self.multiplier} + {self.hundreds_result // 10} = {digit} carry {self.carry}")

        elif action == 6 and self.current_step >= (3 if self.digits == 3 else 4):
            done = True
            if self.digits == 3 and None not in [self.ones_result, self.tens_result, self.hundreds_result]:
                final_answer = (self.hundreds_result * 100) + ((self.tens_result % 10) * 10) + (self.ones_result % 10)
                if final_answer == self.correct_answer:
                    reward = 1.0
                    info['message'] = "Correct full answer!"
                else:
                    reward = -0.5
                    info['message'] = f"Wrong full answer! Correct: {self.correct_answer}"
            elif self.digits == 4 and None not in [self.ones_result, self.tens_result, self.hundreds_result, self.thousands_result]:
                final_answer = (self.thousands_result * 1000) + ((self.hundreds_result % 10) * 100) + ((self.tens_result % 10) * 10) + (self.ones_result % 10)
                if final_answer == self.correct_answer:
                    reward = 1.0
                    info['message'] = "Correct full answer!"
                else:
                    reward = -0.5
                    info['message'] = f"Wrong full answer! Correct: {self.correct_answer}"
            else:
                reward = -1.0
                info['message'] = "Incomplete steps!"

        else:
            reward = -0.5

        self.current_step += 1
        return self._get_state(), reward, done, info


class MathTransformer(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, nhead=4, num_layers=2, output_dim=8):
        super().__init__()
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
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value


class TransformerRLAgent:
    def __init__(self):
        self.model = MathTransformer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.saved_actions = []
        self.rewards = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs, value = self.model(state)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        self.saved_actions.append((m.log_prob(action), value.squeeze()))
        return action.item()

    def update_policy(self):
        if not self.rewards:
            return

        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_loss = []
        value_loss = []
        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(value, R))

        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_actions = []
        self.rewards = []


def train(episodes=500):
    env = MultiplicationEnvironment(digits=3)
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
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")
            print(f"Problem: {env.number} x {env.multiplier}")
            for step in env.steps:
                print(f"- {step}")
            print(info.get('message', ''))
            print()

    return agent


def robust_test(agent, num_tests=5):
    env = MultiplicationEnvironment(digits=4)

    for _ in range(num_tests):
        state = env.reset()
        done = False
        print(f"\nTesting {env.number} x {env.multiplier}")

        while not done:
            action = agent.select_action(state)
            state, _, done, info = env.step(action)
            if env.steps:
                print(f"Step {env.current_step}: {env.steps[-1]}" )

        print(info.get('message', ''))
        print(f"Correct Answer: {env.correct_answer}")


if __name__ == "__main__":
    print("Training Transformer-Based Math RL Agent (3-digit numbers)")
    print("-----------------------------------------------------------")
    agent = train(episodes=10000)

    print("\nTesting the Trained Model on 4-digit numbers")
    print("--------------------------------------------")
    robust_test(agent)
