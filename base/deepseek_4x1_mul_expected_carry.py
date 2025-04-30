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

        self.ones_digit = self.number % 10
        self.tens_digit = (self.number // 10) % 10
        self.hundreds_digit = (self.number // 100) % 10
        self.thousands_digit = (self.number // 1000) % 10

        self.ones_result = None
        self.tens_result = None
        self.hundreds_result = None
        self.thousands_result = None
        self.carry = 0

        return self._get_state()

    def _get_state(self):
        return np.array([
            self.thousands_digit,
            self.hundreds_digit,
            self.tens_digit,
            self.ones_digit,
            self.multiplier,
            self.current_step,
            self.carry,
            0.0 if self.ones_result is None else self.ones_result,
            0.0 if self.tens_result is None else self.tens_result,
            0.0 if self.hundreds_result is None else self.hundreds_result,
            0.0 if self.thousands_result is None else self.thousands_result
        ], dtype=np.float32)

    def step(self, action, predicted_digit=None, predicted_carry=None):
        reward = 0
        done = False
        info = {}

        correct_digit = None
        correct_carry = None

        if self.digits == 3:
            expected_steps = 3
        else:
            expected_steps = 4

        if action in [0, 1, 2, 3]:
            if action == 0 and self.current_step == 0:
                correct_digit = (self.ones_digit * self.multiplier) % 10
                correct_carry = (self.ones_digit * self.multiplier) // 10

            elif action == 1 and self.current_step == 1:
                correct_digit = (self.tens_digit * self.multiplier + self.carry) % 10
                correct_carry = (self.tens_digit * self.multiplier + self.carry) // 10

            elif action == 2 and self.current_step == 2:
                correct_digit = (self.hundreds_digit * self.multiplier + self.carry) % 10
                correct_carry = (self.hundreds_digit * self.multiplier + self.carry) // 10

            elif action == 3 and self.current_step == 3:
                correct_digit = (self.thousands_digit * self.multiplier + self.carry) % 10
                correct_carry = (self.thousands_digit * self.multiplier + self.carry) // 10

            if predicted_digit is not None and predicted_carry is not None:
                if predicted_digit == correct_digit and predicted_carry == correct_carry:
                    reward = 0.5
                else:
                    reward = -0.5

                self.carry = predicted_carry

                if action == 0:
                    self.ones_result = predicted_digit
                    self.steps.append(f"Predicted Ones: {predicted_digit} with Carry: {predicted_carry}")
                elif action == 1:
                    self.tens_result = predicted_digit
                    self.steps.append(f"Predicted Tens: {predicted_digit} with Carry: {predicted_carry}")
                elif action == 2:
                    self.hundreds_result = predicted_digit
                    self.steps.append(f"Predicted Hundreds: {predicted_digit} with Carry: {predicted_carry}")
                elif action == 3:
                    self.thousands_result = predicted_digit
                    self.steps.append(f"Predicted Thousands: {predicted_digit} with Carry: {predicted_carry}")
            else:
                reward = -1.0

        elif action == 6 and self.current_step >= expected_steps:
            done = True

            if (self.digits == 3 and None in [self.ones_result, self.tens_result, self.hundreds_result]) or \
               (self.digits == 4 and None in [self.ones_result, self.tens_result, self.hundreds_result, self.thousands_result]):
                reward = -1.0
                info['message'] = "Incomplete steps!"
                return self._get_state(), reward, done, info, None, None

            if self.digits == 3:
                predicted_answer = (self.hundreds_result * 100) + (self.tens_result * 10) + self.ones_result
            else:
                predicted_answer = (self.thousands_result * 1000) + (self.hundreds_result * 100) + (self.tens_result * 10) + self.ones_result

            if predicted_answer == self.correct_answer:
                reward = 2.0
                info['message'] = "Correct full answer!"
            else:
                reward = -1.0
                info['message'] = f"Wrong full answer! Correct: {self.correct_answer}"

        else:
            reward = -1.0

        self.current_step += 1
        return self._get_state(), reward, done, info, correct_digit, correct_carry

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
        self.digit_head = nn.Linear(hidden_dim, 10)
        self.carry_head = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)

        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        digit_logits = self.digit_head(x)
        carry_logits = self.carry_head(x)

        return action_probs, state_value, digit_logits, carry_logits


class TransformerRLAgent:
    def __init__(self):
        self.model = MathTransformer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        self.saved_actions = []
        self.rewards = []
        self.saved_digits = []
        self.saved_carries = []

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value, digit_logits, carry_logits = self.model(state)
        m = torch.distributions.Categorical(action_probs)
        action = m.sample()
        self.saved_actions.append((m.log_prob(action), value.squeeze(), digit_logits, carry_logits))
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
        digit_loss = []
        carry_loss = []

        for (log_prob, value, digit_logits, carry_logits), R, (true_digit, true_carry) in zip(self.saved_actions, returns, self.saved_digits):
            if true_digit is None or true_carry is None:
                continue  # Skip this step if no digit/carry prediction available

            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(nn.functional.mse_loss(value, R))

            true_digit = torch.tensor([true_digit], dtype=torch.long)
            true_carry = torch.tensor([true_carry], dtype=torch.long)
            digit_loss.append(nn.functional.cross_entropy(digit_logits, true_digit))
            carry_loss.append(nn.functional.cross_entropy(carry_logits, true_carry))


        if not policy_loss:
            return  # No valid actions to update

        loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum() + torch.stack(digit_loss).sum() + torch.stack(carry_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.saved_actions = []
        self.rewards = []
        self.saved_digits = []


def train(episodes=500):
    env = MultiplicationEnvironment(digits=3)
    agent = TransformerRLAgent()

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.select_action(state)
            state, reward, done, info, expected_digit, expected_carry = env.step(action)
            agent.rewards.append(reward)
            agent.saved_digits.append((expected_digit, expected_carry))
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
            state, _, done, info, _, _ = env.step(action)
            if env.steps:
                print(f"Step {env.current_step}: {env.steps[-1]}")

        print(info.get('message', ''))
        print(f"Correct Answer: {env.correct_answer}")


if __name__ == "__main__":
    print("Training Transformer-Based Math RL Agent with Digit and Carry Prediction")
    print("--------------------------------------------------------------------------")
    agent = train(episodes=10000)

    print("\nTesting the Trained Model on 4-digit numbers")
    print("--------------------------------------------")
    robust_test(agent)
