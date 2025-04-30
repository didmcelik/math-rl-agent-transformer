import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

#############################################
# Global Vocabulary and Text Processing
#############################################

vocab = [
            "multiply", "add", "carry", "and", "combine", "do", "nothing", "subtract",
            "output", "ones", "digit", "tens", "hundreds"
        ] + [str(i) for i in range(10)]
vocab = [w.lower() for w in vocab]
vocab_size = len(vocab)
vocab_dict = {word: idx for idx, word in enumerate(vocab)}


def preprocess(text):
    text = text.lower()
    for symbol in [":", "*", "+", "-", "="]:
        text = text.replace(symbol, f" {symbol} ")
    return text.split()


def encode_text(text):
    tokens = preprocess(text)
    vec = torch.zeros(vocab_size)
    for token in tokens:
        if token in vocab_dict:
            vec[vocab_dict[token]] += 1.0
    return vec


def encode_state_text(problem_text, chain_text):
    return torch.cat([encode_text(problem_text), encode_text(chain_text)])


#############################################
# Environment Setup
#############################################

def generate_addition_problem(num_digits=2):
    if num_digits == 2:
        A = random.randint(10, 99)
        B = random.randint(10, 99)
    elif num_digits == 3:
        A = random.randint(100, 999)
        B = random.randint(100, 999)
    else:
        raise ValueError("Only 2 or 3 digit addition supported.")

    correct_answer = A + B
    problem_text = f"add: add {A} + {B}"

    ones_A = A % 10
    ones_B = B % 10
    tens_A = (A // 10) % 10
    tens_B = (B // 10) % 10
    hundreds_A = A // 100
    hundreds_B = B // 100

    S1 = ones_A + ones_B
    carry1 = S1 // 10

    S2 = tens_A + tens_B + carry1
    carry2 = S2 // 10

    S3 = hundreds_A + hundreds_B + carry2

    step1 = f"add ones: {ones_A} and {ones_B}"
    step2 = f"add tens: {tens_A} and {tens_B} with carry {carry1}"
    step3 = f"add hundreds: {hundreds_A} and {hundreds_B} with carry {carry2}"

    correct_steps = [step1.lower(), step2.lower()]
    if num_digits == 3:
        correct_steps.append(step3.lower())

    allowed_actions = [
        step1.lower(), step2.lower(),
        step3.lower() if num_digits == 3 else "dummy_step",
        "do nothing", "subtract", "add tens", "dummy1"
    ]
    return problem_text.lower(), allowed_actions, correct_steps, correct_answer, A, B


class AdditionTeacherSimple:
    def __init__(self, correct_steps, correct_answer, A, B):
        self.correct_steps = correct_steps
        self.correct_answer = correct_answer
        self.A = A
        self.B = B

    def parse_add_ones(self, step_text):
        tokens = step_text.split()
        if len(tokens) != 5:
            return None, None
        try:
            return int(tokens[2]), int(tokens[4])
        except:
            return None, None

    def parse_add_tens(self, step_text):
        tokens = step_text.split()
        if len(tokens) != 8:
            return None, None, None
        try:
            return int(tokens[2]), int(tokens[4]), int(tokens[7])
        except:
            return None, None, None

    def parse_add_hundreds(self, step_text):
        tokens = step_text.split()
        if len(tokens) != 8:
            return None, None, None
        try:
            return int(tokens[2]), int(tokens[4]), int(tokens[7])
        except:
            return None, None, None

    def compute_sum_from_chain(self, chain):
        if len(chain) < 2:
            return None
        X, Y = self.parse_add_ones(chain[0])
        if X is None:
            return None
        S1 = X + Y
        carry1 = S1 // 10
        ones_res = S1 % 10

        T, U, C1 = self.parse_add_tens(chain[1])
        if T is None:
            return None
        if C1 != carry1:
            return None
        S2 = T + U + carry1
        carry2 = S2 // 10
        tens_res = S2 % 10

        if len(chain) == 3:
            H, K, C2 = self.parse_add_hundreds(chain[2])
            if H is None:
                return None
            if C2 != carry2:
                return None
            S3 = H + K + carry2
            hundreds_res = S3
            return hundreds_res * 100 + tens_res * 10 + ones_res
        else:
            return S2 * 10 + ones_res

    def evaluate_solution(self, chain):
        reward = 0.0
        for i, correct_step in enumerate(self.correct_steps):
            if i < len(chain):
                reward += 5 if chain[i] == correct_step else -2
        computed_sum = self.compute_sum_from_chain(chain)

        if(len(chain) == 3 and   chain[-1] == chain[-2]) : #3d add (add hundred at the end)
            ("punish")
            reward -= 15

        if computed_sum is None:
            reward -= 5
            feedback = "Addition procedure invalid."
        else:
            if computed_sum == self.correct_answer:
                reward += 10
                feedback = f"Addition correct! Sum: {computed_sum}."
            else:
                error = abs(computed_sum - self.correct_answer)
                reward -= error
                feedback = f"Procedure may be ok, but computed sum is off (got {computed_sum})."
        return feedback, reward


class AdditionEnvSimple:
    def __init__(self, mode='train'):
        self.mode = mode
        self.reset()

    def reset(self):

        if self.mode == 'train':

            (self.problem_text, self.allowed_actions,
             self.correct_steps, self.correct_answer, self.A, self.B) = generate_addition_problem(num_digits=2)
        else:
            (self.problem_text, self.allowed_actions,
             self.correct_steps, self.correct_answer, self.A, self.B) = generate_addition_problem(num_digits=3)

        self.num_actions = len(self.allowed_actions)
        self.teacher = AdditionTeacherSimple(self.correct_steps, self.correct_answer, self.A, self.B)
        self.chain = []
        self.chain_text = ""
        self.max_steps = len(self.correct_steps)
        return self.get_state()

    def get_state(self):
        return encode_state_text(self.problem_text, self.chain_text)

    def step(self, action_index):
        action = self.allowed_actions[action_index]
        self.chain.append(action)
        self.chain_text = (self.chain_text + " " + action) if self.chain_text else action
        done = (len(self.chain) >= self.max_steps)
        reward = 0.0
        current_step = len(self.chain) - 1
        if current_step < len(self.correct_steps):
            reward = 5.0 if self.chain[current_step] == self.correct_steps[current_step] else -2.0
        if done:
            feedback, final_reward = self.teacher.evaluate_solution(self.chain)
            reward += final_reward
            return self.get_state(), reward, done, feedback
        return self.get_state(), reward, done, ""


#############################################
# Transformer Policy Network
#############################################

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
        x = x.unsqueeze(0)
        x = self.transformer(x)
        x = x.squeeze(0)
        action_probs = torch.softmax(self.policy_head(x), dim=-1)
        state_value = self.value_head(x)
        return action_probs, state_value


#############################################
# Reinforcement Learning Agent
#############################################

class PPOAgent:
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.policy = TransformerPolicy(env.get_state().shape[0], env.num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.gamma = 0.99
        self.clip_epsilon = 0.2
        self.batch_size = 32
        self.memory = deque(maxlen=1000)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs, _ = self.policy(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def remember(self, state, action, log_prob, reward, done):
        self.memory.append((state, action, log_prob, reward, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, rewards, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        discounted_rewards = []
        running_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_reward = 0
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        action_probs, state_values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        advantages = discounted_rewards - state_values.squeeze()

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(state_values.squeeze(), discounted_rewards)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


#############################################
# Training Loop
#############################################

def train_agent(episodes=1000):
    env = AdditionEnvSimple(mode='train')
    agent = PPOAgent(env)

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, log_prob, reward, done)
            state = next_state
            total_reward += reward

        loss = agent.train()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward}, Loss: {loss if loss else 0}")
            # test_env = AdditionEnvSimple(mode='test')
            # test_state = test_env.reset()
            # print(f"Test Variables: {test_env.A, test_env.B}")
            # done = False
            # test_chain = []
            # while not done:
            #     action, _ = agent.select_action(test_state)
            #     next_state, reward, done, info = test_env.step(action)
            #     test_chain.append(test_env.allowed_actions[action])
            #     test_state = next_state
            # print(f"Test Chain: {test_chain}")
            # print(f"Correct Chain: {test_env.correct_steps}")
            # print(f"Feedback: {info}\n")

    return agent


def test_agent_with_3d_add(agent, episodes=1000):
    test_env = AdditionEnvSimple(mode='test')

    for episode in range(episodes):
        if episode % 100 == 0:
            print(f"Episode {episode}")
            test_state = test_env.reset()
            print(f"Test Variables: {test_env.A, test_env.B}")
            done = False
            test_chain = []
            while not done:
                action, _ = agent.select_action(test_state)
                next_state, reward, done, info = test_env.step(action)
                test_chain.append(test_env.allowed_actions[action])
                test_state = next_state
            print(f"Test Chain: {test_chain}")
            print(f"Correct Chain: {test_env.correct_steps}")
            print(f"Feedback: {info}\n")

    return agent


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    trained_agent = train_agent(episodes=5000)
    test_agent_with_3d_add(trained_agent)
