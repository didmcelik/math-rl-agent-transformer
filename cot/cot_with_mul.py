import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import time
import matplotlib.pyplot as plt

#############################################
# Global Vocabulary and Text Processing
#############################################

# Vocabulary covers key procedural words and digits.
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

def generate_addition_problem_simple():
    A = random.randint(10, 99)
    B = random.randint(10, 99)
    correct_answer = A + B
    problem_text = f"add: add {A} + {B}"
    ones_A = A % 10
    ones_B = B % 10
    tens_A = A // 10
    tens_B = B // 10
    S1 = ones_A + ones_B
    carry = S1 // 10
    step1 = f"add ones: {ones_A} and {ones_B}"
    step2 = f"add tens: {tens_A} and {tens_B} with carry {carry}"
    correct_steps = [step1.lower(), step2.lower()]
    # Pad allowed actions to 7 options.
    allowed_actions = [step1.lower(), step2.lower(), "do nothing", "subtract", "add tens", "dummy1", "dummy2"]
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

    def compute_sum_from_chain(self, chain):
        if len(chain) < 2:
            return None
        X, Y = self.parse_add_ones(chain[0])
        if X is None:
            return None
        S1 = X + Y
        carry = S1 // 10
        ones_res = S1 % 10
        T, U, C = self.parse_add_tens(chain[1])
        if T is None:
            return None
        if C != carry:
            return None
        S2 = T + U + carry
        return S2 * 10 + ones_res

    def evaluate_solution(self, chain):
        reward = 0.0
        for i, correct_step in enumerate(self.correct_steps):
            if i < len(chain):
                reward += 5 if chain[i] == correct_step else -2
        computed_sum = self.compute_sum_from_chain(chain)
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
    def __init__(self):
        self.reset()

    def reset(self):
        (self.problem_text, self.allowed_actions,
         self.correct_steps, self.correct_answer, self.A, self.B) = generate_addition_problem_simple()
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
        # Add sequence dimension (batch_size, seq_len=1, hidden_size)
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

        # Sample a batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, old_log_probs, rewards, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        # Calculate discounted rewards
        discounted_rewards = []
        running_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                running_reward = 0
            running_reward = reward + self.gamma * running_reward
            discounted_rewards.insert(0, running_reward)

        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

        # Get new action probabilities and state values
        action_probs, state_values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Calculate advantages
        advantages = discounted_rewards - state_values.squeeze()

        # Policy loss
        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = nn.MSELoss()(state_values.squeeze(), discounted_rewards)

        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()



def generate_multiplication_problem_three():
   """
   Generates a multiplication problem:
     - A: three-digit number (100-999)
     - M: one-digit number (1-9)
   A human-like six-step procedure is created as follows:
     1. "multiply <ones> by <M>"
     2. "multiply <tens> by <M> with carry <carry1>"
     3. "multiply <hundreds> by <M> with carry <carry2>"
     4. "output ones digit: <d1>"
     5. "output tens digit: <d2>"
     6. "output hundreds digits: <P3>"
   where the correct intermediate tokens are generated using arithmetic.

   Note: The multiplication teacher will no longer recompute these values;
         it only compares the agent’s chain (a sequence of text tokens) to the correct chain.
   """
   A = random.randint(100, 999)
   M = random.randint(1, 9)
   correct_answer = A * M
   problem_text = f"mul: multiply {A} x {M}"

   ones = A % 10
   tens = (A // 10) % 10
   hundreds = A // 100

   # Step 1:
   P1 = ones * M
   d1 = P1 % 10
   carry1 = P1 // 10
   step1 = f"multiply {ones} by {M}"

   # Step 2:
   P2 = tens * M + carry1
   d2 = P2 % 10
   carry2 = P2 // 10
   step2 = f"multiply {tens} by {M} with carry {carry1}"

   # Step 3:
   P3 = hundreds * M + carry2
   step3 = f"multiply {hundreds} by {M} with carry {carry2}"

   # Output steps:
   step4 = f"output ones digit: {d1}"
   step5 = f"output tens digit: {d2}"
   step6 = f"output hundreds digits: {P3}"

   # The correct procedure as a chain of strings (all lowercase)
   correct_steps = [step1.lower(), step2.lower(), step3.lower(), step4.lower(), step5.lower(), step6.lower()]
   # Allowed actions: the 6 correct ones, plus one extra dummy option to pad to a total of 7.
   allowed_actions = correct_steps + ["dummy"]
   return problem_text.lower(), allowed_actions, correct_steps, correct_answer, A, M


class MultiplicationTeacherThree:
   """
   Revised Multiplication Teacher.
   Instead of parsing numbers and performing arithmetic, it only compares
   the agent’s chain of actions (text strings) with the pre-generated correct chain.
   This forces the RL agent to learn the full procedure from the text tokens.
   """

   def __init__(self, correct_steps, correct_answer, A, M):
       self.correct_steps = correct_steps
       self.correct_answer = correct_answer
       self.A = A
       self.M = M

   def evaluate_solution(self, chain):
       reward = 0.0
       # Compare each step with expected text
       for i, correct_step in enumerate(self.correct_steps):
           if i < len(chain):
               reward += 5 if chain[i] == correct_step else -2
       # Give bonus reward if the entire chain exactly matches the expert chain.
       if chain == self.correct_steps:
           reward += 10
           feedback = f"Multiplication correct! Product: {self.correct_answer}."
       else:
           feedback = f"Procedure incorrect. Expected: {self.correct_steps}, but got: {chain}."
       return feedback, reward


class MultiplicationEnvThree:
   def __init__(self):
       self.reset()

   def reset(self):
       (self.problem_text, self.allowed_actions,
        self.correct_steps, self.correct_answer, self.A, self.M) = generate_multiplication_problem_three()
       self.num_actions = len(self.allowed_actions)  # 7 actions now
       self.teacher = MultiplicationTeacherThree(self.correct_steps, self.correct_answer, self.A, self.M)
       self.chain = []
       self.chain_text = ""
       self.max_steps = len(self.correct_steps)  # 6 steps expected
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
       return self.get_state(), reward, done, ""


#############################################
# Training Loop
#############################################

def train_agent(episodes=1000):
    env = MultiplicationEnvThree()
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
            # Test the agent
            state = env.reset()
            #print(f"Test Variables: {env.A,env.B}")
            done = False
            test_chain = []
            while not done:
                action, _ = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                #   next_state, reward, done, info = env.step(action)
                test_chain.append(env.allowed_actions[action])
                state = next_state
            print(f"Test Chain: {test_chain}")
            print(f"Correct Chain: {env.correct_steps}")
            print(f"Feedback: {info}\n")

    return agent



trained_agent = train_agent(episodes=15000)




