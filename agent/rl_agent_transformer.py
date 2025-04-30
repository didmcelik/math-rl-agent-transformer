import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.transformer_policy import TransformerPolicy


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
