
# train_rl.py – REINFORCE training for TinyTransformer on math environment
import torch
import torch.optim as optim
from tokenizer import MathTokenizer
from model import TinyTransformer
from math_env import MathEnv

# Initialize components
tokenizer = MathTokenizer()
vocab_size = tokenizer.vocab_size()
model = TinyTransformer(vocab_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop parameters
num_episodes = 1000
max_steps = 20
gamma = 1.0  # No discounting for simplicity

for episode in range(num_episodes):
    env = MathEnv("2x+3=7")
    state = env.reset()
    if len(state) == 0:
        state = ['x', '=']  # <--- initial sequence

    log_probs = []
    rewards = []

    for step in range(max_steps):
        input_ids = torch.tensor([tokenizer.encode(state)], dtype=torch.long)

        logits = model(input_ids)
        next_token_logits = logits[0, -1]
        probs = torch.softmax(next_token_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_token = tokenizer.id_to_token[action.item()]
        state, reward, done = env.step(next_token)

        log_probs.append(log_prob)
        rewards.append(reward)

        if done:
            break

    # Compute total reward
    total_reward = sum(rewards)

    # Compute loss
    loss = -sum(log_probs) * total_reward

    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if episode % 50 == 0:
        print(f"Episode {episode}: Reward = {total_reward}, Tokens = {state}")
