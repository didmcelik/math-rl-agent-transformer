import torch
from env.multiplication_env import MultiplicationEnvTwoTwo

def train(agent, num_episodes=1000):
    env = MultiplicationEnvTwoTwo()
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        log_probs = []
        rewards = []

        while not done:
            problem_text, chain_text = state
            action_token = agent.choose_action(problem_text, chain_text)
            action_ids = torch.tensor(agent.vocab.encode(action_token)).unsqueeze(1)
            logits = agent.model(action_ids)
            last_logits = logits[-1, 0, :]
            prob = torch.softmax(last_logits, dim=0)
            dist = torch.distributions.Categorical(prob)
            log_prob = dist.log_prob(torch.tensor(agent.vocab.token2idx.get(action_token, 0)))

            state, reward, done = env.step(action_token)
            log_probs.append(log_prob)
            rewards.append(reward)

        agent.train_step(log_probs, rewards)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}, Total Reward: {sum(rewards):.2f}")

def solve_examples(agent, num_examples=5):
    env = MultiplicationEnvTwoTwo()
    for example in range(num_examples):
        print("\n" + "="*40)
        problem_text, _ = env.reset()
        print(f"Problem: {problem_text}")
        done = False
        state = (problem_text, "")
        steps = []

        while not done:
            problem_text, chain_text = state
            action_token = agent.choose_action(problem_text, chain_text)
            steps.append(action_token)
            state, reward, done = env.step(action_token)

        print("Agent's Steps:")
        for step in steps:
            print(f" - {step}")
        print(f"True Answer: {env.correct_answer}")
        print("="*40)
