from env.multiplication_env import MultiplicationEnvTwoTwo
from agent.rl_agent_transformer import RLAgentTransformer
from utils.vocab import Vocab



def train_agent(num_episodes=5000):
    env = MultiplicationEnvTwoTwo()
    vocab = Vocab()
    agent = RLAgentTransformer(vocab=vocab)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_proc_reward = 0.0
        problem_text, chain_text = state
        while not done:
            action_index = agent.choose_action(problem_text, chain_text, env.allowed_actions)
            action_text = env.allowed_actions[action_index]
            next_state, reward, done = env.step(action_text)
            agent.accumulate_loss(reward)
            total_proc_reward += reward
            problem_text, chain_text = next_state

        agent.finalize_episode(total_proc_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward: {total_proc_reward:.2f}")

    return agent
