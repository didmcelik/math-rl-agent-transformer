from agent.rl_agent_transformer import PPOAgent
from env.addition_env import AdditionEnvSimple
from env.multiplication_env import MultiplicationEnvThree


def train_agent(episodes=1000, training_type='add'):
    if (training_type == 'add'):
        env = AdditionEnvSimple()
        a = env.A
        b = env.B
    else:
        env = MultiplicationEnvThree()
        a = env.A
        b = env.M

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
            print(f"Test Variables: {a, b}")
            done = False
            test_chain = []
            while not done:
                action, _ = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                test_chain.append(env.allowed_actions[action])
                state = next_state
            print(f"Test Chain: {test_chain}")
            print(f"Correct Chain: {env.correct_steps}")
            print(f"Feedback: {info}\n")

    return agent
