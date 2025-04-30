
import torch
from env.multiplication_env import MultiplicationEnvTwoTwo


def solve_examples(agent, num_examples=5):
    env = MultiplicationEnvTwoTwo()
    for example in range(num_examples):
        print("\n" + "="*40)
        problem_text, _ = env.reset()
        print(f"Problem: {problem_text} (True Answer: {env.correct_answer})")
        done = False
        state = (problem_text, "")
        steps = []

        while not done:
            problem_text, chain_text = state
            action_index = agent.choose_action(problem_text, chain_text, env.allowed_actions)
            action_text = env.allowed_actions[action_index]
            steps.append(action_text)
            state, reward, done = env.step(action_text)

        print("Agent's Steps:")
        for step in steps:
            print(f" - {step}")
        print("="*40)
