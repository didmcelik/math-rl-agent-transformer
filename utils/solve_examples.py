import pickle
from env.multiplication_env import MultiplicationEnvTwoTwo
from train.train_agent import train_agent



def solve_examples(agent, num_examples=5):
    from env.multiplication_env import MultiplicationEnvTwoTwo

    for example in range(num_examples):
        print("\n" + "=" * 40)
        env = MultiplicationEnvTwoTwo()
        problem_text, chain_text = env.reset()
        print(f"Problem: {env.problem_text} (True Answer: {env.correct_answer})")

        done = False
        steps = []

        while not done:
            action_index = agent.choose_action(problem_text, chain_text, env.steps + ["do nothing"])
            action_text = (env.steps + ["do nothing"])[action_index]
            steps.append(action_text)
            (problem_text, chain_text), reward, done = env.step(action_index)

        print("Agent's Steps:")
        for step in steps:
            print(" -", step)
        print("=" * 40)
