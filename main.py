from agent.rl_agent_transformer import RLAgentTransformer
from env.multiplication_env import MultiplicationEnvTwoTwo
from train.train_agent import train_agent
import pickle
from utils.solve_examples import solve_examples
from utils.vocab import Vocab

def test_one_step():
    # End-to-end tek adım testi
    env = MultiplicationEnvTwoTwo()
    vocab = Vocab()
    agent = RLAgentTransformer(vocab=vocab)

    problem_text, chain_text = env.reset()
    print("Problem:", problem_text)
    print("Initial chain_text:", repr(chain_text))

    idx = agent.choose_action(problem_text, chain_text, env.allowed_actions)
    action = env.allowed_actions[idx]
    print(f"Chose action #{idx} → {action}")

    (next_problem, next_chain), reward, done = env.step(action)
    print("Next chain_text:", repr(next_chain))
    print("Reward:", reward, "Done:", done)

if __name__ == "__main__":

    """
        # Train a new agent from scratch
    print("🆕 Creating and training a new agent...")
    trained_agent = train_agent( num_episodes=5000)

    print("\n✅ Training complete!")

    # Solve a few example problems
    print("\n🧠 Testing trained agent on example problems...")
    solve_examples(trained_agent, num_examples=5)
    
    
    """

    # 1) Tek adım uyum testi
    print("🔧 Testing end-to-end one step")
    test_one_step()

    # 2) Eğitim
    print("\n🆕 Creating and training a new agent...")
    trained_agent = train_agent(num_episodes=5000)

    print("\n✅ Training complete!")

    # 3) Örnekleri çöz
    print("\n🧠 Testing trained agent on example problems...")
    solve_examples(trained_agent, num_examples=5)
