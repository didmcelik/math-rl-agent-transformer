from utils.vocab import Vocab
from agent.rl_agent import RLAgent
from train.train_agent import solve_examples
from train.train_agent import train, solve_examples

if __name__ == "__main__":
    vocab = Vocab()
    agent = RLAgent(vocab)
    train(agent, num_episodes=10000)
    solve_examples(agent, num_examples=5)
