# test_agent_step.py
from agent.rl_agent_transformer import RLAgentTransformer
from utils.vocab import Vocab
from env.multiplication_env import MultiplicationEnvTwoTwo

vocab = Vocab()
agent = RLAgentTransformer(vocab)
env = MultiplicationEnvTwoTwo()
problem, chain = env.reset()
idx = agent.choose_action(problem, chain, env.allowed_actions)
assert 0 <= idx < len(env.allowed_actions)
print("Chosen idx:", idx, "action:", env.allowed_actions[idx])
# Log-prob eklendi mi?
assert len(agent.episode_log_probs) == 1
