# run_interaction.py – test model + env
from tokenizer_old import MathTokenizer
from model import TinyTransformer
from math_env import MathEnv
import torch

tokenizer = MathTokenizer()
model = TinyTransformer(tokenizer.vocab_size())

env = MathEnv("2x+3=7")
state = env.reset()

for step in range(10):
    input_ids = torch.tensor([tokenizer.encode("".join(state))])
    with torch.no_grad():
        logits = model(input_ids)
        next_token_logits = logits[0, -1]
        next_token_id = torch.argmax(next_token_logits).item()
        next_token = tokenizer.id_to_token[next_token_id]

    state, reward, done = env.step(next_token)
    print(f"Step {step}: Token = {next_token}, State = {state}, Reward = {reward}")
    if done:
        print("Episode finished.")
        break
