# 🧠 MathRLTransformer

**MathRLTransformer** is a reinforcement learning project where a minimal Transformer model is trained to solve symbolic math problems step-by-step using external reward feedback. The goal is to explore the reasoning capabilities of Transformers in symbolic domains without requiring supervised labels.

---

## 📚 Project Overview

This project trains a Transformer model to interact with a symbolic math environment. The model generates mathematical expressions one token at a time and receives feedback (reward) based on how close its output is to the correct solution.

---

## 🧩 Components

### `math_env.py`
Defines the symbolic math environment:
- Problems must be solved in the format: `x=...`
- Rewards are based on solution accuracy and expression structure
- Includes exploration bonuses and length penalties

### `model.py`
Defines the Transformer model:
- `TinyTransformer`: a small encoder-only transformer used as the policy network
- `PositionalEncoding`: adds sine-cosine positional information to token embeddings

### `tokenizer.py`
A minimal tokenizer that supports symbolic math:
- Vocabulary: digits, `x`, `+`, `-`, `=`
- Encodes and decodes expressions into token IDs for training

### `train_rl.py`
Training script using REINFORCE:
- Trains the `TinyTransformer` in the math environment
- Optimizes the model using policy gradients
- Runs multiple episodes to improve reward-driven behavior

### `test_run.py`
Testing script to run the trained model in the environment:
- Loads the environment, tokenizer, and model
- Steps through token generation and prints rewards

---

## 🎯 Goals

- Train a Transformer from scratch using only rewards
- Explore the model's ability to reason symbolically
- Compare math engine-based vs. LLM-based feedback (future work)

---

## 📄 Reference

This project was developed as part of a lab exploring **Transformer-Based Mathematical Reasoning with Reinforcement Learning**, inspired by techniques discussed in symbolic computation and RL research.
