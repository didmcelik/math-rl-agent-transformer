# Transformer-based PPO Agent for Step-by-Step Arithmetic Reasoning

This project implements a Transformer-based PPO (Proximal Policy Optimization) agent that learns to perform arithmetic tasks such as addition and multiplication by generating a sequence of reasoning steps. The agent operates in environments where each action corresponds to a natural language instruction like "multiply 7 by 3 with carry 1".

---

## 🔍 Overview

- The agent is trained using PPO.
- The policy network is based on a Transformer encoder.
- Environments provide arithmetic problems with expected multi-step reasoning chains.
- The agent must generate these chains through trial and error.

---

## 🤖 TransformerPolicy Network

`TransformerPolicy` is a PyTorch module that maps a bag-of-words input vector (concatenated problem text + current chain text) to:

- Action probabilities: `policy_head`
- State value estimation: `value_head`

### Forward Pass:
```python
x = self.embedding(x)           # Linear projection
x = x.unsqueeze(0)              # Add sequence dimension
x = self.transformer(x)         # Transformer encoder processes input
x = x.squeeze(0)
action_probs = softmax(policy_head(x))
state_value = value_head(x)
```

Although the input is not sequential (it's a dense vector), the Transformer still models contextual relationships across dimensions.

---

## 🏃 PPOAgent Training Flow

In `train()`:

1. A batch is sampled from memory.
2. The Transformer processes states to predict:
   - `action_probs` for computing new log probabilities.
   - `state_values` for computing advantages.
3. PPO loss is calculated:
   - **Policy loss**: encourages better actions.
   - **Value loss**: trains the value head.
   - **Entropy**: encourages exploration.

```python
action_probs, state_values = self.policy(states)
...
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
loss.backward()
optimizer.step()
```

The Transformer is optimized end-to-end with PPO.

---

## 📊 Policy Evaluation in Action

During evaluation:
- The agent constructs a `test_chain` step by step.
- At each step, the Transformer-based policy predicts the most probable next action.
- The environment compares the chain to the expected steps and returns feedback.

Example:
```python
action, _ = agent.select_action(state)
state, reward, done, feedback = env.step(action)
test_chain.append(env.allowed_actions[action])
```

---

## ✨ Why Use a Transformer?

| MLP                            | Transformer                                   |
|-------------------------------|-----------------------------------------------|
| No context modeling           | Learns relationships between text features    |
| Weak for long dependencies    | Good at modeling reasoning chains             |
| Static processing             | Adaptive attention across input dimensions    |

Even without token sequences, the Transformer benefits from richer, contextual representation of the arithmetic state.

---

## 🔧 Project Structure (Simplified)
```
agent/
   rl_agent_transformer.py
env/
   addition_env.py
   multiplication_env.py
model/
   transformer_policy.py
teacher/
   addition_teacher.py
   multiplication_teacher.py
train/
   train_agent_addition.py
utils/
   vocab.py
main.py
```

---

## 🔹 Usage

To train the agent:
```bash
python main.py
```

Training and test output will display the predicted chain vs. expected solution.

---

## 📃 Requirements
```
torch
numpy
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```

---
