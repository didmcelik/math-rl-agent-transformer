# Math RL Agent - 2x2 Multiplication

This project implements a **Transformer-based Reinforcement Learning (RL) agent** that learns to **solve 2-digit × 2-digit multiplication problems step-by-step**.

The agent is trained **without supervision** — it learns **only from external reward signals** by interacting with a custom-designed environment.

---

## 🚀 Project Overview

- **Problem Type**: 2-digit × 2-digit multiplication (e.g., 34 × 76)
- **Environment**: Generates random multiplication problems and checks solution steps.
- **Agent**: A causal **Transformer model** that generates solution steps **token-by-token**.
- **Learning Method**: Reinforcement Learning (Policy Gradient).

The agent must learn to **multiply digits**, **manage carry values**, and **output the final result**, all without any labeled training data.

---

## 🛠 Project Structure

```
math-rl-agent-transformer/
│
├── env/
│   └── multiplication_env.py       # 2x2 multiplication environment with carry handling
│
├── model/
│   └── transformer_policy.py        # Transformer model for step generation
│
├── utils/
│   └── vocab.py                     # Vocabulary management (tokenization)
│
├── agent/
│   └── rl_agent.py                   # RL agent controlling the transformer
│
├── train/
│   └── train_agent.py                # Training and testing functions
│
├── main.py                           # Entry point to run training or testing
│
├── README.md                         # Project description
│
└── requirements.txt                  # Python package requirements
```

---

## 📋 How It Works

1. **Environment** generates a random 2x2 multiplication problem.
2. **Transformer agent** observes the problem and past steps.
3. At each step, the agent **generates the next action** (e.g., `"multiply 4 by 6 output 4 carry 2"`).
4. The environment **evaluates** the action and provides a **reward**:
   - +1 for correct steps
   - -0.5 for incorrect steps
5. The agent **updates its policy** to maximize expected rewards over time.

---

## 🧑‍💻 Key Components

| Component | Description |
|:----------|:------------|
| Environment | Generates problems, defines correct step sequences, gives rewards. |
| Transformer Policy | Predicts the next token in the solution step. |
| RL Agent | Samples actions and updates policy using rewards. |
| Vocab | Encodes/decodes textual actions into token IDs. |

---

## 📈 Training

You can train the agent using the reinforcement learning loop provided.

```bash
python main.py
```

This will:
- Initialize the agent and environment.
- Train the agent for a given number of episodes.
- Display example problems and the agent's generated solution steps.

*Note:* Training purely via RL can be slow due to sparse rewards. Supervised pretraining is recommended for faster convergence.

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch

Install the requirements:

```bash
pip install -r requirements.txt
```

---

## 📢 Future Improvements (Optional)

- **Supervised Pretraining**: Train the agent first using correct step sequences before switching to RL.
- **Curriculum Learning**: Start with simpler problems, gradually increase difficulty.
- **Beam Search**: Improve action selection during inference.
- **Reward Shaping**: Provide partial rewards for partially correct steps.

---

## 📄 License

This project is for educational and research purposes.

---

# ✨ Good luck training your math RL agent! 🚀

