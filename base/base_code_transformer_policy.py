import torch
import torch.nn as nn
import torch.optim as optim
import random
import tkinter as tk

#############################################
# Global Vocabulary and Text Processing
#############################################

vocab = [
    "multiply", "carry", "and", "combine", "do", "nothing", "subtract",
    "output", "ones", "digit", "tens", "hundreds"
] + [str(i) for i in range(10)]
vocab = [w.lower() for w in vocab]
vocab_size = len(vocab)
vocab_dict = {word: idx for idx, word in enumerate(vocab)}

def preprocess(text):
    text = text.lower()
    for symbol in [":", "*", "+", "-", "="]:
        text = text.replace(symbol, f" {symbol} ")
    return text.split()

def encode_text(text):
    tokens = preprocess(text)
    ids = [vocab_dict[token] for token in tokens if token in vocab_dict]
    return torch.tensor(ids, dtype=torch.long)  # 🔥 Long tensor

def encode_state_text(problem_text, chain_text):
    problem_ids = encode_text(problem_text)
    chain_ids = encode_text(chain_text)
    return torch.cat([problem_ids, chain_ids], dim=0)  # 🔥 Long tensor



#############################################
# Multiplication Module (ONLY MULTIPLY)
#############################################

def generate_multiplication_problem_three():
    A = random.randint(100, 999)
    M = random.randint(1, 9)
    correct_answer = A * M
    problem_text = f"mul: multiply {A} x {M}"

    ones = A % 10
    tens = (A // 10) % 10
    hundreds = A // 100

    P1 = ones * M
    d1 = P1 % 10
    carry1 = P1 // 10

    P2 = tens * M + carry1
    d2 = P2 % 10
    carry2 = P2 // 10

    P3 = hundreds * M + carry2

    step1 = f"multiply {ones} by {M}"
    step2 = f"multiply {tens} by {M} with carry {carry1}"
    step3 = f"multiply {hundreds} by {M} with carry {carry2}"
    step4 = f"output ones digit: {d1}"
    step5 = f"output tens digit: {d2}"
    step6 = f"output hundreds digits: {P3}"

    correct_steps = [step1.lower(), step2.lower(), step3.lower(), step4.lower(), step5.lower(), step6.lower()]
    allowed_actions = correct_steps + ["dummy"]
    return problem_text.lower(), allowed_actions, correct_steps, correct_answer, A, M

class MultiplicationTeacherThree:
    def __init__(self, correct_steps, correct_answer, A, M):
        self.correct_steps = correct_steps
        self.correct_answer = correct_answer
        self.A = A
        self.M = M

    def evaluate_solution(self, chain):
        reward = 0.0
        for i, correct_step in enumerate(self.correct_steps):
            if i < len(chain):
                reward += 5 if chain[i] == correct_step else -2
        if chain == self.correct_steps:
            reward += 10
            feedback = f"Multiplication correct! Product: {self.correct_answer}."
        else:
            feedback = f"Procedure incorrect. Expected: {self.correct_steps}, but got: {chain}."
        return feedback, reward

class MultiplicationEnvThree:
    def __init__(self):
        self.reset()

    def reset(self):
        (self.problem_text, self.allowed_actions,
         self.correct_steps, self.correct_answer, self.A, self.M) = generate_multiplication_problem_three()
        self.num_actions = len(self.allowed_actions)
        self.teacher = MultiplicationTeacherThree(self.correct_steps, self.correct_answer, self.A, self.M)
        self.chain = []
        self.chain_text = ""
        self.max_steps = len(self.correct_steps)
        return self.get_state()

    def get_state(self):
        return encode_state_text(self.problem_text, self.chain_text)

    def step(self, action_index):
        action = self.allowed_actions[action_index]
        self.chain.append(action)
        self.chain_text = (self.chain_text + " " + action) if self.chain_text else action
        done = (len(self.chain) >= self.max_steps)
        reward = 0.0
        current_step = len(self.chain) - 1
        if current_step < len(self.correct_steps):
            reward = 5.0 if self.chain[current_step] == self.correct_steps[current_step] else -2.0
        return self.get_state(), reward, done, ""

#############################################
# Transformer Policy Network
#############################################

class TransformerPolicy(nn.Module):

    #    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=256,
                batch_first=False  # Çünkü seq_len, batch_size formatında vereceğiz
            ),
            num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src):
        embedded = self.embedding(src)  # (seq_len, batch_size, d_model)
        output = self.transformer(embedded)  # (seq_len, batch_size, d_model)
        pooled = output.mean(dim=0)  # (batch_size, d_model)
        score = self.fc_out(pooled)  # (batch_size, 1)
        return score.squeeze(-1)  # (batch_size,)


#############################################
# RL Agent
#############################################
class RLAgentSimple:
    def __init__(self, vocab, lr=1e-4, hidden_dim=128):
        self.vocab = vocab
        self.policy_net = TransformerPolicy(vocab_size=len(vocab))
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.episode_log_probs = []
        self.cumulative_loss = 0.0

    def choose_action(self, state, allowed_actions):
        action_scores = []
        for action_text in allowed_actions:
            action_ids = encode_text(action_text)  # 🔥 Already LongTensor
            combined = torch.cat([state, action_ids], dim=0)  # 🔥 IDs birleşiyor
            combined = combined.unsqueeze(1)  # (seq_len, batch_size=1)
            score = self.policy_net(combined)
            action_scores.append(score)

        scores_tensor = torch.stack(action_scores).view(-1)
        probs = torch.softmax(scores_tensor, dim=0)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.episode_log_probs.append(dist.log_prob(action))
        return action.item()

    def accumulate_loss(self, reward):  # 🔥 Bunu yeni ekliyoruz
        if self.episode_log_probs:
            self.cumulative_loss += -self.episode_log_probs[-1] * reward



    def finalize_episode(self, final_reward):
        bonus = sum([-lp * final_reward for lp in self.episode_log_probs])
        self.cumulative_loss += bonus
        self.optimizer.zero_grad()
        self.cumulative_loss.backward()
        self.optimizer.step()
        self.episode_log_probs = []
        self.cumulative_loss = 0.0
#############################################
# Training Loop
#############################################

def train_agent(num_episodes=5000):
    env = MultiplicationEnvThree()  # Artık sadece multiplication var
    agent = RLAgentSimple(vocab=vocab_dict, lr=1e-4)  # Doğru çağrı

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_proc_reward = 0.0

        while not done:

            action = agent.choose_action(state, env.allowed_actions)
            next_state, reward, done, _ = env.step(action)  # ✅ (4 değer alıyoruz, sonuncusu boş _ ile tutuluyor)

            agent.accumulate_loss(reward)
            total_proc_reward += reward
            state = next_state

        teacher = env.teacher
        teacher_feedback, teacher_reward = teacher.evaluate_solution(env.chain)

        total_final_reward = total_proc_reward + teacher_reward
        agent.finalize_episode(total_final_reward)

        if (episode + 1) % 100 == 0:
            computed = teacher.correct_answer if env.chain == teacher.correct_steps else "N/A"
            print(
                f"Episode {episode + 1}: Proc Reward: {total_proc_reward:.2f}, "
                f"Teacher Reward: {teacher_reward:.2f}, Computed: {computed}, True: {env.correct_answer}"
            )

    return agent


#############################################
# GUI for Testing
#############################################

class CompositeApp(tk.Tk):
    def __init__(self, trained_agent):
        super().__init__()
        self.title("Multiplication Solver")
        self.agent = trained_agent
        self.env = MultiplicationEnvThree()
        self.state = self.env.reset()

        self.problem_label = tk.Label(self, text=self.env.problem_text, font=("Helvetica", 16))
        self.problem_label.pack(pady=10)

        self.solution_label = tk.Label(self, text="Agent's procedure will appear here.", font=("Helvetica", 14))
        self.solution_label.pack(pady=5)

        self.result_label = tk.Label(self, text="Computed result: ", font=("Helvetica", 14))
        self.result_label.pack(pady=5)

        self.correct_label = tk.Label(self, text="Correct answer: ", font=("Helvetica", 14))
        self.correct_label.pack(pady=5)

        self.feedback_label = tk.Label(self, text="", font=("Helvetica", 14, "bold"))
        self.feedback_label.pack(pady=5)

        self.log_text = tk.Text(self, height=12, width=80, font=("Courier", 12))
        self.log_text.pack(pady=10)

        self.solve_button = tk.Button(self, text="Solve Problem", command=self.solve_problem)
        self.solve_button.pack(pady=5)

        self.new_button = tk.Button(self, text="New Problem", command=self.new_problem)
        self.new_button.pack(pady=5)

    def new_problem(self):
        self.env.reset()
        self.state = self.env.get_state()
        self.problem_label.config(text=self.env.problem_text)
        self.correct_label.config(text=f"Correct answer: {self.env.correct_answer}")
        self.log_text.delete(1.0, tk.END)
        self.feedback_label.config(text="")
        self.solution_label.config(text="Agent's procedure will appear here.")
        self.result_label.config(text="Computed result: ")

    def solve_problem(self):
        self.log_text.delete(1.0, tk.END)
        self.env.chain = []
        self.env.chain_text = ""
        self.state = self.env.get_state()
        steps = []
        done = False
        while not done:
            action = self.agent.choose_action(self.state, self.env)
            step_text = self.env.allowed_actions[action]
            steps.append(step_text)
            self.log_text.insert(tk.END, f"Action: {step_text}\n")
            self.state, _, done, _ = self.env.step(action)
        self.solution_label.config(text=" -> ".join(steps))
        teacher = self.env.teacher
        computed_result = teacher.correct_answer if self.env.chain == teacher.correct_steps else "N/A"
        self.result_label.config(text=f"Computed result: {computed_result}")
        correct_answer = self.env.correct_answer
        self.correct_label.config(text=f"Correct answer: {correct_answer}")
        feedback, _ = teacher.evaluate_solution(self.env.chain)
        self.feedback_label.config(text=feedback)
        self.problem_label.config(text=self.env.problem_text)

#############################################
# Main Execution
#############################################

if __name__ == '__main__':
    print("Training RL agent for three-digit multiplication procedure...")
    trained_agent = train_agent(num_episodes=20000)
    print("Training complete. Launching GUI.")
    app = CompositeApp(trained_agent)
    app.mainloop()
