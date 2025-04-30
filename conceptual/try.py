import torch
import torch.nn as nn
import torch.optim as optim
import random
import tkinter as tk

#############################################
# Global Vocabulary and Text Processing
#############################################

# Vocabulary covers key procedural words and digits.
vocab = [
           "multiply", "add", "carry", "and", "combine", "do", "nothing", "subtract",
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
   vec = torch.zeros(vocab_size)
   for token in tokens:
       if token in vocab_dict:
           vec[vocab_dict[token]] += 1.0
   return vec


def encode_state_text(problem_text, chain_text):
   return torch.cat([encode_text(problem_text), encode_text(chain_text)])



#############################################
# Policy Network (for Procedure Learning)
#############################################

class PolicyNetwork(nn.Module):
   def __init__(self, input_dim, hidden_dim, output_dim):
       """
       Input: concatenation of state vector (input_dim) and one-hot action vector (output_dim).
       Output: scalar score.
       """
       super(PolicyNetwork, self).__init__()
       self.fc1 = nn.Linear(input_dim + output_dim, hidden_dim)
       self.fc2 = nn.Linear(hidden_dim, 1)

   def forward(self, state_action):
       x = torch.relu(self.fc1(state_action))
       return self.fc2(x)


def generate_addition_problem_simple():
   A = random.randint(10, 99)
   B = random.randint(10, 99)
   correct_answer = A + B
   problem_text = f"add: add {A} + {B}"
   ones_A = A % 10
   ones_B = B % 10
   tens_A = A // 10
   tens_B = B // 10
   S1 = ones_A + ones_B
   carry = S1 // 10
   step1 = f"add ones: {ones_A} and {ones_B}"
   step2 = f"add tens: {tens_A} and {tens_B} with carry {carry}"
   correct_steps = [step1.lower(), step2.lower()]
   # Pad allowed actions to 7 options.
   allowed_actions = [step1.lower(), step2.lower(), "do nothing", "subtract", "add tens", "dummy1", "dummy2"]
   return problem_text.lower(), allowed_actions, correct_steps, correct_answer, A, B


class AdditionTeacherSimple:
   def __init__(self, correct_steps, correct_answer, A, B):
       self.correct_steps = correct_steps
       self.correct_answer = correct_answer
       self.A = A
       self.B = B

   def parse_add_ones(self, step_text):
       tokens = step_text.split()
       if len(tokens) != 5:
           return None, None
       try:
           return int(tokens[2]), int(tokens[4])
       except:
           return None, None

   def parse_add_tens(self, step_text):
       tokens = step_text.split()
       if len(tokens) != 8:
           return None, None, None
       try:
           return int(tokens[2]), int(tokens[4]), int(tokens[7])
       except:
           return None, None, None

   def compute_sum_from_chain(self, chain):
       if len(chain) < 2:
           return None
       X, Y = self.parse_add_ones(chain[0])
       if X is None:
           return None
       S1 = X + Y
       carry = S1 // 10
       ones_res = S1 % 10
       T, U, C = self.parse_add_tens(chain[1])
       if T is None:
           return None
       if C != carry:
           return None
       S2 = T + U + carry
       return S2 * 10 + ones_res

   def evaluate_solution(self, chain):
       reward = 0.0
       for i, correct_step in enumerate(self.correct_steps):
           if i < len(chain):
               reward += 5 if chain[i] == correct_step else -2
       computed_sum = self.compute_sum_from_chain(chain)
       if computed_sum is None:
           reward -= 5
           feedback = "Addition procedure invalid."
       else:
           if computed_sum == self.correct_answer:
               reward += 10
               feedback = f"Addition correct! Sum: {computed_sum}."
           else:
               error = abs(computed_sum - self.correct_answer)
               reward -= error
               feedback = f"Procedure may be ok, but computed sum is off (got {computed_sum})."
       return feedback, reward


class AdditionEnvSimple:
   def __init__(self):
       self.reset()

   def reset(self):
       (self.problem_text, self.allowed_actions,
        self.correct_steps, self.correct_answer, self.A, self.B) = generate_addition_problem_simple()
       self.num_actions = len(self.allowed_actions)
       self.teacher = AdditionTeacherSimple(self.correct_steps, self.correct_answer, self.A, self.B)
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

class RLAgentSimple:
   def __init__(self, lr=1e-3, state_dim=2 * vocab_size, num_actions=5):
       self.num_actions = num_actions
       self.state_dim = state_dim
       self.policy_net = PolicyNetwork(input_dim=state_dim, hidden_dim=64, output_dim=num_actions)
       self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
       self.episode_log_probs = []
       self.cumulative_loss = 0.0

   def choose_action(self, state, env):
       action_scores = []
       for action_index in range(env.num_actions):
           action_tensor = torch.zeros(env.num_actions)
           action_tensor[action_index] = 1.0
           input_tensor = torch.cat([state, action_tensor])
           score = self.policy_net(input_tensor)
           action_scores.append(score)
       scores_tensor = torch.stack(action_scores).view(-1)
       probs = torch.softmax(scores_tensor, dim=0)
       dist = torch.distributions.Categorical(probs)
       action = dist.sample()
       self.episode_log_probs.append(dist.log_prob(action))
       return action.item()

   def accumulate_loss(self, reward):
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
# Composite Environment (Randomly Choose Task)
#############################################

class CompositeMathEnv:
   def __init__(self):
       self.task = None  # "mul" or "add"
       self.env = None
       self.reset()

   def reset(self):
       self.task = "add"
       self.env = AdditionEnvSimple()
       return self.env.get_state()

   def get_state(self):
       return self.env.get_state()

   def step(self, action_index):
       return self.env.step(action_index)

def train_agent(num_episodes=5000):
   env = CompositeMathEnv()
   agent = RLAgentSimple(lr=1e-3, state_dim=2 * vocab_size, num_actions=env.env.num_actions)
   for episode in range(num_episodes):
       state = env.reset()
       done = False
       total_proc_reward = 0.0
       while not done:
           action = agent.choose_action(state, env.env)
           next_state, reward, done, _ = env.step(action)
           agent.accumulate_loss(reward)
           total_proc_reward += reward
           state = next_state
       teacher = env.env.teacher
       # For multiplication, we simply check if the produced chain exactly matches
       # the expert chain. For addition, the teacher still computes a sum.
       if env.task == "mul":
           teacher_feedback, teacher_reward = teacher.evaluate_solution(env.env.chain)
       else:
           teacher_feedback, teacher_reward = teacher.evaluate_solution(env.env.chain)
       total_final_reward = total_proc_reward + teacher_reward
       agent.finalize_episode(total_final_reward)
       if (episode + 1) % 100 == 0:
           if env.task == "mul":
               # For multiplication, report the stored correct answer if chain is perfect.
               computed = teacher.correct_answer if env.env.chain == teacher.correct_steps else "N/A"
           else:
               computed = teacher.compute_sum_from_chain(env.env.chain)
           print(
               f"Episode {episode + 1} [{env.task}], Proc Reward: {total_proc_reward:.2f}, Teacher Reward: {teacher_reward:.2f}, Computed: {computed}, True: {env.env.correct_answer}")
   return agent



#############################################
# Main Execution
#############################################

if __name__ == '__main__':

    trained_agent = train_agent(10)

