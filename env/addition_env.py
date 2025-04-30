import random

from teacher.addition_teacher import AdditionTeacherSimple
from utils.vocab import encode_state_text


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
        if done:
            feedback, final_reward = self.teacher.evaluate_solution(self.chain)
            reward += final_reward
            return self.get_state(), reward, done, feedback
        return self.get_state(), reward, done, ""
