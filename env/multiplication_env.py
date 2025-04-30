import random

from teacher.multiplication_teacher import MultiplicationTeacherThree
from utils.vocab import encode_state_text


def generate_multiplication_problem_three():
    """
    Generates a multiplication problem:
      - A: three-digit number (100-999)
      - M: one-digit number (1-9)
    A human-like six-step procedure is created as follows:
      1. "multiply <ones> by <M>"
      2. "multiply <tens> by <M> with carry <carry1>"
      3. "multiply <hundreds> by <M> with carry <carry2>"
      4. "output ones digit: <d1>"
      5. "output tens digit: <d2>"
      6. "output hundreds digits: <P3>"
    where the correct intermediate tokens are generated using arithmetic.

    Note: The multiplication teacher will no longer recompute these values;
          it only compares the agent’s chain (a sequence of text tokens) to the correct chain.
    """
    A = random.randint(100, 999)
    M = random.randint(1, 9)
    correct_answer = A * M
    problem_text = f"mul: multiply {A} x {M}"

    ones = A % 10
    tens = (A // 10) % 10
    hundreds = A // 100

    # Step 1:
    P1 = ones * M
    d1 = P1 % 10
    carry1 = P1 // 10
    step1 = f"multiply {ones} by {M}"

    # Step 2:
    P2 = tens * M + carry1
    d2 = P2 % 10
    carry2 = P2 // 10
    step2 = f"multiply {tens} by {M} with carry {carry1}"

    # Step 3:
    P3 = hundreds * M + carry2
    step3 = f"multiply {hundreds} by {M} with carry {carry2}"

    # Output steps:
    step4 = f"output ones digit: {d1}"
    step5 = f"output tens digit: {d2}"
    step6 = f"output hundreds digits: {P3}"

    # The correct procedure as a chain of strings (all lowercase)
    correct_steps = [step1.lower(), step2.lower(), step3.lower(), step4.lower(), step5.lower(), step6.lower()]
    # Allowed actions: the 6 correct ones, plus one extra dummy option to pad to a total of 7.
    allowed_actions = correct_steps + ["dummy"]
    return problem_text.lower(), allowed_actions, correct_steps, correct_answer, A, M


class MultiplicationEnvThree:
    def __init__(self):
        self.reset()

    def reset(self):
        (self.problem_text, self.allowed_actions,
         self.correct_steps, self.correct_answer, self.A, self.M) = generate_multiplication_problem_three()
        self.num_actions = len(self.allowed_actions)  # 7 actions now
        self.teacher = MultiplicationTeacherThree(self.correct_steps, self.correct_answer, self.A, self.M)
        self.chain = []
        self.chain_text = ""
        self.max_steps = len(self.correct_steps)  # 6 steps expected
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
