import random
from utils.vocab import Vocab
from env.teacher_two_two import MultiplicationTeacherTwoTwo

class MultiplicationEnvTwoTwo:
    def __init__(self):
        self.vocab = Vocab()
        self.reset()

    def reset(self):
        self.A = random.randint(10, 99)
        self.B = random.randint(10, 99)
        self.problem_text = f"multiply {self.A} x {self.B}"
        self.correct_answer = self.A * self.B
        self.chain = []
        self.chain_text = ""
        self.steps = self.generate_steps()
        self.allowed_actions = self.steps + ["do nothing"]
        self.current_step = 0
        self.done = False

        # NEW: Create teacher instance
        self.teacher = MultiplicationTeacherTwoTwo(self.steps, self.correct_answer, self.A, self.B)

        return self.get_state()

    def generate_steps(self):
        ones_A = self.A % 10
        tens_A = self.A // 10
        ones_B = self.B % 10
        tens_B = self.B // 10

        steps = []

        # First row: multiply A by ones place of B
        # Ones place
        P1 = ones_A * ones_B
        d1 = P1 % 10
        carry1 = P1 // 10
        steps.append(f"multiply {ones_A} by {ones_B} output {d1} carry {carry1}")

        # Tens place
        P2 = tens_A * ones_B + carry1
        d2 = P2 % 10
        carry2 = P2 // 10
        steps.append(f"multiply {tens_A} by {ones_B} output {d2} carry {carry2}")

        # Remember the first partial result (first row)
        first_row = d1 + d2 * 10 + carry2 * 100  # Build number like 161

        # Second row: multiply A by tens place of B
        # Ones place
        P3 = ones_A * tens_B
        d3 = P3 % 10
        carry3 = P3 // 10
        steps.append(f"multiply {ones_A} by {tens_B} output {d3} carry {carry3}")

        # Tens place
        P4 = tens_A * tens_B + carry3
        d4 = P4 % 10
        carry4 = P4 // 10
        steps.append(f"multiply {tens_A} by {tens_B} output {d4} carry {carry4}")

        # Remember second partial result (shifted)
        second_row = (d3 + d4 * 10 + carry4 * 100) * 10  # shift left by one digit

        steps.append(f"add partial results {first_row} and {second_row}")

        steps.append(f"output final result")

        return steps

    def get_state(self):
        return self.problem_text, self.chain_text

    def step(self, action_text):
        if self.done:
            raise Exception("Episode is done. Please reset.")

        self.chain.append(action_text)
        self.chain_text = (self.chain_text + " " + action_text).strip()

        self.current_step += 1
        done = self.current_step >= len(self.steps)

        if done:
            feedback, reward = self.teacher.evaluate_solution(self.chain)
            self.done = True
        else:
            feedback, reward = "", 0.0  # intermediate steps get no reward

        return self.get_state(), reward, done
