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
        self.allowed_actions = self.steps
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

        # First row
        P1 = ones_A * ones_B
        digit1 = P1 % 10
        carry1 = P1 // 10
        steps.append(f"multiply A_ones {ones_A} by B_ones {ones_B} -> output digit1 {digit1}, carry1 {carry1}")

        P2 = tens_A * ones_B
        d2_intermediate = P2
        d2_total = d2_intermediate + carry1
        digit2 = d2_total % 10
        carry2 = d2_total // 10
        steps.append(
            f"multiply A_tens {tens_A} by B_ones {ones_B} and add carry1 {carry1} -> output digit2 {digit2}, carry2 {carry2}")

        # Build first partial result
        first_row = carry2 * 100 + digit2 * 10 + digit1
        steps.append(
            f"build first partial result using carry2 {carry2}, digit2 {digit2}, and digit1 {digit1} -> {first_row}")

        # Second row
        P3 = ones_A * tens_B
        digit3 = P3 % 10
        carry3 = P3 // 10
        steps.append(f"multiply A_ones {ones_A} by B_tens {tens_B} -> output digit3 {digit3}, carry3 {carry3}")

        P4 = tens_A * tens_B
        d4_intermediate = P4
        d4_total = d4_intermediate + carry3
        digit4 = d4_total % 10
        carry4 = d4_total // 10
        steps.append(
            f"multiply A_tens {tens_A} by B_tens {tens_B} and add carry3 {carry3} -> output digit4 {digit4}, carry4 {carry4}")

        # Build second partial result
        second_row = (carry4 * 100 + digit4 * 10 + digit3) * 10
        steps.append(
            f"build second partial result using carry4 {carry4}, digit4 {digit4}, and digit3 {digit3} -> {second_row}")

        # Add partial results
        steps.append(f"add partial results {first_row} and {second_row}")

        # Final result
        steps.append(f"output final result")

        return steps

    def get_state(self):
        return self.problem_text, self.chain_text

    def step(self, action_text):
        if self.done:
            raise Exception("Episode is done. Please reset.")

        self.chain.append(action_text)
        self.chain_text = (self.chain_text + " " + action_text).strip()

        reward = 0.0
        current_step = len(self.chain) - 1
        if current_step < len(self.steps):
            expected_action = self.steps[current_step]
            if action_text.strip().lower() == expected_action.strip().lower():
                reward = 5.0
            else:
                reward = -2.0
        else:
            # Çoktan tüm adımlar bitti, ama agent hâlâ aksiyon basıyor
            reward = -5.0

        self.current_step += 1
        done = (self.current_step >= len(self.steps))

        if done:
            feedback, teacher_reward = self.teacher.evaluate_solution(self.chain)
            reward += teacher_reward
            self.done = True

        return self.get_state(), reward, done



def test():
    env = MultiplicationEnvTwoTwo()
    env.reset()
    for action in env.allowed_actions:
        print(action)

