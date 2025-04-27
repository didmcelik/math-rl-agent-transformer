import random

class MultiplicationEnvTwoTwo:
    def __init__(self):
        self.reset()

    def reset(self):
        self.A = random.randint(10, 99)
        self.B = random.randint(10, 99)
        self.problem_text = f"multiply {self.A} x {self.B}"
        self.correct_answer = self.A * self.B
        self.chain = []
        self.chain_text = ""
        self.steps = self.generate_steps()
        self.current_step = 0
        self.done = False
        return self.get_state()

    def generate_steps(self):
        ones_A = self.A % 10
        tens_A = self.A // 10
        ones_B = self.B % 10
        tens_B = self.B // 10

        steps = []
        P1 = ones_A * ones_B
        carry1 = P1 // 10
        steps.append(f"multiply {ones_A} by {ones_B} output {P1 % 10} carry {carry1}")

        P2 = tens_A * ones_B + carry1
        carry2 = P2 // 10
        steps.append(f"multiply {tens_A} by {ones_B} output {P2 % 10} carry {carry2}")

        P3 = ones_A * tens_B
        carry3 = P3 // 10
        steps.append(f"multiply {ones_A} by {tens_B} output {P3 % 10} carry {carry3}")

        P4 = tens_A * tens_B + carry3
        carry4 = P4 // 10
        steps.append(f"multiply {tens_A} by {tens_B} output {P4 % 10} carry {carry4}")

        steps.append("output final result")
        return steps

    def get_state(self):
        return self.problem_text, self.chain_text

    def step(self, action_text):
        if self.done:
            raise Exception("Episode is done. Please reset.")

        self.chain.append(action_text)
        self.chain_text = (self.chain_text + " " + action_text).strip()
        reward = 0.0
        done = False

        if self.current_step < len(self.steps):
            expected_step = self.steps[self.current_step]
            if action_text.strip().lower() == expected_step.strip().lower():
                reward = 1.0
            else:
                reward = -0.5

        self.current_step += 1
        if self.current_step >= len(self.steps):
            done = True
            self.done = True

        return self.get_state(), reward, done
