class MultiplicationTeacherThree:
    """
    Revised Multiplication Teacher.
    Instead of parsing numbers and performing arithmetic, it only compares
    the agent’s chain of actions (text strings) with the pre-generated correct chain.
    This forces the RL agent to learn the full procedure from the text tokens.
    """

    def __init__(self, correct_steps, correct_answer, A, M):
        self.correct_steps = correct_steps
        self.correct_answer = correct_answer
        self.A = A
        self.M = M

    def evaluate_solution(self, chain):
        reward = 0.0
        # Compare each step with expected text
        for i, correct_step in enumerate(self.correct_steps):
            if i < len(chain):
                reward += 5 if chain[i] == correct_step else -2
        # Give bonus reward if the entire chain exactly matches the expert chain.
        if chain == self.correct_steps:
            reward += 10
            feedback = f"Multiplication correct! Product: {self.correct_answer}."
        else:
            feedback = f"Procedure incorrect. Expected: {self.correct_steps}, but got: {chain}."
        return feedback, reward
