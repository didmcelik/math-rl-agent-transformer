class MultiplicationTeacherTwoTwo:
    """
    Teacher class for 2-digit x 2-digit multiplication.
    Compares the agent’s generated chain of text actions with the pre-generated correct chain.
    Provides reward based on step-by-step correctness and overall full solution match.
    """

    def __init__(self, correct_steps, correct_answer, A, B):
        self.correct_steps = correct_steps
        self.correct_answer = correct_answer
        self.A = A
        self.B = B

    def evaluate_solution(self, chain):
        reward = 0.0
        # Step-by-step reward
        for i, correct_step in enumerate(self.correct_steps):
            if i < len(chain):
                if chain[i].strip().lower() == correct_step.strip().lower():
                    reward += 5.0
                else:
                    reward -= 2.0

        # Bonus reward if complete correct chain
        if len(chain) == len(self.correct_steps) and all(
            chain[i].strip().lower() == self.correct_steps[i].strip().lower()
            for i in range(len(self.correct_steps))
        ):
            reward += 10.0
            feedback = f"Multiplication correct! Product: {self.correct_answer}."
        else:
            feedback = f"Procedure incorrect. Expected: {self.correct_steps}, but got: {chain}."

        return feedback, reward
