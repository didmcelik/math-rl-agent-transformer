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
