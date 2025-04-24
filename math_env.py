
import sympy as sp

class MathEnv:
    """
    Strict symbolic math RL environment with:
    - Must start with 'x='
    - Numeric part must be digits only and max 3 digits
    - Reward based on correctness and closeness
    - Length penalty for over-generation
    - Token diversity exploration bonus
    """
    def __init__(self, problem_str="x+2=5", max_steps=20):
        self.problem_str = problem_str
        self.max_steps = max_steps
        self.reset()

        try:
            x = sp.Symbol('x')
            left, right = self.problem_str.split("=")
            lhs = sp.sympify(left.strip())
            rhs = sp.sympify(right.strip())
            self.ground_truth = float(sp.solve(lhs - rhs, x)[0])
        except Exception:
            self.ground_truth = None

    def reset(self):
        self.tokens = []
        self.current_step = 0
        self.done = False
        return self.tokens

    def step(self, token):
        if self.done:
            raise Exception("Episode has finished. Call reset() to start a new one.")

        self.tokens.append(token)
        self.current_step += 1

        if self._check_done() or self.current_step >= self.max_steps:
            self.done = True
            reward = self._compute_reward()
        else:
            reward = self._exploration_bonus()

        return self.tokens, reward, self.done

    def _check_done(self):
        eq_str = "".join(self.tokens)
        if eq_str.startswith("x="):
            rhs = eq_str[2:]
            if rhs.isdigit() and 1 <= len(rhs) <= 3:
                return True
        return False

    def _compute_reward(self):
        try:
            eq_str = "".join(self.tokens)

            # Must start with 'x='
            if not eq_str.startswith("x="):
                return -0.2

            # Check right side of x=
            rhs = eq_str[2:]
            if not rhs.isdigit() or len(rhs) > 3:
                return -0.3  # invalid format or too long

            predicted_value = int(rhs)
            if self.ground_truth is not None:
                difference = abs(predicted_value - self.ground_truth)
                if difference < 0.1:
                    base_reward = 1.0
                elif difference < 1.0:
                    base_reward = 0.5
                else:
                    base_reward = 0.1
            else:
                base_reward = 1.0

            # Structure bonus
            structure_bonus = 0.3

            # Length penalty
            length_penalty = -0.05 * max(0, len(self.tokens) - 6)

            return base_reward + structure_bonus + length_penalty

        except Exception:
            return -0.2

    def _exploration_bonus(self):
        unique = len(set(self.tokens))
        total = len(self.tokens)
        return 0.05 * (unique / total) if total > 0 else 0.0
