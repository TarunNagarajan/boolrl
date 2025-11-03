from typing import Tuple, Dict, Any
import sympy
from sympy.logic.boolalg import And, Or, Not, Equivalent, Implies, Xor, BooleanFunction
import numpy as np

from .base_env import BaseBooleanEnv

class BooleanSimplificationEnvSeq(BaseBooleanEnv):
    def __init__(self, max_expression_depth: int, max_literals: int, max_steps: int, max_seq_len: int = 128):
        super().__init__(max_expression_depth, max_literals, max_steps)
        self.max_seq_len = max_seq_len

        self.op_map = {And: 1, Or: 2, Not: 3, Equivalent: 4, Implies: 5, Xor: 6, sympy.true: 7, sympy.false: 8}
        self.literal_map = {l: i + 9 for i, l in enumerate(self.literals)}
        self.vocab = {**self.op_map, **self.literal_map}
        self.vocab_size = len(self.vocab) + 1  # +1 for padding token

        self.action_space_size = 7 
        self.reset()

    def _traverse_and_tokenize(self, expr) -> list:
        tokens = []
        def _traverse(sub_expr):
            if isinstance(sub_expr, sympy.Symbol):
                tokens.append(self.literal_map.get(sub_expr, 0))
            elif isinstance(sub_expr, BooleanFunction):
                tokens.append(self.op_map.get(type(sub_expr), 0))
                for arg in sub_expr.args:
                    _traverse(arg)
            else: 
                tokens.append(self.op_map.get(type(sub_expr), 0))
        _traverse(expr)
        return tokens

    def _get_state(self) -> np.ndarray:
        tokens = self._traverse_and_tokenize(self.current_expression)
        padded_tokens = np.zeros(self.max_seq_len, dtype=np.int32)
        seq_len = min(len(tokens), self.max_seq_len)
        padded_tokens[:seq_len] = tokens[:seq_len]
        return padded_tokens

    def get_state_size(self) -> int:
        return self.max_seq_len

    def get_action_size(self) -> int:
        return self.action_space_size

    def _get_available_rules(self):
        return [
            sympy.simplify_logic,
            lambda e: sympy.simplify_logic(e, form='dnf'),
            lambda e: sympy.simplify_logic(e, form='cnf'),
            sympy.logic.boolalg.to_anf,
            sympy.logic.boolalg.to_cnf,
            sympy.logic.boolalg.to_dnf,
            sympy.logic.boolalg.to_nnf,
        ]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.steps_taken += 1

        rules = self._get_available_rules()
        if not (0 <= action < len(rules)):
            return self._get_state(), -10.0, True, {}

        old_complexity = self._get_complexity(self.current_expression)

        from .base_env import _apply_rule_wrapper
        from multiprocessing import Queue, Process

        q = Queue()
        p = Process(target=_apply_rule_wrapper, args=(rules[action], self.current_expression, q))
        p.start()
        p.join(10)

        if p.is_alive():
            p.terminate()
            p.join()
            return self._get_state(), -5.0, True, {}

        result = q.get()
        if isinstance(result, Exception):
            return self._get_state(), -5.0, True, {}

        self.current_expression = result
        self.history.append(self.current_expression)
        new_complexity = self._get_complexity(self.current_expression)

        reward = 0.0
        done = False

        complexity_reduction = old_complexity - new_complexity

        if new_complexity < self.known_best_complexity:
            reward += 100.0
            self.known_best_complexity = new_complexity
            done = True
        elif new_complexity == self.known_best_complexity:
            reward += 50.0
            done = True
        elif complexity_reduction > 0:
            reward += 10.0 * complexity_reduction
        else:
            reward -= 2.0

        reward -= 1.0

        if self.steps_taken >= self.max_steps:
            done = True
            if new_complexity > self.known_best_complexity:
                reward -= 20.0

        return self._get_state(), reward, done, {'history': self.history}
