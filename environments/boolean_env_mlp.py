from typing import Tuple, Dict, Any
import sympy
from sympy.logic.boolalg import And, Or, Not, Equivalent, Implies, Xor
import numpy as np
from multiprocessing import Process, Queue

from .base_env import BaseBooleanEnv, _apply_rule_wrapper

def _simplify_logic(expr) -> sympy.Basic:
    return sympy.simplify_logic(expr)

def _simplify_logic_dnf(expr) -> sympy.Basic:
    return sympy.simplify_logic(expr, form='dnf')

def _simplify_logic_cnf(expr) -> sympy.Basic:
    return sympy.simplify_logic(expr, form='cnf')

def _to_anf(expr) -> sympy.Basic:
    return sympy.logic.boolalg.to_anf(expr)

def _to_cnf(expr) -> sympy.Basic:
    return sympy.logic.boolalg.to_cnf(expr)

def _to_dnf(expr) -> sympy.Basic:
    return sympy.logic.boolalg.to_dnf(expr)

def _to_nnf(expr) -> sympy.Basic:
    return sympy.logic.boolalg.to_nnf(expr)

class BooleanSimplificationEnv(BaseBooleanEnv):
    def __init__(self, max_expression_depth: int, max_literals: int, max_steps: int):
        super().__init__(max_expression_depth, max_literals, max_steps)
        self.action_space_size = len(self._get_available_rules())
        self.reset()

    def _get_state(self) -> np.ndarray:
        count_literals = len(self.current_expression.atoms(sympy.Symbol))
        count_and = len(self.current_expression.atoms(And))
        count_or = len(self.current_expression.atoms(Or))
        count_not = len(self.current_expression.atoms(Not))
        count_equivalent = len(self.current_expression.atoms(Equivalent))
        count_implies = len(self.current_expression.atoms(Implies))
        count_xor = len(self.current_expression.atoms(Xor))

        def get_depth(expr) -> int:
            if not hasattr(expr, 'args') or not expr.args:
                return 0
            return 1 + max(get_depth(arg) for arg in expr.args) if expr.args else 0

        depth = get_depth(self.current_expression)
        current_complexity = self._get_complexity(self.current_expression)

        state = np.array([count_literals, count_and, count_or, count_not, count_equivalent, count_implies, count_xor, depth, current_complexity])
        return state

    def get_state_size(self) -> int:
        return len(self._get_state())

    def get_action_size(self) -> int:
        return len(self._get_available_rules())

    def _get_available_rules(self):
        return [
            _simplify_logic,
            _simplify_logic_dnf,
            _simplify_logic_cnf,
            _to_anf,
            _to_cnf,
            _to_dnf,
            _to_nnf,
        ]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.steps_taken += 1

        rules = self._get_available_rules()
        if not (0 <= action < len(rules)):
            return self._get_state(), -10.0, True, {}

        old_complexity = self._get_complexity(self.current_expression)

        q = Queue()
        p = Process(target=_apply_rule_wrapper, args=(rules[action], self.current_expression, q))
        p.start()
        p.join(10)

        if p.is_alive():
            p.terminate()
            p.join()
            print("--- SymPy operation timed out ---")
            return self._get_state(), -5.0, True, {}

        result = q.get()
        if isinstance(result, Exception):
            print(f"--- SymPy operation failed with error: {result} ---")
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