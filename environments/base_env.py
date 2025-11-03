from typing import Any, Tuple, Dict
import sympy
from sympy.logic.boolalg import BooleanFunction, And, Or, Not
import random
from multiprocessing import Process, Queue

def _apply_rule_wrapper(rule, expr, queue, *args):
    try:
        if args:
            result = rule(expr, *args)
        else:
            result = rule(expr)
        queue.put(result)
    except Exception as e:
        queue.put(e)

def _simplify_logic_default(expr):
    return sympy.simplify_logic(expr)

class BaseBooleanEnv:
    def __init__(self, max_expression_depth: int, max_literals: int, max_steps: int):
        self.max_expression_depth = max_expression_depth
        self.max_literals = max_literals
        self.max_steps = max_steps

        self.literals = [sympy.Symbol(chr(ord('A') + i)) for i in range(max_literals)]
        self.current_expression = None
        self.initial_complexity = 0
        self.known_best_complexity = 0
        self.steps_taken = 0
        self.history = []

    def _generate_random_expr(self, depth: int):
        if depth == 0 or random.random() < 0.3:
            return random.choice(self.literals)
        else:
            op_types = [And, Or, Not]
            op_select = random.choice(op_types)

            if op_select == Not:
                return op_select(self._generate_random_expr(depth - 1))
            else:
                arg1 = self._generate_random_expr(depth - 1)
                arg2 = self._generate_random_expr(depth - 1)
                return op_select(arg1, arg2)

    def _get_complexity(self, expr) -> int:
        if isinstance(expr, sympy.Symbol):
            return 1
        elif isinstance(expr, BooleanFunction):
            complexity = 1
            for arg in expr.args:
                complexity += self._get_complexity(arg)
            return complexity
        else:
            return 0

    def _get_state(self):
        raise NotImplementedError

    def get_action_size(self) -> int:
        raise NotImplementedError

    def step(self, action: int):
        raise NotImplementedError

    def reset(self, max_retries: int = 10):
        for _ in range(max_retries):
            self.current_expression = self._generate_random_expr(self.max_expression_depth)
            if isinstance(self.current_expression, sympy.Symbol) or not self.current_expression.args:
                continue

            self.initial_complexity = self._get_complexity(self.current_expression)

            q = Queue()
            p = Process(target=_apply_rule_wrapper, args=(_simplify_logic_default, self.current_expression, q))
            p.start()
            p.join(5)

            if p.is_alive():
                p.terminate()
                p.join()
                continue

            result = q.get()
            if isinstance(result, Exception):
                continue

            self.known_best_complexity = self._get_complexity(result)

            if self.initial_complexity > self.known_best_complexity:
                self.steps_taken = 0
                self.history = [self.current_expression]
                return self._get_state()

        self.current_expression = self.literals[0] & self.literals[1]
        self.initial_complexity = self._get_complexity(self.current_expression)
        self.known_best_complexity = self._get_complexity(sympy.simplify_logic(self.current_expression))
        self.steps_taken = 0
        self.history = [self.current_expression]
        return self._get_state()