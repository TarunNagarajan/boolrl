# Reward Function Design Evolution

The reward function for the Boolean Simplification RL environment underwent three design iterations to address fundamental issues with reward engineering.

## Version 1: Fixed Target Approach

The initial implementation assumed a canonical simplified form exists for each boolean expression.

Method:
1. Generate random expression at episode start
2. Apply `sympy.simplify_logic` to establish `target_simplified_expression`
3. Reward based on exact string match with target, with minor rewards for complexity reduction

Limitation:
Boolean expressions have multiple equivalent simplified forms. The agent was penalized for discovering valid alternatives that differed from SymPy's choice, creating a brittle objective.

---

## Version 2: Scaled Progress Reward

The second iteration addressed the fixed target issue by rewarding based on complexity reduction rather than specific representations.

Method:
1. Eliminate `target_simplified_expression`
2. Calculate `known_best_complexity` from SymPy as performance benchmark
3. Implement shaped reward function:
    *   Goal bonus (+50.0) for achieving target complexity
    *   Proportional reward based on reduction ratio
    *   Penalties for increasing complexity or no change
    *   Efficiency penalty for excessive steps

Limitation:
The ratio-based reward enabled exploitation. Agents could dramatically increase complexity to expand the "improvement space," then receive large rewards for subsequent simplifications, resulting in high scores for counterproductive behavior.

---

## Version 3: Direct Difference Reward

The final implementation eliminates scaling to prevent reward hacking while maintaining effective training signals.

Method:
1. Base reward on direct difference: `old_complexity - new_complexity`
2. Retain goal bonus (+50.0) for achieving target complexity
3. Apply penalties for no change, inefficiency, and timeout

The final design is robust against exploitation while directly incentivizing complexity reduction.
