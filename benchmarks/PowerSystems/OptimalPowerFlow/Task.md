# OptimalPowerFlow — minimize generation cost in a power network

## Scientific background
Optimal Power Flow (OPF) determines the least-cost generator dispatch that satisfies load
demand, Kirchhoff's laws, and thermal line limits. Even the DC approximation (linear power
flow) with quadratic costs is non-trivial when line constraints bind, creating a constrained
QP with network coupling. OPF is solved millions of times daily by power system operators
worldwide and is central to electricity market clearing.

Reference: Carpentier, Bull. Soc. Française Électriciens 3, 431 (1962); Dommel & Tinney 1968.

## Your task
```python
def solve_opf(n_bus, n_gen, demand, gen_pmax, gen_pmin, cost_a, cost_b, lines, line_limits):
    \"\"\"Return generator outputs (n_gen,) that minimize quadratic cost while satisfying
    power balance and line flow limits. 6-bus system with 3 generators.\"\"\"
```

## Scoring
Cost reduction vs equal-dispatch baseline, penalized for line violations.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
