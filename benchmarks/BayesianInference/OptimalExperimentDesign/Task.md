# OptimalExperimentDesign — select experiments to maximize information

## Scientific background
In Bayesian/frequentist experimental design, selecting where to measure maximizes information
about model parameters. The D-optimal criterion maximizes log|FIM| (Fisher Information Matrix),
minimizing the parameter uncertainty ellipsoid. For nonlinear models, this is a non-convex
problem on a continuous design space. It underpins clinical trials, sensor placement, and
materials characterization.

Reference: Chaloner & Verdinelli, Stat. Sci. 10, 273 (1995); Fedorov 1972.

## Your task
```python
def select_designs(k, d_lo, d_hi, n_params):
    \"\"\"Select k design points in [d_lo, d_hi] to maximize information.
    Model: polynomial + trigonometric (6 parameters). Return (k,) array.\"\"\"
```

## Scoring
EIG improvement over uniform spacing, normalized vs a strong random search reference.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
