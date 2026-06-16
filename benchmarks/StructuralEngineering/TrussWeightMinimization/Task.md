# TrussWeightMinimization — optimize a 10-bar truss under stress constraints

## Scientific background
The 10-bar truss (Schmit & Farshi 1974) is the most-cited structural optimization benchmark:
6 nodes, 10 bars, 2 point loads. The goal is to minimize total weight by sizing bar
cross-sectional areas while keeping all member stresses below the allowable limit. The
coupling between member forces (statically indeterminate) makes naive sizing suboptimal.

Reference: Schmit & Farshi, AIAA J. 12(2), 1974; Haftka & Gürdal 1992.

## Your task
```python
def design_truss(n_bars):
    \"\"\"Return cross-sectional areas (10,) in [0.1, 35.0] in^2.
    Minimize total weight while keeping |stress| <= 25000 psi in all bars.\"\"\"
```

## Scoring
`score = clip((weight_max - weight_found) / (weight_max - weight_optimal), 0, 1)`
All-max-area → score 0. Literature optimum 5060 lbs → score 1.0.

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
