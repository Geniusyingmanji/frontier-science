# ConvectionDiffusionOpt — place heat sources to match a target temperature field

## Scientific background
In PDE-constrained optimization, the forward model (here: steady-state convection-diffusion
equation) must be solved for each candidate design. The agent places 6 heat sources (positions
+ strengths) to produce a temperature field matching a hidden target. The convection term
breaks symmetry and introduces upwind transport effects, making the inverse problem non-trivial.

Reference: Hinze et al., Optimization with PDE Constraints (Springer, 2009).

## Your task
```python
def place_sources(nx, ny, n_sources):
    \"\"\"Return dict with 'positions' (6, 2) in [0,1]² and 'strengths' (6,) > 0.\"\"\"
```
## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
