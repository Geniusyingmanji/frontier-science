# DistillationColumnDesign — optimize a binary distillation column

## Scientific background
Distillation is the workhorse of chemical separations. Designing a binary column (number of
trays, reflux ratio, feed tray location) to meet product purity specs while minimizing
energy (reboiler duty) involves understanding the McCabe-Thiele operating line + equilibrium
curve interaction. The optimal design balances capital cost (trays) vs operating cost (reflux)
with a sharp sensitivity to feed tray location.

Reference: McCabe & Smith, Unit Operations of Chemical Engineering (8th ed, 2018).

## Your task
```python
def design_column(alpha, x_feed, x_dist_target, x_bot_target):
    \"\"\"Return dict with 'n_trays' (3-60), 'reflux_ratio' (0.5-20), 'feed_tray' (1 to n-1).\"\"\"
```
## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
