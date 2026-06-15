# LowThrustTransfer — optimize a low-thrust orbit transfer

## Scientific background

Low-thrust propulsion (ion engines, solar sails) enables fuel-efficient interplanetary
trajectories but requires continuous thrust optimization over long arcs. Finding the optimal
thrust profile is a challenging optimal-control problem with applications to deep-space
missions (Dawn, BepiColombo, Psyche).

## Your task

Edit **`solution.py`** to define:

```python
def design_trajectory(r0, v0, rf, vf, T_max, t_final, n_steps):
    """Return (n_steps, 3) thrust vectors [ax, ay, az] in m/s^2.
    |a| <= T_max at each step. Minimize fuel = sum(|a|) * dt while
    reaching the target orbit (rf, vf) from (r0, v0)."""
```

The evaluator propagates the trajectory under Keplerian gravity + your thrust, measures the
final-state error and total delta-v. Lower fuel with acceptable arrival error scores higher.

## Scoring

```
score = clip( (fuel_baseline - fuel_found) / (fuel_baseline - fuel_optimal), 0, 1 )
```
penalized by arrival error. Baseline: constant-thrust Hohmann-like.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU, seconds. Do not read `verification/`.
