# LotkaVolterraCalibration — fit predator-prey model parameters to data

## Scientific background

The Lotka-Volterra equations model predator-prey population dynamics:
dx/dt = α x - β x y, dy/dt = δ x y - γ y. Fitting these to noisy field data is a
core inverse problem in mathematical ecology and epidemiology. The parameter landscape
is multimodal and sensitive to initial conditions.

## Your task

```python
def calibrate(t_obs, x_obs, y_obs):
    """Given observation times t_obs and noisy prey/predator counts x_obs, y_obs,
    return (alpha, beta, delta, gamma, x0, y0) that best fit the data."""
```

## Scoring

R² of predicted trajectory vs. observations, averaged over instances.
Scored as gap-closed from default parameters to true (hidden) parameters.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
