# SIRParameterEstimation — fit epidemic model to outbreak data

## Scientific background

The SIR (Susceptible-Infectious-Recovered) model is the foundation of mathematical
epidemiology. Given noisy daily case counts from an outbreak, estimating β (transmission
rate) and γ (recovery rate) is critical for forecasting and intervention planning.

## Your task

```python
def estimate_sir(t_obs, I_obs, N_pop):
    """Given observation times, infectious counts, and total population,
    return (beta, gamma, S0, I0, R0) that best fit the data."""
```

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
