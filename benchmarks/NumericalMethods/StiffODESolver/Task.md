# StiffODESolver — design a stiff ODE integrator

## Scientific background

Stiff ODEs arise in chemical kinetics, circuit simulation, and combustion. Standard
explicit methods (Euler, RK4) require impractically small timesteps. The Robertson
chemical kinetics problem (3 species, stiffness ratio ~10^8) is the standard test.

## Your task

```python
def solve_ode(f, y0, t_span, t_eval, rtol, atol):
    """Integrate dy/dt = f(t, y) from t_span[0] to t_span[1].
    Return y at t_eval points. Shape (len(t_eval), len(y0))."""
```

## Scoring

Accuracy (relative error vs scipy.integrate.solve_ivp with Radau) per number of RHS
evaluations. Better accuracy with fewer evals scores higher.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
