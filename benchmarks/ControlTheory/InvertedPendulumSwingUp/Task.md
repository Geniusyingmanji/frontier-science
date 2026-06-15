# InvertedPendulumSwingUp — design a controller to swing up and balance

## Scientific background

The cart-pole swing-up is a classic nonlinear control challenge: move the cart to swing the
pendulum from hanging (stable) to upright (unstable), then balance it. It requires energy
pumping, switching control, and stabilization — a testbed for optimal and nonlinear control.

## Your task

```python
def swing_up_controller(state, t, dt):
    """Given state = (x, x_dot, theta, theta_dot) and time, return force F on cart.
    theta=0 is hanging down, theta=pi is upright. |F| <= F_max=10."""
```

## Scoring

Score = time_balanced_upright / total_time * efficiency_penalty. Higher is better.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
