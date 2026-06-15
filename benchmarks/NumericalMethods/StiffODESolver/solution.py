"""Baseline: implicit Euler (backward Euler) with crude Newton — valid but inaccurate."""
import numpy as np

def solve_ode(f, y0, t_span, t_eval, rtol, atol):
    y = np.array(y0, dtype=float)
    n_steps = 500
    dt = (t_span[1] - t_span[0]) / n_steps
    t = t_span[0]
    trajectory = [(t, y.copy())]
    for _ in range(n_steps):
        # Crude backward Euler: y_{n+1} = y_n + dt * f(t+dt, y_{n+1})
        # Use forward Euler as initial guess, then 3 fixed-point iterations
        y_guess = y + dt * np.array(f(t, y))
        for _ in range(3):
            y_guess = y + dt * np.array(f(t + dt, np.clip(y_guess, 0, 1e20)))
        y = np.clip(y_guess, 0, 1e20)
        t += dt
        trajectory.append((t, y.copy()))
    ts = np.array([p[0] for p in trajectory])
    ys = np.array([p[1] for p in trajectory])
    results = []
    for te in t_eval:
        idx = min(max(0, np.searchsorted(ts, te, side='right') - 1), len(ts) - 1)
        results.append(ys[idx])
    return np.array(results)
