"""Baseline: constant thrust toward target (inefficient but valid)."""
import numpy as np

def design_trajectory(r0, v0, rf, vf, T_max, t_final, n_steps):
    thrust = np.zeros((n_steps, 3))
    for i in range(n_steps):
        t = i / n_steps
        r_interp = (1 - t) * np.array(r0) + t * np.array(rf)
        direction = np.array(rf) - r_interp
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            thrust[i] = direction / norm * T_max * 0.3
    return thrust
