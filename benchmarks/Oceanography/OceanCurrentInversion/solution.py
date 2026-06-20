"""Baseline: zero current field."""
import numpy as np
def invert_currents(obs_traj, init_pos, nx, ny, domain):
    return {"u": np.zeros((nx, ny)), "v": np.zeros((nx, ny))}
