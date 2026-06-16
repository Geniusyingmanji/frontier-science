"""Baseline: zero density model (no inversion attempt)."""
import numpy as np
def invert_gravity(G_kernel, g_obs, nx, nz):
    return np.zeros(nx * nz)
