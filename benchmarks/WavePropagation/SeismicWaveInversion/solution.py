"""Baseline: homogeneous velocity model."""
import numpy as np
def design_velocity(nx, nz):
    return np.full((nz, nx), 2000.0)
