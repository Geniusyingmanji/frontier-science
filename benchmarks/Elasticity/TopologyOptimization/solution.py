"""Baseline: uniform density field (no topology optimization)."""
import numpy as np

def optimize_topology(nelx, nely, volfrac):
    """Return (nely, nelx) density field in [0.001, 1.0] with mean <= volfrac."""
    return np.full((nely, nelx), volfrac)
