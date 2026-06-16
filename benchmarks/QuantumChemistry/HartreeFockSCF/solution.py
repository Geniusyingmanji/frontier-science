"""Baseline: equal-mix orbital (suboptimal)."""
import numpy as np
def optimize_orbitals(H_core, S, ERI, V_nn):
    """Return orbital coefficient vector (2,) for 1 occupied MO of H2/STO-3G."""
    return np.array([1.0, 1.0])  # unnormalized, will be normalized by oracle
