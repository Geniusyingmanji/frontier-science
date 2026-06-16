"""Baseline: uniform mixed strategy (high exploitability)."""
import numpy as np

def find_nash(A, B):
    """Return (p, q) mixed strategy Nash equilibrium of bimatrix game (A, B)."""
    n = A.shape[0]
    return np.ones(n) / n, np.ones(n) / n
