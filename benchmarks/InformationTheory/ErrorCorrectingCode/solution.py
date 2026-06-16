"""Baseline: random binary generator matrix (poor minimum distance)."""
import numpy as np
def design_code(n, k):
    rng = np.random.default_rng(0)
    G = rng.integers(0, 2, (k, n))
    # Make sure it's full rank by setting identity in first k columns
    G[:, :k] = np.eye(k, dtype=int)
    return G
