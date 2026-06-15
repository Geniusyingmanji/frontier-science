"""Initial baseline for GraphMaxCut (weak but valid).

Returns the best of 3 random binary partitions — no local search. Edit this file to do
better (greedy, simulated annealing, SDP relaxation + rounding, spectral bisection, ...).
"""

import numpy as np


def solve_maxcut(n: int, W: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    best_p, best_v = None, -1
    for _ in range(3):
        p = rng.integers(0, 2, n)
        v = sum(W[i, j] for i in range(n) for j in range(i + 1, n) if p[i] != p[j])
        if v > best_v:
            best_v, best_p = v, p
    return best_p
