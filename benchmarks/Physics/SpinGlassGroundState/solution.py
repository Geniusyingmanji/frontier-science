"""Initial baseline for SpinGlassGroundState (weak but valid).

Returns the best of a few random spin configurations — no local search — so it sits far
above the true ground state. Edit this file to do better (greedy descent, simulated
annealing, tabu search, semidefinite relaxation + rounding, ...).
"""

import numpy as np


def solve(n: int, J: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    best_s, best_e = None, np.inf
    for _ in range(3):
        s = rng.choice((-1.0, 1.0), size=n)
        e = -0.5 * float(s @ J @ s)
        if e < best_e:
            best_e, best_s = e, s
    return best_s
