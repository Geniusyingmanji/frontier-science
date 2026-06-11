"""Initial baseline for LennardJonesCluster (weak but valid).

Places atoms at random inside a loose sphere with a minimum-separation rejection. The
atoms barely interact, so the energy is close to zero (far from the global minimum) — there
is a large gap for an optimizer to close. Edit this file to do better.
"""

import numpy as np


def build_cluster(n_atoms: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    radius = 1.5 * (n_atoms ** (1.0 / 3.0))
    pts = []
    while len(pts) < n_atoms:
        p = rng.uniform(-radius, radius, size=3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) > 1.2 for q in pts):
            pts.append(p)
    return np.asarray(pts, dtype=float)
