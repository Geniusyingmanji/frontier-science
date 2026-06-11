"""Frozen oracle for SpinGlassGroundState (hidden from the agent).

Sherrington-Kirkpatrick Ising model: N spins s_i in {-1,+1}, symmetric Gaussian couplings
J_ij ~ N(0,1)/sqrt(N). Energy E(s) = -1/2 sN J s = -sum_{i<j} J_ij s_i s_j. Finding the
ground state is NP-hard in general; for these small, fixed (seeded) instances the exact
minimum was found by full enumeration and embedded below as the normalization ceiling.
"""

from __future__ import annotations

import numpy as np

# Instances: (N, seed). Couplings are regenerated deterministically by make_instance.
INSTANCES = [(16, 0), (18, 0), (20, 0)]

# Exact ground-state energy and all-up reference energy (brute-forced at task build time).
# E_ref (all spins +1) is the fixed normalization baseline; E_min is the true optimum.
REFERENCE = {
    (16, 0): {"e_min": -6.985473, "e_ref": 0.059847},
    (18, 0): {"e_min": -8.609497, "e_ref": 2.327832},
    (20, 0): {"e_min": -9.012512, "e_ref": 1.441443},
}


def make_instance(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n, n))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    J /= np.sqrt(n)
    return J


def energy(J: np.ndarray, s: np.ndarray) -> float:
    return -0.5 * float(s @ J @ s)


def score_instance(n: int, seed: int, solve) -> dict:
    J = make_instance(n, seed)
    try:
        s = np.asarray(solve(n, J.copy()), dtype=float).ravel()
    except Exception as exc:  # noqa: BLE001
        return {"n": n, "seed": seed, "valid": False, "reason": f"raised: {exc}", "score": 0.0}
    if s.shape != (n,) or not np.all(np.isfinite(s)):
        return {"n": n, "seed": seed, "valid": False, "reason": "bad shape/non-finite", "score": 0.0}
    if not np.all(np.isin(s, (-1.0, 1.0))):
        # tolerate any nonzero by taking sign, but require binary-like output
        if np.any(s == 0):
            return {"n": n, "seed": seed, "valid": False, "reason": "zero spin", "score": 0.0}
        s = np.sign(s)
    e = energy(J, s)
    ref = REFERENCE[(n, seed)]
    e_min, e_ref = ref["e_min"], ref["e_ref"]
    score = (e_ref - e) / (e_ref - e_min)
    score = float(min(1.0, max(0.0, score)))
    return {"n": n, "seed": seed, "valid": True, "energy": e, "e_min": e_min, "score": score}


def evaluate(solve) -> dict:
    per = [score_instance(n, seed, solve) for (n, seed) in INSTANCES]
    scores = [r["score"] for r in per]
    n_valid = sum(1 for r in per if r.get("valid"))
    return {
        "combined_score": float(np.mean(scores)) if scores else 0.0,
        "valid": 1.0 if n_valid == len(INSTANCES) else 0.0,
        "feasibility_rate": n_valid / len(INSTANCES),
        "per_instance": per,
    }
