"""Frozen oracle for GraphMaxCut (hidden from the agent).

Weighted Max-Cut: given a graph with non-negative edge weights, partition vertices into two
sets to maximize the total weight of edges crossing the partition. This is NP-hard and
equivalent (via duality) to the Ising ground-state problem. Fixed random weighted graphs
with known exact optima (found by brute-force enumeration) are embedded.

Score = (cut_found - cut_baseline) / (cut_optimal - cut_baseline), clipped to [0,1].
"""

from __future__ import annotations

import numpy as np

INSTANCES = [
    {"n": 18, "density": 0.3, "seed": 42, "optimal": 62.7086, "baseline_cut": 49.4045},
    {"n": 20, "density": 0.25, "seed": 7, "optimal": 63.1423, "baseline_cut": 50.5842},
    {"n": 22, "density": 0.2, "seed": 13, "optimal": 68.4764, "baseline_cut": 38.9108},
]


def make_graph(n: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.uniform(0.5, 3.0)
                W[i, j] = W[j, i] = w
    return W


def cut_value(W: np.ndarray, partition) -> float:
    p = np.asarray(partition, dtype=int).ravel()
    n = W.shape[0]
    if p.shape != (n,):
        return 0.0
    val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] != p[j]:
                val += W[i, j]
    return val


def score_instance(entry: dict, solve_maxcut) -> dict:
    n, density, seed = entry["n"], entry["density"], entry["seed"]
    W = make_graph(n, density, seed)
    try:
        partition = solve_maxcut(n, W.copy())
    except Exception as exc:
        return {"n": n, "valid": False, "reason": f"raised: {exc}", "score": 0.0}
    p = np.asarray(partition, dtype=int).ravel()
    if p.shape != (n,) or not np.all(np.isin(p, (0, 1))):
        if p.shape == (n,) and np.all(p != 0):
            p = (np.sign(p) > 0).astype(int)
        else:
            return {"n": n, "valid": False, "reason": "bad partition shape/values", "score": 0.0}
    cv = cut_value(W, p)
    opt, base = entry["optimal"], entry["baseline_cut"]
    progress = (cv - base) / (opt - base) if opt > base else 1.0
    score = float(min(1.0, max(0.0, progress)))
    return {"n": n, "valid": True, "cut_value": round(cv, 4), "optimal": opt, "score": score}


def evaluate(solve_maxcut) -> dict:
    per = [score_instance(e, solve_maxcut) for e in INSTANCES]
    scores = [r["score"] for r in per]
    n_valid = sum(1 for r in per if r.get("valid"))
    return {
        "combined_score": float(np.mean(scores)) if scores else 0.0,
        "valid": 1.0 if n_valid == len(INSTANCES) else 0.0,
        "feasibility_rate": n_valid / len(INSTANCES),
        "per_instance": per,
    }
