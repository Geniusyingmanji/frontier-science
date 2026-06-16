"""Frozen oracle for NashEquilibrium — exploitability computation.

Given a bimatrix game (A, B), evaluates a candidate mixed-strategy profile (p, q) by computing
its exploitability: the sum of each player's maximum gain from unilateral deviation.
Lower exploitability = closer to Nash equilibrium = higher score.
"""
import numpy as np

def _make_game(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    B = rng.standard_normal((n, n))
    return A, B

INSTANCES = [
    {"n": 10, "seed": 42, "note": "small, calibration"},
    {"n": 30, "seed": 7, "note": "medium, heuristic required"},
    {"n": 50, "seed": 13, "note": "large, approximate only"},
]

def _exploitability(A, B, p, q):
    """Total exploitability of (p, q) in game (A, B)."""
    # Player 1's incentive to deviate
    payoff_1 = p @ A @ q
    best_response_1 = np.max(A @ q)
    eps_1 = best_response_1 - payoff_1
    # Player 2's incentive to deviate
    payoff_2 = p @ B @ q
    best_response_2 = np.max(B.T @ p)
    eps_2 = best_response_2 - payoff_2
    return max(0.0, eps_1) + max(0.0, eps_2)

def evaluate(find_nash):
    results = []
    for inst in INSTANCES:
        A, B = _make_game(inst["n"], inst["seed"])
        n = inst["n"]
        try:
            p, q = find_nash(A.copy(), B.copy())
            p = np.asarray(p, dtype=float).ravel()
            q = np.asarray(q, dtype=float).ravel()
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if p.shape != (n,) or q.shape != (n,):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        # Project to valid probability simplex
        p = np.maximum(p, 0); q = np.maximum(q, 0)
        sp, sq = p.sum(), q.sum()
        if sp < 1e-10 or sq < 1e-10:
            results.append({"valid": False, "reason": "zero strategy", "score": 0.0}); continue
        p /= sp; q /= sq
        eps = _exploitability(A, B, p, q)
        # Baseline: uniform strategy exploitability
        p_unif, q_unif = np.ones(n)/n, np.ones(n)/n
        eps_base = _exploitability(A, B, p_unif, q_unif)
        score = max(0.0, min(1.0, (eps_base - eps) / eps_base)) if eps_base > 1e-10 else (1.0 if eps < 1e-10 else 0.0)
        results.append({"valid": True, "exploitability": round(eps, 6), "n": n, "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid"))/len(results),
            "per_instance": results}
