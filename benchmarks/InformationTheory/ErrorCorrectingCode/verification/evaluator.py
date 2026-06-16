"""Oracle: binary linear error-correcting code design.

Agent provides a generator matrix G (k x n) over GF(2). Oracle verifies it has rank k
and computes the minimum Hamming distance d of the code. Higher d = better error correction.
"""
import numpy as np

INSTANCES = [
    {"n": 16, "k": 8, "d_best_known": 5, "note": "Reed-Muller RM(1,4) achieves d=8; [16,8,5] BCH"},
    {"n": 24, "k": 12, "d_best_known": 8, "note": "extended Golay [24,12,8]"},
    {"n": 32, "k": 16, "d_best_known": 8, "note": "Reed-Muller / BCH"},
]

def _gf2_rank(M):
    """Row rank over GF(2) via Gaussian elimination."""
    M = M.copy() % 2
    rows, cols = M.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if M[row, col] == 1:
                pivot = row; break
        if pivot is None:
            continue
        M[[rank, pivot]] = M[[pivot, rank]]
        for row in range(rows):
            if row != rank and M[row, col] == 1:
                M[row] = (M[row] + M[rank]) % 2
        rank += 1
    return rank

def _min_distance_exhaustive(G):
    """Compute minimum weight of nonzero codewords (exact for k <= 20)."""
    k, n = G.shape
    if k > 20:
        return _min_distance_sample(G)
    min_wt = n + 1
    for i in range(1, 2**k):
        # Binary representation of i gives the information word
        info = np.array([(i >> bit) & 1 for bit in range(k)], dtype=int)
        codeword = info @ G % 2
        wt = int(np.sum(codeword))
        if wt < min_wt:
            min_wt = wt
    return min_wt

def _min_distance_sample(G, n_samples=5000):
    """Probabilistic lower bound via random information words."""
    k, n = G.shape
    rng = np.random.default_rng(42)
    min_wt = n + 1
    for _ in range(n_samples):
        info = rng.integers(0, 2, k)
        if np.sum(info) == 0:
            continue
        codeword = info @ G % 2
        wt = int(np.sum(codeword))
        if wt < min_wt:
            min_wt = wt
    return min_wt

def evaluate(design_code):
    results = []
    for inst in INSTANCES:
        n, k = inst["n"], inst["k"]
        try:
            G = np.asarray(design_code(n, k), dtype=int) % 2
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if G.shape != (k, n):
            results.append({"valid": False, "reason": f"shape {G.shape} != ({k},{n})", "score": 0.0}); continue
        if _gf2_rank(G) < k:
            results.append({"valid": False, "reason": "rank deficient", "score": 0.0}); continue
        d = _min_distance_exhaustive(G)
        # Random baseline: d ≈ n * 2^(-n/k) * k... roughly n/4 for rate 1/2
        d_random = max(1, n // 4 - 1)
        d_best = inst["d_best_known"]
        score = max(0.0, min(1.0, (d - d_random) / (d_best - d_random))) if d_best > d_random else (1.0 if d >= d_best else 0.0)
        results.append({"valid": True, "n": n, "k": k, "d": d, "d_best": d_best, "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_instance": results}
