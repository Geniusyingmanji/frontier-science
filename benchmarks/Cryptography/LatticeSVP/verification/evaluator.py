"""Oracle for LatticeSVP."""
import numpy as np

def _gram_schmidt(B):
    n = B.shape[0]; Q = np.zeros_like(B, dtype=float); mu = np.zeros((n,n))
    for i in range(n):
        Q[i] = B[i].copy()
        for j in range(i):
            mu[i,j] = np.dot(B[i], Q[j]) / max(np.dot(Q[j], Q[j]), 1e-30)
            Q[i] -= mu[i,j] * Q[j]
    return Q, mu

def _lll(B, delta=0.99):
    B = B.copy().astype(float); n = B.shape[0]
    Q, mu = _gram_schmidt(B); k = 1
    while k < n:
        for j in range(k-1, -1, -1):
            if abs(mu[k,j]) > 0.5:
                B[k] -= round(mu[k,j]) * B[j]
                Q, mu = _gram_schmidt(B)
        if np.dot(Q[k], Q[k]) >= (delta - mu[k,k-1]**2) * np.dot(Q[k-1], Q[k-1]):
            k += 1
        else:
            B[[k-1,k]] = B[[k,k-1]]
            Q, mu = _gram_schmidt(B); k = max(k-1, 1)
    return B

def make_instance(n, seed):
    rng = np.random.default_rng(seed)
    B = rng.integers(-20, 21, (n, n))
    while abs(np.linalg.det(B.astype(float))) < 1:
        B = rng.integers(-20, 21, (n, n))
    return B

INSTANCES = [(8, 42), (10, 7), (12, 13)]

def evaluate(find_short_vector):
    results = []
    for n, seed in INSTANCES:
        B = make_instance(n, seed)
        reduced = _lll(B.astype(float))
        ref_norm = float(np.min(np.linalg.norm(reduced, axis=1)))
        try:
            v = np.asarray(find_short_vector(B.copy()), dtype=float).ravel()
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if v.shape != (n,) or np.allclose(v, 0):
            results.append({"valid": False, "reason": "zero or bad shape", "score": 0.0}); continue
        # Verify it is a lattice vector: v = c @ B for some integer c
        try:
            c = np.linalg.solve(B.astype(float).T, v)
            if not np.allclose(c, np.round(c), atol=1e-6):
                results.append({"valid": False, "reason": "not a lattice vector", "score": 0.0}); continue
        except:
            results.append({"valid": False, "reason": "singular", "score": 0.0}); continue
        norm = float(np.linalg.norm(v))
        baseline_norm = float(np.min(np.linalg.norm(B.astype(float), axis=1)))
        gap = baseline_norm - ref_norm
        score = max(0.0, min(1.0, (baseline_norm - norm) / gap)) if gap > 1e-6 else (1.0 if norm <= ref_norm else 0.0)
        results.append({"valid": True, "norm": round(norm, 4), "ref_norm": round(ref_norm, 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_instance": results}
