"""Oracle: permutation flow shop scheduling — minimize makespan.

N jobs on M machines in fixed order. Agent returns a permutation of jobs. Oracle computes
makespan (completion time of last job on last machine). This is NP-hard for M>=3.
Uses Taillard-style benchmark instances.
"""
import numpy as np

def _make_instance(n_jobs, n_machines, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(1, 100, (n_jobs, n_machines))  # processing times

def _compute_makespan(processing_times, permutation):
    n, m = processing_times.shape
    C = np.zeros((n, m))
    perm = list(permutation)
    # First job
    C[0, 0] = processing_times[perm[0], 0]
    for j in range(1, m):
        C[0, j] = C[0, j-1] + processing_times[perm[0], j]
    # Remaining jobs
    for i in range(1, n):
        C[i, 0] = C[i-1, 0] + processing_times[perm[i], 0]
        for j in range(1, m):
            C[i, j] = max(C[i-1, j], C[i, j-1]) + processing_times[perm[i], j]
    return int(C[-1, -1])

INSTANCES = [
    {"n": 20, "m": 5, "seed": 42, "makespan_baseline": 1518, "makespan_neh": 1268},
    {"n": 20, "m": 10, "seed": 7, "makespan_baseline": 1967, "makespan_neh": 1607},
    {"n": 50, "m": 5, "seed": 13, "makespan_baseline": 3424, "makespan_neh": 2969},
]

def evaluate(schedule_flowshop):
    results = []
    for inst in INSTANCES:
        P = _make_instance(inst["n"], inst["m"], inst["seed"])
        n = inst["n"]
        try:
            perm = list(schedule_flowshop(P.copy(), n, inst["m"]))
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if sorted(perm) != list(range(n)):
            results.append({"valid": False, "reason": "not a valid permutation", "score": 0.0}); continue
        ms = _compute_makespan(P, perm)
        ms_base = inst["makespan_baseline"]
        ms_best = inst["makespan_neh"]
        score = max(0.0, min(1.0, (ms_base - ms) / (ms_base - ms_best))) if ms_base > ms_best else (1.0 if ms <= ms_best else 0.0)
        results.append({"valid": True, "makespan": ms, "baseline": ms_base, "neh_ref": ms_best, "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results), "per_instance": results}
