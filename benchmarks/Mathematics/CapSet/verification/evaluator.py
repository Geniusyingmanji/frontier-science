"""Frozen oracle for CapSet (hidden from the agent).

A cap set in Z_3^n is a subset with no three distinct collinear points, i.e. no distinct
x, y, z with x + y + z == 0 (mod 3). The oracle verifies the cap property and returns |S|.
Larger is better; the score is uncapped relative to the best known size (reaching it = 1.0,
beating it > 1.0). The {0,1}^n hypercube is always a valid cap of size 2^n and is the floor.
"""

from __future__ import annotations

import numpy as np

# dim n -> (baseline {0,1}^n size = 2^n, best-known max cap size).
# Maxima for n<=6 are proven; treat as the score=1.0 anchor.
SIZES = {
    4: {"baseline": 16, "sota_ref": 20},
    5: {"baseline": 32, "sota_ref": 45},
    6: {"baseline": 64, "sota_ref": 112},
}


def verify_cap(vecs, n: int) -> tuple[bool, int, str]:
    S = set()
    for v in vecs:
        try:
            t = tuple(int(x) % 3 for x in v)
        except Exception as exc:  # noqa: BLE001
            return False, 0, f"bad vector: {exc}"
        if len(t) != n:
            return False, 0, f"vector length != {n}"
        S.add(t)
    arr = list(S)
    Sset = set(arr)
    for a in range(len(arr)):
        xa = arr[a]
        for b in range(a + 1, len(arr)):
            yb = arr[b]
            z = tuple((-(xa[i] + yb[i])) % 3 for i in range(n))
            if z in Sset and z != xa and z != yb:
                return False, len(S), f"collinear triple found"
    return True, len(S), "ok"


def score_dim(n: int, ref: dict, build_capset) -> dict:
    try:
        vecs = build_capset(n)
    except Exception as exc:  # noqa: BLE001
        return {"n": n, "valid": False, "reason": f"raised: {exc}", "score": 0.0}
    ok, size, reason = verify_cap(vecs, n)
    if not ok:
        return {"n": n, "valid": False, "reason": reason, "size": size, "score": 0.0}
    base, sota = ref["baseline"], ref["sota_ref"]
    progress = (size - base) / (sota - base)          # 0 at 2^n, 1 at best known, >1 beyond
    return {"n": n, "valid": True, "size": size, "sota_ref": sota,
            "score": float(max(0.0, progress))}        # uncapped above


def evaluate(build_capset) -> dict:
    per = [score_dim(n, ref, build_capset) for n, ref in SIZES.items()]
    scores = [r["score"] for r in per]
    n_valid = sum(1 for r in per if r.get("valid"))
    return {
        "combined_score": float(np.mean(scores)) if scores else 0.0,
        "valid": 1.0 if n_valid == len(SIZES) else 0.0,
        "feasibility_rate": n_valid / len(SIZES),
        "beat_sota": bool(any(r.get("score", 0.0) > 1.0 for r in per)),
        "per_dim": per,
    }
