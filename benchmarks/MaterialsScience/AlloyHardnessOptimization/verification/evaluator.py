"""Oracle for AlloyHardnessOptimization — surrogate model."""
import numpy as np

ELEMENTS = ["Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Al", "Ti"]
N_ELEM = 5  # use first 5

def _surrogate_hardness(comps):
    """Deterministic surrogate: polynomial + interaction terms on composition."""
    comps = np.asarray(comps, dtype=float)
    if comps.ndim == 1: comps = comps.reshape(1, -1)
    # Coefficients (fixed, pseudo-physical)
    linear = np.array([300, 250, 200, 350, 150])[:comps.shape[1]]
    quad = np.array([50, -30, 20, 80, -10])[:comps.shape[1]]
    h = comps @ linear + (comps**2) @ quad
    # Interaction bonus for balanced compositions (entropy effect)
    entropy = -np.sum(comps * np.log(np.maximum(comps, 1e-10)), axis=1)
    h += entropy * 30
    # Penalty for extreme compositions
    max_frac = np.max(comps, axis=1)
    h -= (max_frac > 0.5) * 100
    return h.ravel()

def evaluate(design_alloy):
    def predict(c): return _surrogate_hardness(c)
    try:
        comps = np.asarray(design_alloy(N_ELEM, ELEMENTS[:N_ELEM], predict, 50), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "reason": str(e), "feasibility_rate": 0.0}
    if comps.ndim != 2 or comps.shape[1] != N_ELEM:
        return {"combined_score": 0.0, "valid": 0.0, "reason": "bad shape", "feasibility_rate": 0.0}
    # Check compositions sum to 1 and non-negative
    valid_mask = (np.abs(comps.sum(axis=1) - 1.0) < 0.01) & np.all(comps >= -0.01, axis=1)
    if not np.any(valid_mask):
        return {"combined_score": 0.0, "valid": 0.0, "reason": "no valid compositions", "feasibility_rate": 0.0}
    comps_valid = comps[valid_mask]
    h = _surrogate_hardness(comps_valid)
    best = float(np.max(h))
    # Baseline: random equimolar
    rng = np.random.default_rng(0)
    h_base = float(np.max(_surrogate_hardness(rng.dirichlet(np.ones(N_ELEM), 50))))
    h_sota = 420.0  # theoretical max of surrogate
    score = max(0.0, min(1.0, (best - h_base) / (h_sota - h_base))) if h_sota > h_base else 0.0
    return {"combined_score": float(score), "valid": 1.0,
            "best_hardness": round(best, 2), "feasibility_rate": float(np.mean(valid_mask))}
