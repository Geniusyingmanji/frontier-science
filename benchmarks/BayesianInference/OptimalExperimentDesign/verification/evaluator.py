"""Oracle: D-optimal experimental design — maximize information gain.

Agent selects K design points from a continuous space to maximize the log-determinant of
the Fisher Information Matrix (FIM), equivalent to minimizing the volume of the parameter
confidence ellipsoid. This is the classic D-optimal criterion.
"""
import numpy as np

def _regressor_matrix(designs, degree=3):
    """Polynomial regression model: y = sum_j theta_j * phi_j(d)."""
    d = np.asarray(designs).ravel()
    # Basis functions: 1, d, d^2, d^3, sin(2*pi*d), cos(2*pi*d)
    H = np.column_stack([np.ones_like(d), d, d**2, d**3,
                         np.sin(2*np.pi*d), np.cos(2*np.pi*d)])
    return H

N_PARAMS = 6  # number of model parameters
K_DESIGNS = 12  # number of design points to select
SIGMA = 0.1  # noise std

INSTANCES = [
    {"d_range": [0, 1], "seed": 42},
    {"d_range": [-1, 2], "seed": 7},
]

def _compute_eig(designs):
    """Expected Information Gain = 0.5 * log|FIM| (for linear-Gaussian model)."""
    H = _regressor_matrix(designs)
    FIM = H.T @ H / SIGMA**2
    sign, logdet = np.linalg.slogdet(FIM)
    if sign <= 0:
        return -1e10
    return 0.5 * float(logdet)

def evaluate(select_designs):
    results = []
    for inst in INSTANCES:
        d_lo, d_hi = inst["d_range"]
        try:
            designs = np.asarray(select_designs(K_DESIGNS, d_lo, d_hi, N_PARAMS), dtype=float).ravel()
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if designs.shape != (K_DESIGNS,):
            results.append({"valid": False, "reason": "need K designs", "score": 0.0}); continue
        designs = np.clip(designs, d_lo, d_hi)
        eig = _compute_eig(designs)
        # Baseline: uniform spacing
        d_uniform = np.linspace(d_lo, d_hi, K_DESIGNS)
        eig_base = _compute_eig(d_uniform)
        # SoTA: D-optimal for polynomial (known to place designs at Chebyshev-like nodes)
        # Approximate by Fedorov exchange on a fine grid
        rng = np.random.default_rng(inst["seed"])
        best_eig = eig_base
        for _ in range(200):
            trial = d_lo + (d_hi - d_lo) * rng.random(K_DESIGNS)
            e = _compute_eig(trial)
            if e > best_eig:
                best_eig = e
        eig_sota = best_eig * 1.02  # slightly above our search as target
        score = max(0.0, min(1.0, (eig - eig_base) / (eig_sota - eig_base))) if eig_sota > eig_base else 0.0
        results.append({"valid": True, "eig": round(eig, 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results), "per_instance": results}
