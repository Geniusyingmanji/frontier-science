"""Oracle: linear antenna array pattern synthesis — minimize peak sidelobe level.

Agent designs complex excitation weights for an N-element uniform linear array. Oracle
computes the array factor AF(θ) and measures the peak sidelobe level (PSLL). Lower PSLL
= better directivity = higher score. Dolph-Chebyshev weights are the known optimum.
"""
import numpy as np

INSTANCES = [
    {"n_elements": 16, "d_lambda": 0.5, "mainlobe_width_deg": 10.0,
     "psll_baseline": -13.3, "psll_chebyshev": -30.0},  # dB
    {"n_elements": 32, "d_lambda": 0.5, "mainlobe_width_deg": 5.0,
     "psll_baseline": -13.3, "psll_chebyshev": -40.0},
]

def _array_factor(weights, d_lambda, theta_deg):
    """Compute array factor magnitude in dB at angles theta."""
    n = len(weights)
    theta = np.deg2rad(theta_deg)
    k = 2 * np.pi
    positions = np.arange(n) * d_lambda  # in wavelengths
    # AF(θ) = Σ w_n exp(j k d n sin θ)
    phase = k * np.outer(positions, np.sin(theta))  # (n, n_angles)
    af = weights @ np.exp(1j * phase)  # (n_angles,)
    af_mag = np.abs(af)
    af_mag /= np.max(af_mag)  # normalize peak to 1
    af_db = 20 * np.log10(np.maximum(af_mag, 1e-15))
    return af_db

def _peak_sidelobe(af_db, mainlobe_half_deg, theta_deg):
    """Find peak sidelobe level outside the mainlobe region."""
    mask = np.abs(theta_deg) > mainlobe_half_deg
    if not np.any(mask):
        return -100.0
    return float(np.max(af_db[mask]))

def evaluate(design_array):
    results = []
    for inst in INSTANCES:
        n = inst["n_elements"]
        try:
            weights = np.asarray(design_array(n, inst["d_lambda"], inst["mainlobe_width_deg"]), dtype=complex)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if weights.shape != (n,):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        theta = np.linspace(-90, 90, 1801)
        af_db = _array_factor(weights, inst["d_lambda"], theta)
        psll = _peak_sidelobe(af_db, inst["mainlobe_width_deg"] / 2, theta)
        # Score: how much PSLL was reduced vs uniform (baseline -13.3 dB)
        psll_base = inst["psll_baseline"]
        psll_best = inst["psll_chebyshev"]
        # Lower PSLL (more negative) is better
        score = max(0.0, min(1.0, (psll_base - psll) / (psll_base - psll_best))) if psll_base > psll_best else 0.0
        results.append({"valid": True, "psll_dB": round(psll, 2), "n": n, "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results), "per_instance": results}
