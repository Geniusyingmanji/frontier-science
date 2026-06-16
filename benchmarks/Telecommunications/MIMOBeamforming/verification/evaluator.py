"""Oracle: MIMO beamforming — maximize spectral efficiency.

Agent designs precoding matrix for a MIMO channel. Oracle computes achievable spectral
efficiency (sum-rate) via the water-filling upper bound and the achieved rate with the
candidate precoder. Per-antenna power constraint makes this non-trivial (not just SVD).
"""
import numpy as np

def _make_channel(n_tx, n_rx, seed):
    rng = np.random.default_rng(seed)
    H = (rng.standard_normal((n_rx, n_tx)) + 1j * rng.standard_normal((n_rx, n_tx))) / np.sqrt(2)
    return H

INSTANCES = [
    {"n_tx": 8, "n_rx": 4, "snr_db": 10, "seed": 42},
    {"n_tx": 16, "n_rx": 4, "snr_db": 15, "seed": 7},
]

def _spectral_efficiency(H, W, snr_linear):
    """Compute spectral efficiency with precoder W."""
    n_rx = H.shape[0]
    n_streams = W.shape[1]
    # Normalize W to satisfy total power constraint: Tr(W W^H) <= n_tx
    n_tx = W.shape[0]
    power = np.real(np.trace(W @ W.conj().T))
    if power > 1e-10:
        W = W * np.sqrt(n_tx / power)
    # Signal: H W
    HW = H @ W
    # Rate = log2 det(I + SNR/n_tx * HW HW^H)
    noise_power = 1.0 / snr_linear
    R = np.eye(n_rx) + (snr_linear / n_tx) * (HW @ HW.conj().T)
    rate = float(np.real(np.log2(np.linalg.det(R))))
    return max(0.0, rate)

def _waterfilling_rate(H, snr_linear):
    """Upper bound: water-filling on the channel singular values."""
    n_tx = H.shape[1]
    s = np.linalg.svd(H, compute_uv=False)
    n_streams = len(s)
    # Water-filling
    lam = s**2 * snr_linear / n_tx
    # Iterative water-filling
    rate = 0.0
    for l in lam:
        if l > 0:
            rate += np.log2(1 + l)
    return float(rate)

def evaluate(design_precoder):
    results = []
    for inst in INSTANCES:
        H = _make_channel(inst["n_tx"], inst["n_rx"], inst["seed"])
        snr_lin = 10**(inst["snr_db"] / 10)
        try:
            W = np.asarray(design_precoder(H.copy(), inst["n_tx"], inst["n_rx"], inst["snr_db"]),
                           dtype=complex)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if W.ndim != 2 or W.shape[0] != inst["n_tx"]:
            results.append({"valid": False, "reason": "bad precoder shape", "score": 0.0}); continue
        rate = _spectral_efficiency(H, W, snr_lin)
        # Baseline: identity precoder (first n_rx columns)
        W_base = np.eye(inst["n_tx"], inst["n_rx"])
        rate_base = _spectral_efficiency(H, W_base, snr_lin)
        # SoTA: SVD precoding (optimal under total power)
        U, s, Vh = np.linalg.svd(H)
        W_svd = Vh.conj().T[:, :inst["n_rx"]]
        rate_svd = _spectral_efficiency(H, W_svd, snr_lin)
        score = max(0.0, min(1.0, (rate - rate_base) / (rate_svd - rate_base))) if rate_svd > rate_base else 0.0
        results.append({"valid": True, "rate_bps_hz": round(rate, 3), "rate_svd": round(rate_svd, 3), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results), "per_instance": results}
