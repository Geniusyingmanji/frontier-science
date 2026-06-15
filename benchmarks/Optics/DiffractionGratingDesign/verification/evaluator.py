import numpy as np
def _rcwa_1d_simple(depths, wavelength, period, n_sub, order):
    """Simplified 1D scalar diffraction efficiency calculation."""
    n = len(depths); dx = period / n
    phase = 2*np.pi*n_sub*np.array(depths)/wavelength
    field = np.exp(1j*phase)
    # Fourier transform to get diffraction orders
    orders = np.fft.fftshift(np.fft.fft(field))/n
    n_orders = len(orders); center = n_orders//2
    idx = center + order
    if 0 <= idx < n_orders:
        return float(np.abs(orders[idx])**2)
    return 0.0

SCENARIOS = [
    {"wl": 0.5e-6, "period": 1e-6, "n_sub": 1.5, "order": 1, "n_grooves": 64},
    {"wl": 1.0e-6, "period": 2e-6, "n_sub": 1.45, "order": 1, "n_grooves": 64},
    {"wl": 0.633e-6, "period": 1.5e-6, "n_sub": 1.52, "order": 1, "n_grooves": 64},
]

def evaluate(design_grating):
    results = []
    for sc in SCENARIOS:
        try:
            d = np.asarray(design_grating(sc["wl"], sc["period"], sc["n_sub"], sc["order"], sc["n_grooves"]), dtype=float)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if d.shape != (sc["n_grooves"],):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        eff = _rcwa_1d_simple(np.clip(d, 0, sc["wl"]*2), sc["wl"], sc["period"], sc["n_sub"], sc["order"])
        base_eff = _rcwa_1d_simple(np.full(sc["n_grooves"], sc["wl"]/(2*sc["n_sub"])), sc["wl"], sc["period"], sc["n_sub"], sc["order"])
        max_eff = 0.8  # theoretical max for simple binary grating
        score = max(0.0, min(1.0, (eff - base_eff) / (max_eff - base_eff))) if max_eff > base_eff else 0.0
        results.append({"valid": True, "efficiency": round(eff, 6), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)), "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid"))/len(results), "per_scenario": results}
