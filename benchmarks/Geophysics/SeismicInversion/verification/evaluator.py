"""Oracle for SeismicInversion — 1D layered model."""
import numpy as np

def forward_model(velocities, layer_thickness, source, receiver):
    """Ray-trace through 1D layers, return travel time."""
    dist = abs(receiver - source)
    depth = sum(layer_thickness)
    n = len(velocities)
    # Simple vertical-ray approximation for 1D: sum(thickness_i / velocity_i)
    # Plus horizontal correction
    horizontal = dist
    vertical_time = sum(layer_thickness[i] / velocities[i] for i in range(n))
    # Approximate: Pythagorean
    avg_v = sum(velocities) / n
    total_path = np.sqrt(horizontal**2 + depth**2)
    return total_path / avg_v  # simplified

def make_scenario(seed):
    rng = np.random.default_rng(seed)
    n_layers = 5
    true_v = rng.uniform(2000, 6000, n_layers)  # m/s
    true_v.sort()  # velocity increases with depth (typical)
    thickness = np.full(n_layers, 500.0)  # 500m per layer
    n_shots = 8; n_recv = 10
    sources = np.linspace(0, 4000, n_shots)
    receivers = np.linspace(500, 4500, n_recv)
    times = []
    for s in sources:
        for r in receivers:
            t = forward_model(true_v, thickness, s, r)
            t += rng.normal(0, 0.005)  # noise
            times.append(t)
    return {"times": np.array(times), "sources": sources, "receivers": receivers,
            "n_layers": n_layers, "true_v": true_v, "thickness": thickness}

SCENARIOS = [make_scenario(42), make_scenario(7)]

def evaluate(invert_seismic):
    results = []
    for sc in SCENARIOS:
        sources, receivers = sc["sources"], sc["receivers"]
        src_pos = [s for s in sources for _ in receivers]
        rcv_pos = [r for _ in sources for r in receivers]
        try:
            v_hat = np.asarray(invert_seismic(sc["times"].copy(), src_pos, rcv_pos, sc["n_layers"]), dtype=float)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if v_hat.shape != (sc["n_layers"],):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        # Predict times with recovered model
        t_pred = []
        for s in sources:
            for r in receivers:
                t_pred.append(forward_model(v_hat, sc["thickness"], s, r))
        t_pred = np.array(t_pred)
        ss_res = np.sum((sc["times"] - t_pred)**2)
        ss_tot = np.sum((sc["times"] - np.mean(sc["times"]))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        # Baseline R2
        v_const = np.full(sc["n_layers"], np.mean(sc["true_v"]))
        t_base = [forward_model(v_const, sc["thickness"], s, r) for s in sources for r in receivers]
        ss_res_b = np.sum((sc["times"] - np.array(t_base))**2)
        r2_base = 1 - ss_res_b / max(ss_tot, 1e-10)
        gap = 1.0 - r2_base
        score = max(0.0, min(1.0, (r2 - r2_base) / gap)) if gap > 1e-6 else 0.0
        results.append({"valid": True, "r2": round(float(r2), 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_scenario": results}
