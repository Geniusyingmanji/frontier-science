"""Oracle for LowThrustTransfer."""
import numpy as np

MU = 3.986004418e14  # Earth GM, m^3/s^2

SCENARIOS = [
    {"r0": [7000e3, 0, 0], "v0": [0, 7546.0, 0],
     "rf": [42164e3, 0, 0], "vf": [0, 3074.7, 0],
     "T_max": 0.01, "t_final": 86400*30, "n_steps": 1000,
     "fuel_baseline": 280.0, "fuel_optimal": 45.0,
     "note": "LEO to GEO, 30-day low-thrust"},
]

def propagate(r0, v0, thrust, dt):
    r = np.array(r0, dtype=float); v = np.array(v0, dtype=float)
    for a in thrust:
        a = np.array(a, dtype=float)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e3: return r, v, False
        acc = -MU * r / r_norm**3 + a
        v = v + acc * dt
        r = r + v * dt
    return r, v, True

def evaluate(design_trajectory):
    results = []
    for sc in SCENARIOS:
        n = sc["n_steps"]; dt = sc["t_final"] / n
        try:
            thrust = np.asarray(design_trajectory(
                sc["r0"], sc["v0"], sc["rf"], sc["vf"],
                sc["T_max"], sc["t_final"], n), dtype=float)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if thrust.shape != (n, 3):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        # Clamp thrust magnitude
        mags = np.linalg.norm(thrust, axis=1)
        over = mags > sc["T_max"]
        if np.any(over):
            thrust[over] *= sc["T_max"] / mags[over, None]
        fuel = float(np.sum(np.linalg.norm(thrust, axis=1)) * dt)
        rf, vf, ok = propagate(sc["r0"], sc["v0"], thrust, dt)
        if not ok:
            results.append({"valid": False, "reason": "crash", "score": 0.0}); continue
        pos_err = np.linalg.norm(rf - np.array(sc["rf"])) / np.linalg.norm(sc["rf"])
        vel_err = np.linalg.norm(vf - np.array(sc["vf"])) / np.linalg.norm(sc["vf"])
        arrival_penalty = min(1.0, pos_err * 10 + vel_err * 10)
        fb, fo = sc["fuel_baseline"], sc["fuel_optimal"]
        raw = (fb - fuel) / (fb - fo) if fb > fo else 0.0
        score = max(0.0, min(1.0, raw)) * (1.0 - arrival_penalty)
        results.append({"valid": True, "fuel": round(fuel, 2),
                        "pos_err": round(pos_err, 6), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)), "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results), "per_scenario": results}
