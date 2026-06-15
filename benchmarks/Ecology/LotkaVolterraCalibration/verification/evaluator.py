"""Oracle for LotkaVolterraCalibration."""
import numpy as np
from scipy.integrate import solve_ivp

def simulate(params, t):
    a, b, d, g, x0, y0 = params
    def rhs(t, z): return [a*z[0] - b*z[0]*z[1], d*z[0]*z[1] - g*z[1]]
    sol = solve_ivp(rhs, [t[0], t[-1]], [x0, y0], t_eval=t, rtol=1e-8, atol=1e-8)
    if sol.success and sol.y.shape[1] == len(t):
        return sol.y[0], sol.y[1]
    return np.full(len(t), np.nan), np.full(len(t), np.nan)

INSTANCES = []
def _make():
    if INSTANCES: return
    for seed, true_params in enumerate([
        (1.5, 0.8, 0.6, 1.2, 10.0, 5.0),
        (0.8, 0.4, 0.3, 0.7, 20.0, 8.0),
    ]):
        rng = np.random.default_rng(seed + 200)
        t = np.linspace(0, 15, 80)
        x_true, y_true = simulate(true_params, t)
        x_obs = x_true + rng.normal(0, 0.5, len(t))
        y_obs = y_true + rng.normal(0, 0.3, len(t))
        x_obs = np.maximum(x_obs, 0.1); y_obs = np.maximum(y_obs, 0.1)
        INSTANCES.append({"t": t, "x_obs": x_obs, "y_obs": y_obs,
                          "true_params": true_params, "x_true": x_true, "y_true": y_true})

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / max(ss_tot, 1e-10)

def evaluate(calibrate):
    _make()
    results = []
    for inst in INSTANCES:
        try:
            params = calibrate(inst["t"].copy(), inst["x_obs"].copy(), inst["y_obs"].copy())
            params = tuple(float(p) for p in params)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        x_pred, y_pred = simulate(params, inst["t"])
        if np.any(np.isnan(x_pred)):
            results.append({"valid": False, "reason": "sim failed", "score": 0.0}); continue
        r2x = r2(inst["x_obs"], x_pred); r2y = r2(inst["y_obs"], y_pred)
        fit = (r2x + r2y) / 2
        # Baseline fit
        default = (1.0, 0.5, 0.5, 1.0, inst["x_obs"][0], inst["y_obs"][0])
        xb, yb = simulate(default, inst["t"])
        fit_base = (r2(inst["x_obs"], xb) + r2(inst["y_obs"], yb)) / 2 if not np.any(np.isnan(xb)) else -1.0
        # True fit
        xt, yt = simulate(inst["true_params"], inst["t"])
        fit_true = (r2(inst["x_obs"], xt) + r2(inst["y_obs"], yt)) / 2
        gap = fit_true - fit_base
        score = max(0.0, min(1.0, (fit - fit_base) / gap)) if gap > 1e-6 else 0.0
        results.append({"valid": True, "r2": round(fit, 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_instance": results}
