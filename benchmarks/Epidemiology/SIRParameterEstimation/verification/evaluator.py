"""Oracle for SIRParameterEstimation."""
import numpy as np
from scipy.integrate import solve_ivp

def sim_sir(beta, gamma, S0, I0, R0, t):
    def rhs(t, y):
        S, I, R = y; N = S+I+R
        return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]
    sol = solve_ivp(rhs, [t[0], t[-1]], [S0, I0, R0], t_eval=t, rtol=1e-8, atol=1e-8)
    return sol.y[1] if sol.success else np.full(len(t), np.nan)

INSTANCES = []
def _make():
    if INSTANCES: return
    for seed, params in enumerate([
        (0.4, 0.15, 10000),  # beta, gamma, N
        (0.25, 0.08, 50000),
    ]):
        beta, gamma, N = params
        rng = np.random.default_rng(seed + 300)
        t = np.arange(0, 60, dtype=float)
        I_true = sim_sir(beta, gamma, N-10, 10, 0, t)
        I_obs = np.maximum(I_true + rng.normal(0, max(I_true.max()*0.05, 1), len(t)), 0)
        INSTANCES.append({"t": t, "I_obs": I_obs, "N": N, "beta": beta, "gamma": gamma, "I_true": I_true})

def evaluate(estimate_sir):
    _make()
    results = []
    for inst in INSTANCES:
        try:
            params = estimate_sir(inst["t"].copy(), inst["I_obs"].copy(), inst["N"])
            beta, gamma, S0, I0, R0 = [float(p) for p in params]
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        I_pred = sim_sir(beta, gamma, S0, I0, R0, inst["t"])
        if np.any(np.isnan(I_pred)):
            results.append({"valid": False, "reason": "sim failed", "score": 0.0}); continue
        ss_res = np.sum((inst["I_obs"] - I_pred)**2)
        ss_tot = np.sum((inst["I_obs"] - np.mean(inst["I_obs"]))**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        # Baseline R2
        I_base = sim_sir(0.3, 0.1, inst["N"]-inst["I_obs"][0], inst["I_obs"][0], 0, inst["t"])
        r2_base = 1 - np.sum((inst["I_obs"]-I_base)**2)/max(ss_tot, 1e-10) if not np.any(np.isnan(I_base)) else -1
        gap = 1.0 - r2_base
        score = max(0.0, min(1.0, (r2 - r2_base) / gap)) if gap > 1e-6 else 0.0
        results.append({"valid": True, "r2": round(float(r2), 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_instance": results}
