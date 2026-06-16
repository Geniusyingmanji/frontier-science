"""Oracle: fit Arrhenius parameters to concentration-time data.

A→B→C consecutive first-order reaction system with temperature-dependent rate constants
k_i = A_i * exp(-E_i / RT). Agent fits the 4 Arrhenius parameters from noisy concentration
profiles at multiple temperatures. The stiff ODE + exponential parameter sensitivity makes
this a classic hard inverse problem in chemical kinetics.
"""
import numpy as np
from scipy.integrate import solve_ivp

R_GAS = 8.314  # J/(mol K)

# True parameters (hidden)
_TRUE = {"A1": 1e13, "E1": 80000.0, "A2": 1e11, "E2": 60000.0}  # s^-1, J/mol

TEMPERATURES = [350, 400, 450]  # K
T_SPAN = [0, 100]  # seconds
T_EVAL = np.linspace(0, 100, 50)

def _simulate(params, T_kelvin):
    A1, E1, A2, E2 = params["A1"], params["E1"], params["A2"], params["E2"]
    k1 = A1 * np.exp(-E1 / (R_GAS * T_kelvin))
    k2 = A2 * np.exp(-E2 / (R_GAS * T_kelvin))
    def rhs(t, y):
        cA, cB, cC = y
        return [-k1*cA, k1*cA - k2*cB, k2*cB]
    sol = solve_ivp(rhs, T_SPAN, [1.0, 0.0, 0.0], t_eval=T_EVAL, method='Radau', rtol=1e-8, atol=1e-10)
    return sol.y.T if sol.success else None

# Generate synthetic data
_DATA = {}
def _gen_data():
    if _DATA: return
    rng = np.random.default_rng(42)
    for T in TEMPERATURES:
        y = _simulate(_TRUE, T)
        noise = rng.normal(0, 0.02, y.shape)
        _DATA[T] = np.clip(y + noise, 0, None)

def evaluate(fit_kinetics):
    _gen_data()
    try:
        params = fit_kinetics({T: d.copy() for T, d in _DATA.items()}, TEMPERATURES, T_EVAL)
        if not isinstance(params, dict):
            arr = np.asarray(params, dtype=float).ravel()
            params = {"A1": arr[0], "E1": arr[1], "A2": arr[2], "E2": arr[3]}
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    # Validate ranges
    for k in ["A1", "E1", "A2", "E2"]:
        if k not in params:
            return {"combined_score": 0.0, "valid": 0.0, "error_message": f"missing {k}", "feasibility_rate": 0.0}
    # Compute residuals
    total_ss_res, total_ss_tot = 0.0, 0.0
    for T in TEMPERATURES:
        y_pred = _simulate(params, T)
        if y_pred is None:
            return {"combined_score": 0.0, "valid": 0.0, "error_message": "sim diverged", "feasibility_rate": 0.0}
        y_obs = _DATA[T]
        total_ss_res += np.sum((y_obs - y_pred)**2)
        total_ss_tot += np.sum((y_obs - np.mean(y_obs, axis=0))**2)
    r2 = 1 - total_ss_res / max(total_ss_tot, 1e-10)
    # Baseline R2 (wrong params)
    base_params = {"A1": 1e10, "E1": 70000, "A2": 1e10, "E2": 70000}
    ss_base = 0.0
    for T in TEMPERATURES:
        yb = _simulate(base_params, T)
        if yb is not None:
            ss_base += np.sum((_DATA[T] - yb)**2)
    r2_base = 1 - ss_base / max(total_ss_tot, 1e-10)
    score = max(0.0, min(1.0, (r2 - r2_base) / (1.0 - r2_base))) if r2_base < 1.0 else (1.0 if r2 > 0.99 else 0.0)
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "r2": round(float(r2), 6)}
