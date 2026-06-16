"""Frozen oracle for HodgkinHuxleyFit — action potential waveform matching.

Integrates the Hodgkin-Huxley ODEs with candidate parameters and scores the resulting
voltage trace against a target (generated from true parameters). The 4-ODE system with
gating variables m, h, n and voltage V has 8 core parameters.
"""
import numpy as np
from scipy.integrate import solve_ivp

# True parameters (hidden): standard HH (squid giant axon, Hodgkin & Huxley 1952)
_TRUE = {"g_Na": 120.0, "g_K": 36.0, "g_L": 0.3,
         "E_Na": 50.0, "E_K": -77.0, "E_L": -54.4, "Cm": 1.0, "I_ext": 10.0}
DT = 0.01  # ms
T_TOTAL = 50.0  # ms
T_STIM_START, T_STIM_END = 5.0, 35.0  # current injection window

def _alpha_m(V): return 0.1*(V+40)/(1 - np.exp(-(V+40)/10)) if abs(V+40) > 1e-6 else 1.0
def _beta_m(V): return 4.0*np.exp(-(V+65)/18)
def _alpha_h(V): return 0.07*np.exp(-(V+65)/20)
def _beta_h(V): return 1.0/(1 + np.exp(-(V+35)/10))
def _alpha_n(V): return 0.01*(V+55)/(1 - np.exp(-(V+55)/10)) if abs(V+55) > 1e-6 else 0.1
def _beta_n(V): return 0.125*np.exp(-(V+65)/80)

def _hh_rhs(t, state, params):
    V, m, h, n = state
    g_Na, g_K, g_L = params["g_Na"], params["g_K"], params["g_L"]
    E_Na, E_K, E_L = params["E_Na"], params["E_K"], params["E_L"]
    Cm = params["Cm"]
    I = params["I_ext"] if T_STIM_START <= t <= T_STIM_END else 0.0
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)
    dV = (I - I_Na - I_K - I_L) / Cm
    dm = _alpha_m(V)*(1-m) - _beta_m(V)*m
    dh = _alpha_h(V)*(1-h) - _beta_h(V)*h
    dn = _alpha_n(V)*(1-n) - _beta_n(V)*n
    return [dV, dm, dh, dn]

def _simulate(params):
    V0 = -65.0; m0 = 0.05; h0 = 0.6; n0 = 0.32
    t_eval = np.arange(0, T_TOTAL, DT)
    sol = solve_ivp(lambda t, y: _hh_rhs(t, y, params),
                    [0, T_TOTAL], [V0, m0, h0, n0], t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=0.1)
    if not sol.success:
        return None
    return sol.y[0]  # voltage trace

# Precompute target
_TARGET = _simulate(_TRUE)
_T_EVAL = np.arange(0, T_TOTAL, DT)

def _rmse(a, b):
    L = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:L] - b[:L])**2)))

def evaluate(fit_hh):
    if _TARGET is None:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "target generation failed"}
    try:
        params = fit_hh()
        if not isinstance(params, dict):
            # Try as array: [g_Na, g_K, g_L, E_Na, E_K, E_L, Cm, I_ext]
            arr = np.asarray(params, dtype=float).ravel()
            params = {"g_Na": arr[0], "g_K": arr[1], "g_L": arr[2],
                      "E_Na": arr[3], "E_K": arr[4], "E_L": arr[5],
                      "Cm": arr[6], "I_ext": arr[7]}
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e)}

    # Validate parameter ranges
    for k in ["g_Na", "g_K", "g_L", "E_Na", "E_K", "E_L", "Cm", "I_ext"]:
        if k not in params:
            return {"combined_score": 0.0, "valid": 0.0, "error_message": f"missing param {k}"}

    V = _simulate(params)
    if V is None or len(V) < 100:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "simulation failed/diverged"}

    rmse = _rmse(V, _TARGET)
    # Baseline: perturbed parameters (20% off)
    baseline_params = {k: v * 1.2 for k, v in _TRUE.items()}
    V_base = _simulate(baseline_params)
    rmse_base = _rmse(V_base, _TARGET) if V_base is not None else 50.0

    score = max(0.0, min(1.0, (rmse_base - rmse) / rmse_base)) if rmse_base > 1e-6 else (1.0 if rmse < 0.1 else 0.0)
    return {
        "combined_score": float(score),
        "valid": 1.0,
        "feasibility_rate": 1.0,
        "rmse": round(rmse, 4),
        "rmse_baseline": round(rmse_base, 4),
    }
