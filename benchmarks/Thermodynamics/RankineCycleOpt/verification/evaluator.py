"""Oracle: Rankine cycle optimization — maximize thermal efficiency.

Agent selects cycle parameters (boiler pressure, superheat temp, condenser pressure,
reheat). Oracle computes efficiency using simplified polynomial steam correlations.
"""
import numpy as np

def _h_steam(T_C, P_MPa, phase="superheated"):
    """Simplified specific enthalpy (kJ/kg) via polynomial fit."""
    if phase == "liquid":
        return 4.18 * T_C + 0.001 * P_MPa * 100  # approximate
    # Superheated steam (rough correlation for P=1-30 MPa, T=300-600°C)
    return 2500 + 2.0 * T_C + 0.5 * (T_C - 200) - 15 * P_MPa

def _s_steam(T_C, P_MPa):
    """Simplified entropy (kJ/kg/K)."""
    return 6.5 + 0.003 * T_C - 0.15 * np.log(P_MPa + 0.1)

def _compute_efficiency(params):
    P_boiler, T_super, P_cond_kPa, reheat_frac = params
    P_cond = P_cond_kPa / 1000.0  # MPa
    eta_turbine = 0.88
    eta_pump = 0.85
    # State points (simplified)
    h1 = _h_steam(T_super, P_boiler)  # turbine inlet
    # Isentropic expansion to condenser
    h2s = 2300 - 50 * P_boiler + 5 * T_super - 200 * P_cond  # rough
    h2 = h1 - eta_turbine * (h1 - h2s)
    # Condenser exit (saturated liquid)
    T_cond = 20 + 5 * P_cond_kPa / 10  # rough sat temperature
    h3 = _h_steam(T_cond, P_cond, "liquid")
    # Pump work
    v_liquid = 0.001  # m^3/kg
    w_pump = v_liquid * (P_boiler - P_cond) * 1000 / eta_pump  # kJ/kg
    h4 = h3 + w_pump
    # Reheat effect (if reheat_frac > 0)
    reheat_bonus = reheat_frac * 0.03  # up to 3% efficiency gain
    # Net work and heat input
    w_net = (h1 - h2) - w_pump
    q_in = h1 - h4
    if q_in <= 0 or w_net <= 0:
        return 0.0
    eta = w_net / q_in + reheat_bonus
    # Moisture constraint: exit quality > 0.88
    quality = 0.88 + 0.001 * T_super - 0.01 * P_boiler
    if quality < 0.88:
        eta *= 0.5  # heavy penalty
    return float(np.clip(eta, 0, 0.55))

def evaluate(optimize_rankine):
    try:
        params = np.asarray(optimize_rankine(), dtype=float).ravel()
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if params.shape != (4,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need [P_boiler, T_super, P_cond_kPa, reheat_frac]", "feasibility_rate": 0.0}
    P_b, T_s, P_c, rh = params
    if not (5 <= P_b <= 30 and 400 <= T_s <= 620 and 3 <= P_c <= 15 and 0 <= rh <= 1):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "params out of range", "feasibility_rate": 0.0}
    eta = _compute_efficiency(params)
    eta_base = _compute_efficiency(np.array([10.0, 500.0, 10.0, 0.0]))  # simple Rankine
    eta_sota = 0.46  # ultra-supercritical target
    score = max(0.0, min(1.0, (eta - eta_base) / (eta_sota - eta_base))) if eta_sota > eta_base else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "thermal_efficiency": round(eta, 4), "baseline_efficiency": round(eta_base, 4)}
