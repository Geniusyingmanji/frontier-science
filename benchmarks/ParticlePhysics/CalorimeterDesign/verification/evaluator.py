"""Oracle: sampling calorimeter layer optimization for energy resolution.

Agent designs Pb/scintillator layer thicknesses for a 30-layer electromagnetic calorimeter.
Oracle computes energy resolution via parameterized shower model (longitudinal profile +
sampling statistics). Lower stochastic resolution term = better.
"""
import numpy as np
from scipy.special import gamma as gamma_fn

N_LAYERS = 30
X0_PB = 5.6  # radiation length of Pb in mm
MAX_TOTAL_LENGTH = 500.0  # mm
MIN_X0_TOTAL = 20.0  # minimum total depth in radiation lengths

def _shower_profile(t, E_GeV=10.0):
    """Longitudinal EM shower profile dE/dt (t = depth in X0)."""
    # Longo-Sestili parameterization
    a = 1.0 + 0.5 * np.log(E_GeV * 1000)  # ~ 4.5 for 10 GeV
    b = 0.5
    return (b**(a+1) / gamma_fn(a+1)) * t**a * np.exp(-b*t)

def _compute_resolution(passive_mm, active_mm):
    """Compute energy resolution sigma/E at 10 GeV."""
    n = len(passive_mm)
    # Total depth in X0
    total_x0 = np.sum(passive_mm) / X0_PB
    if total_x0 < MIN_X0_TOTAL:
        return 1.0, 0.0  # poor containment
    # Sampling fraction and stochastic term
    t_current = 0.0  # accumulated depth in X0
    e_sampled = 0.0
    e_total = 0.0
    layer_signals = []
    for i in range(n):
        # Passive layer
        dt_passive = passive_mm[i] / X0_PB
        t_mid_passive = t_current + dt_passive / 2
        e_passive = _shower_profile(t_mid_passive) * dt_passive
        t_current += dt_passive
        e_total += e_passive
        # Active layer (scintillator, X0 ~ 42cm = 420mm for polystyrene)
        dt_active = active_mm[i] / 420.0  # in X0
        t_mid_active = t_current + dt_active / 2
        e_active = _shower_profile(t_mid_active) * dt_active
        t_current += dt_active
        e_total += e_active
        e_sampled += e_active
        layer_signals.append(e_active)
    sampling_fraction = e_sampled / max(e_total, 1e-10)
    # Stochastic term: a/sqrt(E) where a depends on sampling
    # Empirical: a ~ 0.05 / sqrt(f_s * d_active_mean_mm)
    d_active_mean = np.mean(active_mm)
    a_stoch = 0.05 / np.sqrt(max(sampling_fraction * d_active_mean, 0.01))
    # At E = 10 GeV: sigma/E = a/sqrt(10)
    sigma_over_E = a_stoch / np.sqrt(10.0)
    # Containment factor (leakage degrades resolution)
    containment = min(1.0, total_x0 / 25.0)  # ~95% at 25 X0
    sigma_over_E /= max(containment, 0.5)
    return sigma_over_E, sampling_fraction

def evaluate(design_calorimeter):
    try:
        result = design_calorimeter(N_LAYERS, MAX_TOTAL_LENGTH)
        passive = np.asarray(result["passive_thicknesses_mm"], dtype=float)
        active = np.asarray(result["active_thicknesses_mm"], dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if passive.shape != (N_LAYERS,) or active.shape != (N_LAYERS,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 30 layers each", "feasibility_rate": 0.0}
    passive = np.clip(passive, 0.5, 5.0)
    active = np.clip(active, 1.0, 10.0)
    total_length = np.sum(passive) + np.sum(active)
    if total_length > MAX_TOTAL_LENGTH * 1.01:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": f"total {total_length:.1f} > {MAX_TOTAL_LENGTH}", "feasibility_rate": 0.0}
    sigma, fs = _compute_resolution(passive, active)
    # Baseline: uniform 2mm Pb + 4mm scint → sigma ~ 3.8%
    sigma_baseline = 0.038
    sigma_sota = 0.016  # optimized graded design
    score = max(0.0, min(1.0, (sigma_baseline - sigma) / (sigma_baseline - sigma_sota)))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "sigma_over_E_percent": round(sigma * 100, 3), "sampling_fraction": round(fs, 4),
            "total_length_mm": round(total_length, 1)}
