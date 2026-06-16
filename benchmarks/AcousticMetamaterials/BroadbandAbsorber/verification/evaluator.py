"""Oracle: acoustic metamaterial absorber — coupled Helmholtz resonators.

Agent designs 8 resonator geometries (cavity depth, neck length, neck radius). Oracle
computes impedance-matched absorption via transfer-matrix method for parallel-connected
resonators with viscothermal losses. Target: broadband absorption 200-2000 Hz.
"""
import numpy as np

RHO = 1.21; C = 343.0  # air properties
N_RES = 8
F_RANGE = (200, 2000)
N_FREQ = 100

def _resonator_impedance(f, cavity_depth_m, neck_length_m, neck_radius_m, cavity_radius_m):
    """Impedance of a Helmholtz resonator with viscothermal losses (Stinson model)."""
    omega = 2 * np.pi * f
    # Neck impedance (radiation + viscous)
    S_neck = np.pi * neck_radius_m**2
    # End correction
    delta = 0.85 * neck_radius_m
    l_eff = neck_length_m + 2 * delta
    # Acoustic mass
    m_a = RHO * l_eff / S_neck
    # Viscous resistance (Poiseuille)
    mu = 1.8e-5  # dynamic viscosity of air
    R_visc = 8 * mu * l_eff / (np.pi * neck_radius_m**4)
    # Cavity compliance
    V_cav = np.pi * cavity_radius_m**2 * cavity_depth_m
    C_a = V_cav / (RHO * C**2)
    # Impedance: Z = R + j(omega*m - 1/(omega*C))
    Z = R_visc + 1j * (omega * m_a - 1.0 / (omega * C_a + 1e-20))
    return Z

def _compute_absorption(depths_m, neck_lengths_m, neck_radii_m, cavity_radii_m):
    """Absorption coefficient spectrum for parallel-connected resonators."""
    freqs = np.linspace(F_RANGE[0], F_RANGE[1], N_FREQ)
    alpha = np.zeros(N_FREQ)
    for i_f, f in enumerate(freqs):
        # Parallel combination: 1/Z_total = sum(A_i/A_total * 1/Z_i)
        Y_total = 0.0 + 0j
        for r in range(N_RES):
            Z_r = _resonator_impedance(f, depths_m[r], neck_lengths_m[r],
                                        neck_radii_m[r], cavity_radii_m[r])
            A_r = np.pi * cavity_radii_m[r]**2
            Y_total += A_r / Z_r
        A_total = np.sum(np.pi * cavity_radii_m**2)
        Z_total = A_total / (Y_total + 1e-20)
        # Absorption coefficient
        Z_norm = Z_total / (RHO * C)
        r_coeff = (Z_norm - 1) / (Z_norm + 1)
        alpha[i_f] = 1 - np.abs(r_coeff)**2
    return float(np.mean(np.clip(alpha, 0, 1)))

def evaluate(design_absorber):
    try:
        result = design_absorber(N_RES, F_RANGE)
        depths = np.asarray(result["cavity_depths_mm"], dtype=float) * 1e-3
        necks_l = np.asarray(result["neck_lengths_mm"], dtype=float) * 1e-3
        necks_r = np.asarray(result["neck_radii_mm"], dtype=float) * 1e-3
        cav_r = np.asarray(result["cavity_radii_mm"], dtype=float) * 1e-3
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if any(a.shape != (N_RES,) for a in [depths, necks_l, necks_r, cav_r]):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 8 values each", "feasibility_rate": 0.0}
    # Bounds check
    depths = np.clip(depths, 5e-3, 100e-3)
    necks_l = np.clip(necks_l, 1e-3, 20e-3)
    necks_r = np.clip(necks_r, 0.5e-3, 5e-3)
    cav_r = np.clip(cav_r, 5e-3, 30e-3)
    # Total depth constraint
    if np.max(depths) + np.max(necks_l) > 120e-3:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "total depth > 120mm", "feasibility_rate": 0.0}
    mean_alpha = _compute_absorption(depths, necks_l, necks_r, cav_r)
    # Baseline: linearly spaced depths (naive)
    depths_base = np.linspace(10e-3, 80e-3, N_RES)
    necks_l_base = np.full(N_RES, 5e-3)
    necks_r_base = np.full(N_RES, 2e-3)
    cav_r_base = np.full(N_RES, 15e-3)
    alpha_base = _compute_absorption(depths_base, necks_l_base, necks_r_base, cav_r_base)
    alpha_sota = 0.92  # optimized designs from literature
    score = max(0.0, min(1.0, (mean_alpha - alpha_base) / (alpha_sota - alpha_base))) if alpha_sota > alpha_base else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "mean_absorption": round(mean_alpha, 4), "baseline_absorption": round(alpha_base, 4)}
