"""Oracle: MOSFET doping profile optimization for on/off current ratio.

1D Poisson-Drift-Diffusion in equilibrium: solve for electrostatic potential given doping
profile N(x), then compute threshold voltage and subthreshold slope.
"""
import numpy as np
from scipy.linalg import solve

L_CHANNEL = 50e-9  # 50nm channel
N_POINTS = 50
Q = 1.6e-19; EPS = 11.7 * 8.854e-12; KT = 0.026; NI = 1e16  # intrinsic carrier density

def _solve_poisson(doping_profile):
    """Solve 1D Poisson in equilibrium: d²φ/dx² = -q/ε (N_D - n_i exp(φ/V_T))."""
    dx = L_CHANNEL / (N_POINTS - 1)
    phi = np.zeros(N_POINTS)
    # Newton iteration
    for _ in range(50):
        n = NI * np.exp(phi / KT)
        rhs = -Q / EPS * (doping_profile - n)
        # Tridiagonal: (phi[i-1] - 2phi[i] + phi[i+1])/dx^2 = rhs[i]
        A = np.zeros((N_POINTS, N_POINTS))
        b = np.zeros(N_POINTS)
        for i in range(1, N_POINTS - 1):
            A[i, i-1] = 1/dx**2
            A[i, i] = -2/dx**2 - Q*NI*np.exp(phi[i]/KT)/(EPS*KT)
            A[i, i+1] = 1/dx**2
            b[i] = rhs[i] - (phi[i-1] - 2*phi[i] + phi[i+1])/dx**2
        A[0, 0] = 1; A[-1, -1] = 1  # BCs: phi=0 at contacts
        try:
            dphi = solve(A, b)
            phi += 0.3 * dphi  # damped Newton
        except:
            break
    return phi

def _compute_ion_ioff_ratio(doping):
    phi = _solve_poisson(doping)
    # Threshold voltage proxy: max potential barrier in channel
    V_barrier = np.max(phi[N_POINTS//4:3*N_POINTS//4])
    # Ion/Ioff ratio (exponential of barrier height)
    ratio = np.exp(V_barrier / KT)
    return min(ratio, 1e12)  # cap

def evaluate(design_doping):
    try:
        doping = np.asarray(design_doping(N_POINTS, L_CHANNEL), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if doping.shape != (N_POINTS,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "bad shape", "feasibility_rate": 0.0}
    doping = np.clip(doping, 1e15, 1e20)
    ratio = _compute_ion_ioff_ratio(doping)
    # Baseline: uniform doping
    ratio_base = _compute_ion_ioff_ratio(np.full(N_POINTS, 1e17))
    ratio_sota = 1e8  # target for good MOSFET
    score = max(0.0, min(1.0, (np.log10(ratio) - np.log10(max(ratio_base, 1))) /
                              (np.log10(ratio_sota) - np.log10(max(ratio_base, 1)))))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "ion_ioff_ratio_log10": round(float(np.log10(max(ratio, 1))), 2)}
