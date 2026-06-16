"""Oracle: 1D latitude-dependent Energy Balance Model calibration.

Budyko-Sellers type: incoming solar - outgoing LW + meridional diffusion = 0 at steady state.
Agent fits parameters to match observed zonal-mean temperatures (ERA5 climatology, embedded).
Ice-albedo feedback creates bistability (ice-free vs snowball), making the landscape tricky.
"""
import numpy as np

N_LAT = 45  # latitude bands (0 to 90°, symmetric)
# Observed zonal-mean surface temperatures (°C), ERA5 1991-2020 climatology (NH, annual mean)
T_OBS = np.array([
    26.5, 26.3, 25.8, 25.0, 24.0, 22.8, 21.3, 19.6, 17.8, 15.8,
    13.7, 11.5, 9.3, 7.1, 5.0, 3.0, 1.2, -0.5, -2.1, -3.7,
    -5.3, -7.0, -8.8, -10.7, -12.7, -14.8, -17.0, -19.3, -21.7, -24.2,
    -26.7, -29.2, -31.6, -33.9, -36.0, -37.8, -39.3, -40.5, -41.4, -42.0,
    -42.5, -42.8, -43.0, -43.1, -43.2
])

def _solve_ebm(params):
    """Solve steady-state 1D EBM. params = [A, B, D, alpha_ice, alpha_ocean, T_ice, S_mult]."""
    A, B, D, alpha_ice, alpha_ocean, T_ice, S_mult = params
    x = np.linspace(0, 1, N_LAT)  # x = sin(latitude)
    S0 = 1361.0 / 4.0 * S_mult
    s = 1.0 - 0.477 * (3*x**2 - 1) / 2  # Legendre P2 solar distribution
    T = np.linspace(25, -40, N_LAT)  # initial guess
    dx = x[1] - x[0] if N_LAT > 1 else 1.0
    for _ in range(200):
        alpha = np.where(T < T_ice, alpha_ice, alpha_ocean)
        # Absorbed solar
        Q = S0 * s * (1 - alpha)
        # Diffusion (finite difference on 1-x^2 d/dx grid, simplified)
        T_new = T.copy()
        for i in range(1, N_LAT - 1):
            diff = D * ((1 - x[i]**2) * (T[i+1] - 2*T[i] + T[i-1]) / dx**2
                        - 2*x[i] * (T[i+1] - T[i-1]) / (2*dx))
            T_new[i] = T[i] + 0.1 * (Q[i] - A - B*T[i] + diff)
        T_new[0] = T_new[1]; T_new[-1] = T_new[-2]  # BCs
        if np.max(np.abs(T_new - T)) < 1e-4:
            T = T_new; break
        T = T_new
    return T

def evaluate(calibrate_ebm):
    try:
        params = np.asarray(calibrate_ebm(T_OBS.copy()), dtype=float).ravel()
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if params.shape != (7,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 7 params", "feasibility_rate": 0.0}
    # Bounds check
    if params[0] < 100 or params[0] > 300: return {"combined_score": 0.0, "valid": 0.0, "error_message": "A out of range"}
    T_model = _solve_ebm(params)
    rmse = float(np.sqrt(np.mean((T_model - T_OBS)**2)))
    # Baseline: textbook North & Coakley params
    T_base = _solve_ebm([203.3, 2.09, 0.44, 0.62, 0.30, -10.0, 1.0])
    rmse_base = float(np.sqrt(np.mean((T_base - T_OBS)**2)))
    score = max(0.0, min(1.0, (rmse_base - rmse) / rmse_base)) if rmse_base > 0.1 else (1.0 if rmse < 1.0 else 0.0)
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "rmse": round(rmse, 3), "rmse_baseline": round(rmse_base, 3)}
