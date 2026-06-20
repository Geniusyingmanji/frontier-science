"""Oracle: invert ocean surface drifter trajectories to recover current field.

Agent designs a 2D current field u(x,y), v(x,y). Oracle advects virtual drifters through
it and compares to observed trajectories. The inverse problem is ill-posed (many current
fields produce similar trajectories) and requires spatially smooth regularization.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

NX, NY = 20, 20
N_DRIFTERS = 15
N_STEPS = 50
DT = 3600.0  # 1 hour steps
DOMAIN = [0, 200e3, 0, 200e3]  # 200km x 200km

def _true_current():
    """Hidden: double-gyre current field."""
    x = np.linspace(0, 200e3, NX); y = np.linspace(0, 200e3, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    psi = np.sin(np.pi * X / 200e3) * np.sin(2 * np.pi * Y / 200e3) * 0.5
    u = np.gradient(psi, y, axis=1)  # u = dpsi/dy
    v = -np.gradient(psi, x, axis=0)  # v = -dpsi/dx
    return u, v

def _advect_drifters(u_field, v_field, initial_pos):
    """Advect drifters using bilinear interpolation of current field."""
    x_grid = np.linspace(DOMAIN[0], DOMAIN[1], NX)
    y_grid = np.linspace(DOMAIN[2], DOMAIN[3], NY)
    interp_u = RegularGridInterpolator((x_grid, y_grid), u_field, bounds_error=False, fill_value=0)
    interp_v = RegularGridInterpolator((x_grid, y_grid), v_field, bounds_error=False, fill_value=0)
    trajectories = np.zeros((N_DRIFTERS, N_STEPS + 1, 2))
    trajectories[:, 0, :] = initial_pos
    for t in range(N_STEPS):
        pos = trajectories[:, t, :]
        u_vals = interp_u(pos)
        v_vals = interp_v(pos)
        new_pos = pos + DT * np.column_stack([u_vals, v_vals])
        new_pos[:, 0] = np.clip(new_pos[:, 0], DOMAIN[0], DOMAIN[1])
        new_pos[:, 1] = np.clip(new_pos[:, 1], DOMAIN[2], DOMAIN[3])
        trajectories[:, t+1, :] = new_pos
    return trajectories

# Precompute observations
_U_TRUE, _V_TRUE = _true_current()
_rng = np.random.default_rng(42)
_INIT_POS = np.column_stack([
    _rng.uniform(20e3, 180e3, N_DRIFTERS),
    _rng.uniform(20e3, 180e3, N_DRIFTERS)
])
_OBS_TRAJ = _advect_drifters(_U_TRUE, _V_TRUE, _INIT_POS)
# Add noise
_OBS_TRAJ += _rng.normal(0, 1000, _OBS_TRAJ.shape)  # 1km noise

def evaluate(invert_currents):
    try:
        result = invert_currents(_OBS_TRAJ.copy(), _INIT_POS.copy(), NX, NY, DOMAIN)
        u = np.asarray(result["u"], dtype=float)
        v = np.asarray(result["v"], dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if u.shape != (NX, NY) or v.shape != (NX, NY):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "bad shape", "feasibility_rate": 0.0}
    # Forward model: advect with candidate field
    pred_traj = _advect_drifters(u, v, _INIT_POS)
    err = float(np.sqrt(np.mean((pred_traj - _OBS_TRAJ)**2)))
    # Baseline: zero current
    err_base = float(np.sqrt(np.mean((_advect_drifters(np.zeros((NX,NY)), np.zeros((NX,NY)), _INIT_POS) - _OBS_TRAJ)**2)))
    score = max(0.0, min(1.0, (err_base - err) / err_base)) if err_base > 1 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "trajectory_rmse_m": round(err, 0)}
