"""Oracle: PDE-constrained heat source optimization.

Agent places heat sources in a 2D domain with convection-diffusion. Oracle solves the
steady-state convection-diffusion PDE and compares the temperature field to a target.
The PDE solve is required for each evaluation — can't shortcut.
"""
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve

NX, NY = 40, 40
N_SOURCES = 6
KAPPA = 0.01  # thermal diffusivity
VX, VY = 0.5, 0.3  # convection velocity

def _target_field():
    """Hidden target temperature distribution."""
    x = np.linspace(0, 1, NX); y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    T = 0.5 * np.exp(-20*((X-0.3)**2 + (Y-0.7)**2)) + \
        0.8 * np.exp(-15*((X-0.7)**2 + (Y-0.3)**2)) + \
        0.3 * np.exp(-25*((X-0.5)**2 + (Y-0.5)**2))
    return T

def _solve_convdiff(source_field):
    """Solve steady-state convection-diffusion: -kappa*lap(T) + v*grad(T) = Q."""
    N = NX * NY; dx = 1.0/(NX-1); dy = 1.0/(NY-1)
    idx = lambda i,j: i*NY+j
    # Build sparse system
    data_main = np.zeros(N); data_xm = np.zeros(N); data_xp = np.zeros(N)
    data_ym = np.zeros(N); data_yp = np.zeros(N)
    rhs = np.zeros(N)
    for i in range(NX):
        for j in range(NY):
            k = idx(i,j)
            if i==0 or i==NX-1 or j==0 or j==NY-1:
                data_main[k] = 1.0; rhs[k] = 0.0  # Dirichlet BC
                continue
            # Diffusion: -kappa*(T_{i+1}+T_{i-1}-2T)/dx^2
            cx = KAPPA/dx**2; cy = KAPPA/dy**2
            # Upwind convection
            ax = VX/(2*dx); ay = VY/(2*dy)
            data_main[k] = 2*cx + 2*cy + abs(VX)/dx + abs(VY)/dy
            data_xm[k] = -cx - max(ax, 0); data_xp[k] = -cx + min(ax, 0)
            data_ym[k] = -cy - max(ay, 0); data_yp[k] = -cy + min(ay, 0)
            rhs[k] = source_field[i, j]
    # Assemble
    offsets = [0, -NY, NY, -1, 1]
    A = diags([data_main, data_xm[NY:], data_xp[:-NY], data_ym[1:], data_yp[:-1]],
              offsets, shape=(N,N), format='csc')
    T = spsolve(A, rhs).reshape(NX, NY)
    return T

_T_TARGET = _target_field()

def evaluate(place_sources):
    try:
        result = place_sources(NX, NY, N_SOURCES)
        positions = np.asarray(result["positions"], dtype=float)  # (N_SOURCES, 2) in [0,1]
        strengths = np.asarray(result["strengths"], dtype=float)  # (N_SOURCES,) > 0
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if positions.shape != (N_SOURCES, 2) or strengths.shape != (N_SOURCES,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "bad shape", "feasibility_rate": 0.0}
    positions = np.clip(positions, 0.05, 0.95)
    strengths = np.clip(strengths, 0, 10)
    # Build source field from Gaussians
    x = np.linspace(0, 1, NX); y = np.linspace(0, 1, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    Q = np.zeros((NX, NY))
    for s in range(N_SOURCES):
        sx, sy = positions[s]
        Q += strengths[s] * np.exp(-50*((X-sx)**2 + (Y-sy)**2))
    T = _solve_convdiff(Q)
    err = float(np.sqrt(np.mean((T - _T_TARGET)**2)))
    # Baseline: sources at center
    Q_base = np.zeros((NX, NY))
    for s in range(N_SOURCES):
        Q_base += 1.0 * np.exp(-50*((X-0.5)**2 + (Y-0.5)**2))
    T_base = _solve_convdiff(Q_base)
    err_base = float(np.sqrt(np.mean((T_base - _T_TARGET)**2)))
    score = max(0.0, min(1.0, (err_base - err) / err_base)) if err_base > 1e-6 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "temperature_rmse": round(err, 6)}
