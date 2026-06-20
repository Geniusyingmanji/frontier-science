"""Oracle: 2D acoustic wave equation forward modeling + full waveform inversion.

Agent designs a velocity model. Oracle propagates waves through it using finite-difference
time-domain (FDTD) and compares synthetic seismograms to observed data. The 2D wave equation
is solved on a 100x100 grid — a non-trivial forward model that defeats naive optimization.
"""
import numpy as np

NX, NZ = 80, 60  # grid
DX = 10.0  # meters
DT = 0.001  # seconds
NT = 300  # time steps
N_SOURCES = 3
N_RECEIVERS = 20
F_PEAK = 15.0  # Hz, Ricker wavelet

def _ricker(t, f0):
    a = (np.pi * f0)**2
    return (1 - 2*a*t**2) * np.exp(-a*t**2)

def _true_velocity():
    """Hidden layered velocity model with a fault."""
    v = np.ones((NZ, NX)) * 2000.0  # m/s background
    v[20:, :] = 2500.0  # layer 2
    v[40:, :] = 3000.0  # layer 3
    # Fault: shifted boundary on right half
    v[25:, NX//2:] = 2500.0
    v[45:, NX//2:] = 3000.0
    return v

def _forward_model(velocity):
    """2D acoustic FDTD wave propagation, return seismograms at receivers."""
    v = np.clip(velocity, 1000, 5000)
    # Source and receiver positions
    src_x = np.linspace(10, NX-10, N_SOURCES).astype(int)
    src_z = np.full(N_SOURCES, 2, dtype=int)
    rec_x = np.linspace(5, NX-5, N_RECEIVERS).astype(int)
    rec_z = np.full(N_RECEIVERS, 2, dtype=int)
    
    seismograms = np.zeros((N_SOURCES, N_RECEIVERS, NT))
    for isrc in range(N_SOURCES):
        # Wavefield arrays
        p = np.zeros((NZ, NX))
        p_old = np.zeros((NZ, NX))
        # Source wavelet
        t_axis = np.arange(NT) * DT
        wavelet = _ricker(t_axis - 0.05, F_PEAK)
        
        for it in range(NT):
            # Laplacian (2nd order FD)
            lap = np.zeros((NZ, NX))
            lap[1:-1, 1:-1] = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4*p[1:-1, 1:-1]) / DX**2
            # Update: p_new = 2*p - p_old + v^2 * dt^2 * lap
            p_new = 2*p - p_old + v**2 * DT**2 * lap
            # Inject source
            p_new[src_z[isrc], src_x[isrc]] += wavelet[it] * DT**2 * v[src_z[isrc], src_x[isrc]]**2
            # Absorbing boundary (simple)
            p_new[0, :] = 0; p_new[-1, :] = 0
            p_new[:, 0] = 0; p_new[:, -1] = 0
            # Record at receivers
            for ir in range(N_RECEIVERS):
                seismograms[isrc, ir, it] = p_new[rec_z[ir], rec_x[ir]]
            p_old, p = p, p_new
    return seismograms

# Precompute observed data from true model
_V_TRUE = _true_velocity()
_OBS = None
def _get_obs():
    global _OBS
    if _OBS is None:
        _OBS = _forward_model(_V_TRUE)
    return _OBS

def evaluate(design_velocity):
    obs = _get_obs()
    try:
        v = np.asarray(design_velocity(NX, NZ), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if v.shape != (NZ, NX):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": f"shape {v.shape} != ({NZ},{NX})", "feasibility_rate": 0.0}
    v = np.clip(v, 1000, 5000)
    syn = _forward_model(v)
    # Waveform misfit
    misfit = float(np.sum((syn - obs)**2))
    misfit_norm = misfit / max(float(np.sum(obs**2)), 1e-10)
    # Velocity model accuracy
    v_rmse = float(np.sqrt(np.mean((v - _V_TRUE)**2)))
    v_rmse_base = float(np.sqrt(np.mean((2000 - _V_TRUE)**2)))
    # Baseline: homogeneous 2000 m/s
    syn_base = _forward_model(np.full((NZ, NX), 2000.0))
    misfit_base = float(np.sum((syn_base - obs)**2)) / max(float(np.sum(obs**2)), 1e-10)
    # Score: waveform fit improvement
    score = max(0.0, min(1.0, (misfit_base - misfit_norm) / misfit_base)) if misfit_base > 1e-10 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "waveform_misfit_norm": round(misfit_norm, 6), "velocity_rmse": round(v_rmse, 1)}
