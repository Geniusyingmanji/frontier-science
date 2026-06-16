"""Oracle: atmospheric temperature profile retrieval from thermal IR radiances.

Agent fits a temperature profile T(p) to synthetic satellite-observed radiances.
Forward model: multi-layer two-stream thermal emission through 20 atmospheric layers.
Weighting functions determine which layers contribute to each channel.
"""
import numpy as np

N_LAYERS = 20
N_CHANNELS = 10
P_LEVELS = np.linspace(1000, 50, N_LAYERS + 1)  # hPa, surface to top

# True temperature profile (hidden): US Standard Atmosphere simplified
_T_TRUE = 288.0 - 6.5 * np.linspace(0, 12, N_LAYERS)  # lapse rate 6.5 K/km
_T_TRUE[15:] = 216.0  # tropopause/stratosphere

def _planck(T, nu):
    """Simplified Planck function B(T, nu) in arbitrary units."""
    h, k, c = 6.626e-34, 1.381e-23, 3e8
    x = h * nu / (k * T)
    x = np.clip(x, 0, 500)
    return 2 * h * nu**3 / c**2 / (np.exp(x) - 1 + 1e-30)

def _forward_model(T_profile):
    """Compute TOA radiances for the given temperature profile."""
    # Channel frequencies (covering 15μm CO2 band region)
    nu_channels = np.linspace(600e9, 750e9, N_CHANNELS) * 100  # wavenumber-like
    # Optical depths per layer (fixed absorber amounts, exponential with height)
    tau_layers = 0.3 * np.exp(-np.arange(N_LAYERS) * 0.15)
    radiances = np.zeros(N_CHANNELS)
    for ch in range(N_CHANNELS):
        nu = nu_channels[ch]
        # Channel weighting depends on frequency (different penetration depth)
        tau_ch = tau_layers * (1 + 0.3 * np.sin(ch * 0.5))
        # TOA radiance: sum of layer emissions weighted by transmittance
        transmittance = 1.0
        L = 0.0
        for k in range(N_LAYERS - 1, -1, -1):  # top to bottom... actually bottom to top
            emission = _planck(T_profile[k], nu) * (1 - np.exp(-tau_ch[k]))
            L += emission * transmittance
            transmittance *= np.exp(-tau_ch[k])
        # Add surface contribution
        L += _planck(T_profile[0], nu) * transmittance * 0.98  # surface emissivity
        radiances[ch] = L
    return radiances

# Generate observed radiances with noise
_rng = np.random.default_rng(42)
_L_OBS = _forward_model(_T_TRUE) * (1 + _rng.normal(0, 0.01, N_CHANNELS))

def evaluate(retrieve_profile):
    try:
        T = np.asarray(retrieve_profile(_L_OBS.copy(), N_LAYERS, N_CHANNELS), dtype=float).ravel()
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if T.shape != (N_LAYERS,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 20 layers", "feasibility_rate": 0.0}
    T = np.clip(T, 150, 350)
    # Forward model residual
    L_pred = _forward_model(T)
    residual = np.linalg.norm(L_pred - _L_OBS) / np.linalg.norm(_L_OBS)
    # Profile accuracy
    rmse_T = float(np.sqrt(np.mean((T - _T_TRUE)**2)))
    # Baseline: isothermal 250K
    T_base = np.full(N_LAYERS, 250.0)
    rmse_base = float(np.sqrt(np.mean((T_base - _T_TRUE)**2)))
    score = max(0.0, min(1.0, (rmse_base - rmse_T) / rmse_base)) if rmse_base > 0.1 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "rmse_K": round(rmse_T, 3), "rmse_baseline_K": round(rmse_base, 3),
            "spectral_residual": round(residual, 6)}
