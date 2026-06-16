"""Oracle: 2D gravity anomaly inversion — recover density from surface gravity.

Forward: g_i = sum_j G_ij * rho_j (linear). Inverse: ill-posed (condition number > 10^6).
Agent must regularize intelligently. Score combines data fit + structural recovery (SSIM).
"""
import numpy as np

NX, NZ = 20, 10  # subsurface grid
N_OBS = 30  # surface measurement stations
NOISE_STD = 0.05  # mGal

def _build_kernel():
    """Gravity kernel for rectangular prisms (Newton's law)."""
    G_CONST = 6.674e-11  # m^3/(kg s^2)
    dx, dz = 200.0, 100.0  # meters per cell
    x_obs = np.linspace(100, (NX-1)*dx - 100, N_OBS)  # observation points at surface
    G = np.zeros((N_OBS, NX * NZ))
    for iz in range(NZ):
        z_top = (iz + 0.5) * dz
        for ix in range(NX):
            x_center = (ix + 0.5) * dx
            col = iz * NX + ix
            for i_obs in range(N_OBS):
                r = np.sqrt((x_obs[i_obs] - x_center)**2 + z_top**2)
                # Simplified: point-mass approximation
                G[i_obs, col] = G_CONST * (dx * dz * 1.0) * z_top / (r**3) * 1e5  # convert to mGal-equivalent
    return G, x_obs

def _make_instance(seed):
    rng = np.random.default_rng(seed)
    G_kernel, x_obs = _build_kernel()
    # True density model: a few anomalous bodies
    rho_true = np.zeros(NX * NZ)
    # Body 1: shallow, left
    for iz in range(2, 4):
        for ix in range(3, 7):
            rho_true[iz * NX + ix] = 500.0  # kg/m^3 anomaly
    # Body 2: deep, right
    for iz in range(6, 9):
        for ix in range(12, 17):
            rho_true[iz * NX + ix] = -300.0
    g_obs = G_kernel @ rho_true + rng.normal(0, NOISE_STD, N_OBS)
    return G_kernel, g_obs, rho_true

INSTANCES = [_make_instance(42), _make_instance(7)]

def _ssim_1d(a, b):
    """Simplified structural similarity."""
    mu_a, mu_b = np.mean(a), np.mean(b)
    sig_a, sig_b = np.std(a), np.std(b)
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = 1e-4, 1e-4
    return float(((2*mu_a*mu_b + c1)*(2*sig_ab + c2)) / ((mu_a**2 + mu_b**2 + c1)*(sig_a**2 + sig_b**2 + c2)))

def evaluate(invert_gravity):
    results = []
    for G_kernel, g_obs, rho_true in INSTANCES:
        try:
            rho = np.asarray(invert_gravity(G_kernel.copy(), g_obs.copy(), NX, NZ), dtype=float).ravel()
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if rho.shape != (NX * NZ,):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        # Data fit
        g_pred = G_kernel @ rho
        chi2 = float(np.sum((g_obs - g_pred)**2) / NOISE_STD**2)
        chi2_perfect = N_OBS  # expected for noise-level fit
        chi2_base = float(np.sum(g_obs**2) / NOISE_STD**2)  # zero-model
        data_score = max(0.0, min(1.0, (chi2_base - chi2) / (chi2_base - chi2_perfect)))
        # Structural recovery
        ssim = max(0.0, _ssim_1d(rho, rho_true))
        score = 0.6 * data_score + 0.4 * ssim
        results.append({"valid": True, "chi2": round(chi2, 2), "ssim": round(ssim, 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results), "per_instance": results}
