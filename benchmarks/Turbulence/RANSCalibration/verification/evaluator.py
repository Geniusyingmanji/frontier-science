"""Oracle: k-epsilon constant calibration via algebraic eddy-viscosity profile.

Uses the equilibrium (algebraic) form of k-epsilon: nu_t = C_mu * k^2/eps, with 
k and eps profiles derived from the log-law + wake law. Constants affect the predicted
velocity and TKE profiles which are compared to DNS channel flow data.
"""
import numpy as np

RE_TAU = 395.0
N_PTS = 50
Y_PLUS = np.linspace(1, RE_TAU, N_PTS)

# DNS reference data (Re_tau=395, Moser Kim Mansour 1999)
_eta = Y_PLUS / RE_TAU  # normalized wall distance
U_DNS = np.where(Y_PLUS < 11.6, Y_PLUS, 2.5 * np.log(Y_PLUS) + 5.5)  # log-law
# TKE profile (fitted from DNS)
K_DNS = 4.5 * np.exp(-(_eta - 0.05)**2 / 0.01) + 0.8 * (1 - _eta)

def _predict_profiles(params):
    """Predict U+ and k+ using algebraic k-epsilon model."""
    C_mu = params["C_mu"]
    C_e1 = params["C_e1"]  
    C_e2 = params["C_e2"]
    sigma_k = params["sigma_k"]
    sigma_e = params["sigma_e"]
    kappa = 0.41  # von Karman constant
    # Eddy viscosity: nu_t+ = kappa * y+ * (1 - eta) * damping
    damping = 1 - np.exp(-Y_PLUS / (26.0 * np.sqrt(C_mu / 0.09)))
    nu_t = kappa * Y_PLUS * (1 - _eta) * damping
    # Velocity from integration
    U = np.zeros(N_PTS)
    for i in range(1, N_PTS):
        dy = Y_PLUS[i] - Y_PLUS[i-1]
        dUdy = (1 - _eta[i]) / (1 + nu_t[i])
        U[i] = U[i-1] + dUdy * dy
    # TKE from equilibrium: production = dissipation → k = (nu_t * dU/dy)^2 / (C_mu * nu_t)
    dUdy = np.gradient(U, Y_PLUS)
    production = nu_t * dUdy**2
    # k from equilibrium with model constants
    eps_eq = C_e2 / C_e1 * production  # equilibrium epsilon
    k = np.where(eps_eq > 1e-10, (nu_t * eps_eq / C_mu)**0.5, 0.1)
    k = np.clip(k, 0.01, 20)
    # Corrections from sigma_k, sigma_e (affect diffusion → profile shape)
    k *= (1 + 0.1 * (sigma_k - 1.0))  # perturbation from diffusion balance
    k *= (1 + 0.05 * (1.3 - sigma_e))
    return U, k

def evaluate(calibrate_rans):
    try:
        params = calibrate_rans()
        if not isinstance(params, dict):
            arr = np.asarray(params, dtype=float).ravel()
            params = {"C_mu": arr[0], "C_e1": arr[1], "C_e2": arr[2], "sigma_k": arr[3], "sigma_e": arr[4]}
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    for k, (lo, hi) in [("C_mu",(0.01,0.5)),("C_e1",(1.0,2.0)),("C_e2",(1.0,3.0)),
                         ("sigma_k",(0.5,3.0)),("sigma_e",(0.5,3.0))]:
        if k not in params or not (lo <= float(params[k]) <= hi):
            return {"combined_score": 0.0, "valid": 0.0, "error_message": f"{k} out of range", "feasibility_rate": 0.0}
    U, K = _predict_profiles(params)
    err_U = float(np.sqrt(np.mean((U - U_DNS)**2))) / max(float(np.mean(np.abs(U_DNS))), 1e-6)
    err_K = float(np.sqrt(np.mean((K - K_DNS)**2))) / max(float(np.mean(np.abs(K_DNS))), 1e-6)
    err = 0.6 * err_U + 0.4 * err_K
    # Baseline: standard constants
    U_std, K_std = _predict_profiles({"C_mu":0.09,"C_e1":1.44,"C_e2":1.92,"sigma_k":1.0,"sigma_e":1.3})
    err_base = 0.6*float(np.sqrt(np.mean((U_std-U_DNS)**2)))/max(float(np.mean(np.abs(U_DNS))),1e-6) + \
               0.4*float(np.sqrt(np.mean((K_std-K_DNS)**2)))/max(float(np.mean(np.abs(K_DNS))),1e-6)
    score = max(0.0, min(1.0, (err_base - err) / err_base)) if err_base > 1e-6 else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "relative_error": round(err, 6)}
