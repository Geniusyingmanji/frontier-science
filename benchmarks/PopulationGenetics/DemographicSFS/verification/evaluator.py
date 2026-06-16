"""Oracle: demographic model fitting to SFS via chi-squared divergence.

Simplified: uses the analytical formula for the expected SFS under a piecewise-constant
population size model (compound coalescent, Fu 1995 approximation). Avoids unstable PDE
integration. Score = chi-squared reduction vs constant-N model.
"""
import numpy as np

N_SAMPLE = 30

# True model: 3-epoch history (sizes relative to N_anc=1)
_TRUE_PARAMS = np.array([1.0, 0.1, 3.0, 0.08, 0.02])
# [N_anc, N_bot, N_cur, T_bot_start, T_bot_end] in 2*N_anc generations

def _expected_sfs_piecewise(params, n=N_SAMPLE):
    """Approximate expected SFS under piecewise-constant N(t).
    Uses the result that E[xi_i] = theta * integral_0^infty 1/N(t) * C_n(i,t) dt
    where C_n(i,t) involves coalescent probabilities. Here we use a simplified
    moment-matching approach: the SFS shape depends on the harmonic mean pop size
    seen by lineages of different ages.
    """
    N_anc, N_bot, N_cur, T_start, T_end = params
    # Time bins for numerical integration of coalescent rates
    dt = 0.001; max_t = 0.5
    ts = np.arange(0, max_t, dt)
    # Population size at each time (backward)
    N_t = np.ones_like(ts) * N_cur
    N_t[ts > T_end] = N_bot
    N_t[ts > T_start] = N_anc
    # Expected SFS: E[xi_i] ~ theta * sum_t (1/N(t)) * prob(i lineages at time t)
    # Simplified: use the reciprocal harmonic weighted by frequency
    # For frequency i/n: relevant timescale ~ i*(n-i) / (n choose 2)
    theta = 50.0  # fixed
    sfs = np.zeros(n - 1)
    for i in range(1, n):
        # Timescale where frequency i/n is most relevant
        tau = 2.0 * i * (n - i) / (n * (n - 1))  # approximate TMRCA contribution
        # Effective population size at that timescale
        idx = min(int(tau / dt), len(N_t) - 1)
        N_eff = N_t[idx]
        sfs[i-1] = theta / (i * N_eff)  # modified Watterson with effective N
    return np.maximum(sfs, 0.01)

# Generate observed SFS
_OBS = None
def _get_obs():
    global _OBS
    if _OBS is None:
        expected = _expected_sfs_piecewise(_TRUE_PARAMS)
        rng = np.random.default_rng(42)
        _OBS = expected + rng.normal(0, np.sqrt(expected) * 0.3)
        _OBS = np.maximum(_OBS, 0.1)
    return _OBS

def _chi2(obs, exp):
    exp = np.maximum(exp, 0.01)
    return float(np.sum((obs - exp)**2 / exp))

def evaluate(fit_demography):
    obs = _get_obs()
    try:
        params = np.asarray(fit_demography(obs.copy(), N_SAMPLE), dtype=float).ravel()
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if params.shape != (5,) or np.any(params <= 0):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 5 positive params", "feasibility_rate": 0.0}
    expected = _expected_sfs_piecewise(params)
    chi2_cand = _chi2(obs, expected)
    # Baseline: constant N
    chi2_base = _chi2(obs, _expected_sfs_piecewise(np.array([1.0, 1.0, 1.0, 0.08, 0.02])))
    # True params
    chi2_true = _chi2(obs, _expected_sfs_piecewise(_TRUE_PARAMS))
    score = max(0.0, min(1.0, (chi2_base - chi2_cand) / (chi2_base - chi2_true))) if chi2_base > chi2_true else 0.0
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "chi2": round(chi2_cand, 2), "chi2_baseline": round(chi2_base, 2)}
