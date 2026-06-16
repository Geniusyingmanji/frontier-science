"""Oracle: 1D two-group neutron diffusion k-eigenvalue problem.

Optimizes fuel enrichment distribution across 20 zones in a 1D slab reactor (100 cm)
to maximize k_eff subject to average enrichment <= 5%. Uses power iteration on the
two-group diffusion equation discretized with finite differences.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

N_ZONES = 20
N_MESH = 200
SLAB_WIDTH = 100.0  # cm
AVG_ENRICH_MAX = 0.05

def _cross_sections(enrichment):
    """Two-group macroscopic cross-sections as linear functions of U-235 enrichment."""
    e = np.clip(enrichment, 0.02, 0.20)
    # Group 1 (fast): D1, Sigma_r1, nu*Sigma_f1, Sigma_s12
    D1 = 1.5 - 0.5 * e
    Sr1 = 0.02 + 0.08 * e
    nuSf1 = 0.005 + 0.05 * e
    Ss12 = 0.015 + 0.01 * e
    # Group 2 (thermal): D2, Sigma_a2, nu*Sigma_f2
    D2 = 0.4 - 0.3 * e
    Sa2 = 0.08 + 0.6 * e
    nuSf2 = 0.05 + 1.5 * e
    return D1, Sr1, nuSf1, Ss12, D2, Sa2, nuSf2

def _compute_keff(enrichments):
    """Power iteration for k_eff given enrichment per zone."""
    h = SLAB_WIDTH / N_MESH
    zone_width = N_MESH // N_ZONES
    # Map each mesh point to its zone enrichment
    e_mesh = np.zeros(N_MESH)
    for z in range(N_ZONES):
        start = z * zone_width
        end = start + zone_width if z < N_ZONES - 1 else N_MESH
        e_mesh[start:end] = enrichments[z]
    D1, Sr1, nuSf1, Ss12, D2, Sa2, nuSf2 = _cross_sections(e_mesh)
    # Build diffusion matrices (tridiagonal)
    def diffusion_matrix(D, Sigma_r):
        diag_main = 2 * D / h**2 + Sigma_r
        diag_off = -D[:-1] / h**2  # simplified: average D at interfaces
        A = diags([diag_main, np.append(diag_off, 0), np.append(0, diag_off)],
                  [0, -1, 1], shape=(N_MESH, N_MESH), format='csc')
        return A
    A1 = diffusion_matrix(D1, Sr1)
    A2 = diffusion_matrix(D2, Sa2)
    # Power iteration
    phi1 = np.ones(N_MESH)
    phi2 = np.ones(N_MESH)
    k = 1.0
    for _ in range(500):
        # Fission source
        F = nuSf1 * phi1 + nuSf2 * phi2
        # Solve group 1: A1 phi1 = (1/k) chi1 F
        rhs1 = F / k  # chi1 = 1.0 (all fission neutrons born fast)
        phi1_new = spsolve(A1, rhs1)
        # Scattering source for group 2
        rhs2 = Ss12 * phi1_new
        phi2_new = spsolve(A2, rhs2)
        # Update k
        F_new = nuSf1 * phi1_new + nuSf2 * phi2_new
        k_new = k * np.sum(F_new) / np.sum(F)
        # Normalize
        norm = np.sqrt(np.sum(phi1_new**2) + np.sum(phi2_new**2))
        phi1, phi2 = phi1_new / norm, phi2_new / norm
        if abs(k_new - k) < 1e-7:
            k = k_new
            break
        k = k_new
    return float(k)

# Reference: uniform 5% enrichment
_K_UNIFORM = None

def evaluate(optimize_enrichment):
    global _K_UNIFORM
    if _K_UNIFORM is None:
        _K_UNIFORM = _compute_keff(np.full(N_ZONES, 0.05))
    try:
        enrichments = np.asarray(optimize_enrichment(N_ZONES, AVG_ENRICH_MAX), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if enrichments.shape != (N_ZONES,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "bad shape", "feasibility_rate": 0.0}
    enrichments = np.clip(enrichments, 0.02, 0.20)
    if np.mean(enrichments) > AVG_ENRICH_MAX * 1.01:
        return {"combined_score": 0.0, "valid": 0.0,
                "error_message": f"avg enrichment {np.mean(enrichments):.4f} > {AVG_ENRICH_MAX}", "feasibility_rate": 0.0}
    k = _compute_keff(enrichments)
    k_baseline = _K_UNIFORM
    k_sota = k_baseline + 0.07  # ~7% pcm improvement is a good target
    score = max(0.0, min(1.0, (k - k_baseline) / (k_sota - k_baseline)))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "k_eff": round(k, 6), "k_baseline": round(k_baseline, 6), "avg_enrichment": round(float(np.mean(enrichments)), 4)}
