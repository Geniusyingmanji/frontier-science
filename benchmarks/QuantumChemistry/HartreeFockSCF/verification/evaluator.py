"""Oracle: Hartree-Fock SCF for H2 in minimal basis (STO-3G).

Agent provides orbital coefficients; oracle computes the HF energy using precomputed
one- and two-electron integrals. The energy is variational — lower is better.
Simplified to H2 molecule (2 basis functions, 1 occupied orbital) so all integrals
are hardcoded 2x2 matrices.
"""
import numpy as np

# Precomputed integrals for H2 at R=1.4 bohr, STO-3G basis (Szabo & Ostlund, Ch. 3)
S = np.array([[1.0, 0.6593], [0.6593, 1.0]])            # overlap
T = np.array([[0.7600, 0.2365], [0.2365, 0.7600]])      # kinetic
V = np.array([[-1.2266, -0.5974], [-0.5974, -1.2266]])  # nuclear attraction
H_CORE = T + V  # one-electron hamiltonian
# Two-electron integrals (ij|kl) stored as 2x2x2x2
ERI = np.zeros((2,2,2,2))
ERI[0,0,0,0] = 0.7746; ERI[1,1,1,1] = 0.7746
ERI[0,0,1,1] = 0.5697; ERI[1,1,0,0] = 0.5697
ERI[0,1,0,1] = 0.4441; ERI[1,0,1,0] = 0.4441
ERI[0,1,1,0] = 0.4441; ERI[1,0,0,1] = 0.4441
ERI[0,0,0,1] = 0.4441; ERI[0,0,1,0] = 0.4441
ERI[0,1,0,0] = 0.4441; ERI[1,0,0,0] = 0.4441
ERI[1,1,0,1] = 0.4441; ERI[1,1,1,0] = 0.4441
ERI[0,1,1,1] = 0.4441; ERI[1,0,1,1] = 0.4441
V_NN = 0.7143  # nuclear repulsion at R=1.4 bohr

# Exact HF energy for H2/STO-3G
E_HF_EXACT = -1.1167  # Hartree

def _hf_energy(C):
    """Compute RHF energy from orbital coefficient vector C (2,) for 1 occupied MO."""
    C = np.asarray(C, dtype=float).ravel()
    if C.shape != (2,):
        return None
    # Normalize: C^T S C = 1
    norm = C @ S @ C
    if norm < 1e-12:
        return None
    C = C / np.sqrt(norm)
    # Density matrix P = 2 * C C^T (closed-shell, 1 occupied orbital with 2 electrons)
    P = 2 * np.outer(C, C)
    # Fock matrix
    G = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    G[i,j] += P[k,l] * (ERI[i,j,k,l] - 0.5*ERI[i,l,k,j])
    F = H_CORE + G
    E_elec = 0.5 * np.sum(P * (H_CORE + F))
    return float(E_elec + V_NN)

def evaluate(optimize_orbitals):
    try:
        C = np.asarray(optimize_orbitals(H_CORE.copy(), S.copy(), ERI.copy(), V_NN), dtype=float).ravel()
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    E = _hf_energy(C)
    if E is None:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "invalid orbital", "feasibility_rate": 0.0}
    # Baseline: equal-mix orbital C = [1, 1]/sqrt(2)
    E_base = _hf_energy(np.array([1.0, 1.0]))
    # Score: energy lowering (variational)
    if E_base is None:
        E_base = -0.5
    score = max(0.0, min(1.0, (E_base - E) / (E_base - E_HF_EXACT)))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "energy_hartree": round(E, 6), "exact_hf": E_HF_EXACT}
