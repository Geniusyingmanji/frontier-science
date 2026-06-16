"""Frozen oracle for TopologyOptimization — SIMP FEM compliance.

Evaluates a 2D material density field on a 60x20 grid cantilever beam (fixed left edge,
point load at bottom-right). Uses 4-node bilinear quad FEM + SIMP penalization (p=3) to
compute structural compliance. Lower compliance = stiffer = better. Volume constraint: 40%.
Based on Sigmund's "99 line" formulation (Struct. Multidisc. Optim. 21, 2001).
"""
import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

NELX, NELY = 60, 20  # mesh size
VOLFRAC = 0.4
PENAL = 3.0
E0 = 1.0  # Young's modulus of solid
Emin = 1e-9  # void stiffness


def _ke():
    """Element stiffness matrix for unit square, plane stress, E=1, nu=0.3."""
    nu = 0.3
    k = np.array([
        1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
        -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8
    ])
    KE = E0 / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
    ])
    return KE


def _compute_compliance(density):
    """FEM solve for compliance given density field (NELY, NELX)."""
    KE = _ke()
    ndof = 2 * (NELX + 1) * (NELY + 1)
    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)
    # Load: unit force downward at bottom-right corner
    F[2 * (NELX + 1) * (NELY + 1) - 1] = -1.0
    # Assemble
    for elx in range(NELX):
        for ely in range(NELY):
            n1 = (NELY + 1) * elx + ely
            n2 = (NELY + 1) * (elx + 1) + ely
            edof = np.array([
                2*n1, 2*n1+1, 2*n2, 2*n2+1,
                2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3
            ])
            rho_e = float(density[ely, elx])
            Ee = Emin + rho_e**PENAL * (E0 - Emin)
            for i in range(8):
                for j in range(8):
                    K[edof[i], edof[j]] += Ee * KE[i, j]
    # BC: fix left edge (x=0)
    fixed = []
    for ny in range(NELY + 1):
        fixed.extend([2 * ny, 2 * ny + 1])
    fixed = np.array(fixed)
    free = np.setdiff1d(np.arange(ndof), fixed)
    K_csc = csc_matrix(K)
    u = np.zeros(ndof)
    u[free] = spsolve(K_csc[np.ix_(free, free)], F[free])
    compliance = float(F @ u)
    return compliance


# Reference values (precomputed)
_C_UNIFORM = None  # lazy


def evaluate(optimize_topology):
    """Score a topology design."""
    global _C_UNIFORM
    if _C_UNIFORM is None:
        _C_UNIFORM = _compute_compliance(np.full((NELY, NELX), VOLFRAC))

    try:
        density = np.asarray(optimize_topology(NELX, NELY, VOLFRAC), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}

    if density.shape != (NELY, NELX):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": f"shape {density.shape} != ({NELY},{NELX})", "feasibility_rate": 0.0}

    # Clamp to valid range
    density = np.clip(density, 0.001, 1.0)

    # Volume constraint check
    vol = float(np.mean(density))
    if vol > VOLFRAC * 1.05:  # 5% tolerance
        return {"combined_score": 0.0, "valid": 0.0,
                "error_message": f"volume {vol:.3f} > {VOLFRAC*1.05:.3f}", "feasibility_rate": 0.0}

    compliance = _compute_compliance(density)

    # Scoring: lower compliance = better
    C_baseline = _C_UNIFORM  # uniform density
    C_sota = C_baseline * 0.30  # optimal topology ~70% reduction (literature)
    score = (C_baseline - compliance) / (C_baseline - C_sota)
    score = float(max(0.0, min(1.0, score)))

    return {
        "combined_score": score,
        "valid": 1.0,
        "feasibility_rate": 1.0,
        "compliance": round(compliance, 4),
        "compliance_baseline": round(C_baseline, 4),
        "volume_fraction": round(vol, 4),
    }
