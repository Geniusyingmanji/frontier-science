"""Oracle: 2D truss weight minimization under stress constraints.

10-bar truss benchmark (Schmit & Farshi 1974). Agent assigns cross-sectional areas;
oracle performs FEM analysis (stiffness method) and checks stress/displacement constraints.
"""
import numpy as np

# 10-bar truss benchmark: 6 nodes, 10 bars
NODES = np.array([
    [720, 360], [720, 0], [360, 360], [360, 0], [0, 360], [0, 0]
], dtype=float)  # inches

ELEMENTS = np.array([
    [2,4], [0,2], [3,5], [1,3], [2,3], [0,1], [3,4], [1,2], [5,4], [3,2]
]) # 0-indexed node pairs

LOADS = {1: (0, -100000), 3: (0, -100000)}  # node: (Fx, Fy) in lbs
FIXED_NODES = [4, 5]  # both x and y fixed
E_MATERIAL = 1e7  # psi (aluminum)
RHO = 0.1  # lb/in^3
SIGMA_MAX = 25000.0  # psi allowable stress
A_MIN, A_MAX = 0.1, 35.0  # in^2

def _fem_analysis(areas):
    n_nodes = len(NODES); n_elem = len(ELEMENTS)
    ndof = 2 * n_nodes
    K = np.zeros((ndof, ndof))
    lengths = np.zeros(n_elem)
    cos_sin = np.zeros((n_elem, 2))
    for e in range(n_elem):
        i, j = ELEMENTS[e]
        dx = NODES[j, 0] - NODES[i, 0]; dy = NODES[j, 1] - NODES[i, 1]
        L = np.sqrt(dx**2 + dy**2); lengths[e] = L
        c, s = dx/L, dy/L; cos_sin[e] = [c, s]
        k_local = areas[e] * E_MATERIAL / L
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        T = np.array([[c*c, c*s, -c*c, -c*s],
                      [c*s, s*s, -c*s, -s*s],
                      [-c*c, -c*s, c*c, c*s],
                      [-c*s, -s*s, c*s, s*s]])
        for ii in range(4):
            for jj in range(4):
                K[dofs[ii], dofs[jj]] += k_local * T[ii, jj]
    # Apply loads
    F = np.zeros(ndof)
    for node, (fx, fy) in LOADS.items():
        F[2*node] = fx; F[2*node+1] = fy
    # Apply BCs
    free = []
    for n in range(n_nodes):
        if n not in FIXED_NODES:
            free.extend([2*n, 2*n+1])
    free = np.array(free)
    u = np.zeros(ndof)
    K_ff = K[np.ix_(free, free)]
    try:
        u[free] = np.linalg.solve(K_ff, F[free])
    except:
        return None, None
    # Compute stresses
    stresses = np.zeros(n_elem)
    for e in range(n_elem):
        i, j = ELEMENTS[e]
        c, s = cos_sin[e]
        L = lengths[e]
        du = np.array([u[2*j]-u[2*i], u[2*j+1]-u[2*i+1]])
        strain = (c * du[0] + s * du[1]) / L
        stresses[e] = E_MATERIAL * strain
    return stresses, u

def evaluate(design_truss):
    try:
        areas = np.asarray(design_truss(10), dtype=float).ravel()
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if areas.shape != (10,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 10 areas", "feasibility_rate": 0.0}
    areas = np.clip(areas, A_MIN, A_MAX)
    stresses, u = _fem_analysis(areas)
    if stresses is None:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "FEM singular", "feasibility_rate": 0.0}
    # Check stress constraints
    max_stress = float(np.max(np.abs(stresses)))
    feasible = max_stress <= SIGMA_MAX * 1.01
    if not feasible:
        return {"combined_score": 0.0, "valid": 0.0,
                "error_message": f"stress {max_stress:.0f} > {SIGMA_MAX}", "feasibility_rate": 0.0}
    # Weight
    lengths = np.array([np.linalg.norm(NODES[ELEMENTS[e,1]] - NODES[ELEMENTS[e,0]]) for e in range(10)])
    weight = float(RHO * np.sum(areas * lengths))
    # Baseline: all bars at maximum area
    weight_baseline = float(RHO * np.sum(np.full(10, A_MAX) * lengths))
    weight_sota = 5060.0  # literature optimum (Schmit & Farshi)
    score = max(0.0, min(1.0, (weight_baseline - weight) / (weight_baseline - weight_sota)))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "weight_lbs": round(weight, 1), "max_stress_psi": round(max_stress, 0)}
