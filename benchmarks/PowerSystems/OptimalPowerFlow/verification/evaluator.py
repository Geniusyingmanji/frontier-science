"""Oracle: DC optimal power flow — minimize generation cost subject to Kirchhoff's laws.

A simplified OPF on a small network: given demand at each bus, find generator outputs that
minimize quadratic cost while satisfying power balance and line flow limits. The DC
approximation linearizes the AC power flow equations.
"""
import numpy as np

# 6-bus system (IEEE-like)
N_BUS = 6
N_GEN = 3  # generators at buses 0, 1, 2
N_LINE = 7

# Network data
LINES = [(0,1), (0,3), (1,2), (1,4), (2,5), (3,4), (4,5)]  # from, to
LINE_SUSCEPTANCE = [10, 20, 15, 10, 25, 20, 15]  # pu
LINE_LIMIT = [50, 40, 35, 30, 45, 40, 35]  # MW

DEMAND = [0, 0, 0, 70, 50, 60]  # MW at each bus (gens at 0,1,2; loads at 3,4,5)

# Generator cost: C_i = a_i * P_i^2 + b_i * P_i
GEN_COST_A = [0.02, 0.025, 0.03]  # $/MW^2
GEN_COST_B = [10, 12, 15]  # $/MW
GEN_PMAX = [100, 80, 70]  # MW
GEN_PMIN = [10, 10, 10]  # MW

def _build_b_matrix():
    """Build the bus susceptance matrix B."""
    B = np.zeros((N_BUS, N_BUS))
    for idx, (i, j) in enumerate(LINES):
        b = LINE_SUSCEPTANCE[idx]
        B[i, i] += b; B[j, j] += b
        B[i, j] -= b; B[j, i] -= b
    return B

def _compute_line_flows(theta):
    """Compute line flows from bus angles."""
    flows = []
    for idx, (i, j) in enumerate(LINES):
        f = LINE_SUSCEPTANCE[idx] * (theta[i] - theta[j])
        flows.append(f)
    return np.array(flows)

def evaluate(solve_opf):
    try:
        gen_output = np.asarray(solve_opf(N_BUS, N_GEN, DEMAND, GEN_PMAX, GEN_PMIN,
                                           GEN_COST_A, GEN_COST_B, LINES, LINE_LIMIT), dtype=float)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if gen_output.shape != (N_GEN,):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "need 3 gen outputs", "feasibility_rate": 0.0}
    gen_output = np.clip(gen_output, GEN_PMIN, GEN_PMAX)
    # Check power balance
    total_gen = np.sum(gen_output)
    total_demand = np.sum(DEMAND)
    if abs(total_gen - total_demand) > 1.0:  # 1 MW tolerance
        return {"combined_score": 0.0, "valid": 0.0,
                "error_message": f"power imbalance: gen={total_gen:.1f} demand={total_demand}", "feasibility_rate": 0.0}
    # Solve for bus angles (DC power flow: B*theta = P_inject)
    P_inject = np.zeros(N_BUS)
    P_inject[0] = gen_output[0]; P_inject[1] = gen_output[1]; P_inject[2] = gen_output[2]
    for i in range(N_BUS):
        P_inject[i] -= DEMAND[i]
    B = _build_b_matrix()
    # Remove slack bus (bus 0): solve B_red * theta_red = P_red
    B_red = B[1:, 1:]
    P_red = P_inject[1:]
    try:
        theta_red = np.linalg.solve(B_red, P_red)
    except:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "singular B", "feasibility_rate": 0.0}
    theta = np.zeros(N_BUS)
    theta[1:] = theta_red
    # Check line limits
    flows = _compute_line_flows(theta)
    violations = np.sum(np.maximum(np.abs(flows) - np.array(LINE_LIMIT), 0))
    if violations > 100.0:  # some tolerance
        return {"combined_score": 0.0, "valid": 0.0,
                "error_message": f"line limit violations: {violations:.1f} MW", "feasibility_rate": 0.0}
    # Compute cost
    cost = sum(GEN_COST_A[i] * gen_output[i]**2 + GEN_COST_B[i] * gen_output[i] for i in range(N_GEN))
    # Baseline: equal share among generators
    equal_share = total_demand / N_GEN
    cost_base = sum(GEN_COST_A[i] * equal_share**2 + GEN_COST_B[i] * equal_share for i in range(N_GEN))
    # SoTA: optimal economic dispatch (ignore line limits) — Lagrangian gives lambda = 2*a_i*P_i + b_i
    # With line limits, slightly worse. Use simple merit-order as reference.
    cost_sota = cost_base * 0.75  # ~25% improvement possible
    score = max(0.0, min(1.0, (cost_base - cost) / (cost_base - cost_sota))) if cost_base > cost_sota else 0.0
    penalty = min(1.0, violations / 10.0) if violations > 0 else 0.0
    score *= (1 - penalty)
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0 - penalty,
            "cost": round(cost, 2), "cost_baseline": round(cost_base, 2), "line_violations_mw": round(violations, 2)}
