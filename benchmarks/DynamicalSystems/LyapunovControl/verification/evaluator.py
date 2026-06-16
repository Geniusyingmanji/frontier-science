"""Oracle: Lorenz system chaos control via Lyapunov exponent reduction.

Agent designs a feedback controller u(x) for the Lorenz system (σ=10, ρ=28, β=8/3).
Oracle integrates the controlled system + variational equations to compute the maximum
Lyapunov exponent (MLE). Goal: make MLE negative (stabilize) with minimal control energy.
"""
import numpy as np

SIGMA, RHO, BETA = 10.0, 28.0, 8.0/3.0
DT = 0.01
T_TOTAL = 100.0
N_STEPS = int(T_TOTAL / DT)
U_MAX_SQ = 50.0  # max average ||u||^2

def _rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def _compute_mle(controller):
    """Compute MLE of controlled Lorenz via Benettin's algorithm."""
    x = np.array([1.0, 1.0, 1.0])
    # Perturbation vector for variational equation
    w = np.array([1.0, 0.0, 0.0])
    w = w / np.linalg.norm(w)
    lyap_sum = 0.0
    total_u_sq = 0.0
    renorm_interval = 10  # steps between QR renormalizations

    for step in range(N_STEPS):
        # Get control
        try:
            u = np.asarray(controller(x.copy()), dtype=float).ravel()[:3]
            u = np.clip(u, -20, 20)
        except:
            u = np.zeros(3)
        total_u_sq += np.sum(u**2)

        # Lorenz + control
        def lorenz_controlled(t, state):
            s = state[:3]
            dx = SIGMA * (s[1] - s[0]) + u[0]
            dy = s[0] * (RHO - s[2]) - s[1] + u[1]
            dz = s[0] * s[1] - BETA * s[2] + u[2]
            return np.array([dx, dy, dz])

        # Jacobian of Lorenz at current state (for variational eq)
        J = np.array([
            [-SIGMA, SIGMA, 0],
            [RHO - x[2], -1, -x[0]],
            [x[1], x[0], -BETA]
        ])

        # Advance state
        x = _rk4_step(lorenz_controlled, step * DT, x, DT)

        # Advance perturbation: dw/dt = J w
        w = w + DT * (J @ w)

        # Renormalize periodically
        if (step + 1) % renorm_interval == 0:
            norm_w = np.linalg.norm(w)
            if norm_w > 1e-15:
                lyap_sum += np.log(norm_w)
                w = w / norm_w

    mle = lyap_sum / T_TOTAL
    avg_u_sq = total_u_sq / N_STEPS
    return float(mle), float(avg_u_sq)

# Uncontrolled MLE (precomputed once)
_MLE_UNCONTROLLED = None

def evaluate(design_controller):
    global _MLE_UNCONTROLLED
    if _MLE_UNCONTROLLED is None:
        _MLE_UNCONTROLLED, _ = _compute_mle(lambda x: np.zeros(3))

    try:
        controller = design_controller(SIGMA, RHO, BETA)
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}

    if not callable(controller):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "not callable", "feasibility_rate": 0.0}

    mle, avg_u_sq = _compute_mle(controller)

    # Constraint: average control energy
    if avg_u_sq > U_MAX_SQ:
        return {"combined_score": 0.0, "valid": 0.0,
                "error_message": f"control energy {avg_u_sq:.2f} > {U_MAX_SQ}", "feasibility_rate": 0.0}

    # Score: reduction in MLE (uncontrolled ~0.9, target -0.5 for full stabilization)
    mle_target = -0.5
    mle_unc = _MLE_UNCONTROLLED
    score = max(0.0, min(1.0, (mle_unc - mle) / (mle_unc - mle_target)))
    return {"combined_score": float(score), "valid": 1.0, "feasibility_rate": 1.0,
            "mle": round(mle, 4), "mle_uncontrolled": round(mle_unc, 4), "avg_control_energy": round(avg_u_sq, 4)}
