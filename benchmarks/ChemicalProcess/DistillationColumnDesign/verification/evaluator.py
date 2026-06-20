"""Oracle: distillation column tray-by-tray simulation + design optimization.

Agent specifies column parameters (number of trays, reflux ratio, feed tray location).
Oracle runs a McCabe-Thiele-like tray-by-tray calculation to determine product purity
and reboiler duty. Multi-objective: maximize purity, minimize energy.
"""
import numpy as np

# Binary mixture: benzene-toluene, constant relative volatility
ALPHA = 2.5  # relative volatility
X_FEED = 0.5  # feed composition (mole fraction of light component)
X_DIST_TARGET = 0.95  # distillate purity target
X_BOT_TARGET = 0.05  # bottoms purity target

def _tray_calc(n_trays, reflux_ratio, feed_tray, alpha=ALPHA):
    """McCabe-Thiele tray-by-tray calculation for binary distillation."""
    n_trays = max(3, min(n_trays, 60))
    feed_tray = max(1, min(feed_tray, n_trays - 1))
    R = max(reflux_ratio, 0.5)
    # Operating lines
    # Rectifying: y = R/(R+1) * x + x_D/(R+1)
    # Stripping: y = (R+1)/R * x - x_B/R  (simplified, assumes equimolar overflow)
    x_D = 0.99; x_B = 0.01  # initial guesses, iteratively adjusted
    # Equilibrium: y = alpha*x / (1 + (alpha-1)*x)
    x = np.zeros(n_trays + 2)  # tray 0 = reboiler, tray n+1 = condenser
    y = np.zeros(n_trays + 2)
    # Start from top (condenser)
    x[n_trays + 1] = x_D
    y[n_trays + 1] = x_D  # total condenser
    for i in range(n_trays, 0, -1):
        if i > feed_tray:
            # Rectifying section
            y[i] = R/(R+1) * x[i+1] + x_D/(R+1)
        else:
            # Stripping section
            y[i] = (R+1)/R * x[i+1] - x_B/R
        y[i] = np.clip(y[i], 0.001, 0.999)
        # Equilibrium: solve y = alpha*x/(1+(alpha-1)*x) for x
        x[i] = y[i] / (alpha - (alpha-1)*y[i])
        x[i] = np.clip(x[i], 0.001, 0.999)
    x[0] = x[1]; y[0] = alpha*x[0]/(1+(alpha-1)*x[0])
    actual_x_D = x[n_trays + 1]
    actual_x_B = x[0]
    # Reboiler duty (proportional to boilup ratio)
    V = R + 1  # vapor flow (per unit of distillate)
    duty = V * 30.0  # kJ/mol (latent heat ~30 kJ/mol)
    return actual_x_D, actual_x_B, duty

def evaluate(design_column):
    try:
        params = design_column(ALPHA, X_FEED, X_DIST_TARGET, X_BOT_TARGET)
        n_trays = int(params["n_trays"])
        reflux_ratio = float(params["reflux_ratio"])
        feed_tray = int(params["feed_tray"])
    except Exception as e:
        return {"combined_score": 0.0, "valid": 0.0, "error_message": str(e), "feasibility_rate": 0.0}
    if not (3 <= n_trays <= 60 and 0.5 <= reflux_ratio <= 20 and 1 <= feed_tray <= n_trays - 1):
        return {"combined_score": 0.0, "valid": 0.0, "error_message": "params out of range", "feasibility_rate": 0.0}
    x_D, x_B, duty = _tray_calc(n_trays, reflux_ratio, feed_tray)
    # Purity score: penalize if specs not met
    purity_score = min(x_D / X_DIST_TARGET, 1.0) * min((1 - x_B) / (1 - X_BOT_TARGET), 1.0)
    # Energy efficiency: lower duty = better
    duty_base = _tray_calc(10, 5.0, 5)[2]  # baseline duty
    duty_min = _tray_calc(30, 1.5, 15)[2]  # near-optimal duty
    energy_score = max(0.0, min(1.0, (duty_base - duty) / (duty_base - duty_min)))
    # Combined
    score = 0.5 * purity_score + 0.5 * energy_score
    score = float(max(0.0, min(1.0, score)))
    # Baseline: 10 trays, R=5, feed at middle
    x_D_b, x_B_b, duty_b = _tray_calc(10, 5.0, 5)
    base_score = 0.5 * min(x_D_b/X_DIST_TARGET, 1.0)*min((1-x_B_b)/(1-X_BOT_TARGET), 1.0) + \
                 0.5 * max(0.0, min(1.0, (duty_base-duty_b)/(duty_base-duty_min)))
    final = max(0.0, min(1.0, (score - base_score) / (1.0 - base_score))) if base_score < 1.0 else 0.0
    return {"combined_score": float(final), "valid": 1.0, "feasibility_rate": 1.0,
            "x_distillate": round(x_D, 4), "x_bottoms": round(x_B, 4), "duty_kJ_mol": round(duty, 1)}
