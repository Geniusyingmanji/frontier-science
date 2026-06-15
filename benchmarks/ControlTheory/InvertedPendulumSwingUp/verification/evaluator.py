"""Oracle for InvertedPendulumSwingUp."""
import numpy as np

G = 9.81; L = 1.0; M_CART = 1.0; M_PEND = 0.1; F_MAX = 10.0
DT = 0.02; T_TOTAL = 10.0; N_STEPS = int(T_TOTAL / DT)

def simulate(controller):
    state = np.array([0.0, 0.0, 0.0, 0.0])  # x, xdot, theta, thetadot (theta=0 = down)
    balanced = 0; total_force = 0.0
    for i in range(N_STEPS):
        t = i * DT
        try:
            F = float(controller(state.copy(), t, DT))
        except: F = 0.0
        F = np.clip(F, -F_MAX, F_MAX)
        total_force += abs(F)
        x, xd, th, thd = state
        sth, cth = np.sin(th), np.cos(th)
        denom = M_CART + M_PEND * sth**2
        thdd = (G * sth - cth * (F + M_PEND * L * thd**2 * sth) / (M_CART + M_PEND)) / (L * (4/3 - M_PEND * cth**2 / (M_CART + M_PEND)))
        xdd = (F + M_PEND * L * (thd**2 * sth - thdd * cth)) / (M_CART + M_PEND)
        state = np.array([x + xd*DT, xd + xdd*DT, th + thd*DT, thd + thdd*DT])
        # Check if balanced upright (theta near pi, mod 2pi)
        th_mod = state[2] % (2*np.pi)
        if abs(th_mod - np.pi) < 0.2 and abs(state[3]) < 2.0:
            balanced += 1
    return balanced / N_STEPS, total_force / N_STEPS

def evaluate(swing_up_controller):
    frac, avg_force = simulate(swing_up_controller)
    efficiency = max(0.0, 1.0 - avg_force / F_MAX * 0.3)
    score = frac * efficiency
    baseline_score = 0.0  # bang-bang rarely balances
    sota_score = 0.8  # good controller
    norm = max(0.0, min(1.0, (score - baseline_score) / (sota_score - baseline_score)))
    return {"combined_score": float(norm), "valid": 1.0,
            "frac_balanced": round(frac, 4), "avg_force": round(avg_force, 4), "feasibility_rate": 1.0}
