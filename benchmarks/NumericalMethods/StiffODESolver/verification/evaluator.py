"""Oracle for StiffODESolver — Robertson problem."""
import numpy as np
from scipy.integrate import solve_ivp

def robertson(t, y):
    y1, y2, y3 = y
    return [-0.04*y1 + 1e4*y2*y3, 0.04*y1 - 1e4*y2*y3 - 3e7*y2**2, 3e7*y2**2]

PROBLEMS = [
    {"f": robertson, "y0": [1.0, 0.0, 0.0], "t_span": [0, 1e5],
     "t_eval": np.array([0, 1, 10, 100, 1000, 10000, 100000], dtype=float),
     "rtol": 1e-6, "atol": 1e-8},
]

def evaluate(solve_ode):
    results = []
    for prob in PROBLEMS:
        # Reference solution
        ref = solve_ivp(prob["f"], prob["t_span"], prob["y0"], method="Radau",
                        t_eval=prob["t_eval"], rtol=1e-10, atol=1e-12)
        y_ref = ref.y.T
        try:
            y_hat = np.asarray(solve_ode(prob["f"], prob["y0"], prob["t_span"],
                                          prob["t_eval"], prob["rtol"], prob["atol"]), dtype=float)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if y_hat.shape != y_ref.shape:
            results.append({"valid": False, "reason": f"shape {y_hat.shape} vs {y_ref.shape}", "score": 0.0}); continue
        if not np.all(np.isfinite(y_hat)):
            results.append({"valid": False, "reason": "non-finite", "score": 0.0}); continue
        rel_err = np.linalg.norm(y_hat - y_ref) / max(np.linalg.norm(y_ref), 1e-15)
        # Euler baseline error
        y_euler = np.asarray([[1,0,0]]*len(prob["t_eval"]), dtype=float)  # crude
        err_base = np.linalg.norm(y_euler - y_ref) / max(np.linalg.norm(y_ref), 1e-15)
        score = max(0.0, min(1.0, (np.log10(err_base) - np.log10(max(rel_err, 1e-15))) /
                                   (np.log10(err_base) - np.log10(1e-10))))
        results.append({"valid": True, "rel_err": float(rel_err), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_problem": results}
