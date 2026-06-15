"""Oracle for LidDrivenCavity — compares to Ghia et al. 1982."""
import numpy as np

# Ghia et al. 1982 tabulated data for Re=100
# (y_position, u_centerline) along vertical centerline
GHIA_RE100_U = np.array([
    [1.0000, 1.0000], [0.9766, 0.8412], [0.9688, 0.7887], [0.9609, 0.7372],
    [0.9531, 0.6872], [0.8516, 0.2315], [0.7344, 0.0033], [0.6172, -0.1364],
    [0.5000, -0.2058], [0.4531, -0.2109], [0.2813, -0.1566], [0.1719, -0.1015],
    [0.1016, -0.0643], [0.0703, -0.0478], [0.0625, -0.0419], [0.0547, -0.0372],
    [0.0000, 0.0000],
])

# (x_position, v_centerline) along horizontal centerline
GHIA_RE100_V = np.array([
    [1.0000, 0.0000], [0.9688, -0.0591], [0.9609, -0.0739], [0.9531, -0.0886],
    [0.9453, -0.1031], [0.9063, -0.1656], [0.8594, -0.2109], [0.8047, -0.2566],
    [0.5000, -0.0575], [0.2344, 0.1753], [0.2266, 0.1780], [0.1563, 0.1607],
    [0.0938, 0.1232], [0.0781, 0.1089], [0.0625, 0.0923], [0.0469, 0.0713],
    [0.0000, 0.0000],
])

INSTANCES = [{"Re": 100, "N": 41}]

def evaluate(solve_cavity):
    results = []
    for inst in INSTANCES:
        Re, N = inst["Re"], inst["N"]
        try:
            u, v, p = solve_cavity(Re, N)
            u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if u.shape != (N, N) or v.shape != (N, N):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        # Extract centerline profiles
        ys = np.linspace(0, 1, N)
        xs = np.linspace(0, 1, N)
        mid = N // 2
        u_center = u[:, mid]  # u along vertical centerline
        v_center = v[mid, :]  # v along horizontal centerline
        # Interpolate to Ghia positions
        u_interp = np.interp(GHIA_RE100_U[:, 0], ys, u_center)
        v_interp = np.interp(GHIA_RE100_V[:, 0], xs, v_center)
        err_u = np.linalg.norm(u_interp - GHIA_RE100_U[:, 1])
        err_v = np.linalg.norm(v_interp - GHIA_RE100_V[:, 1])
        err = (err_u + err_v) / 2
        # Baseline error (Stokes: u=0 except top)
        u_stokes = np.zeros(N); u_stokes[-1] = 1.0
        err_base_u = np.linalg.norm(np.interp(GHIA_RE100_U[:, 0], ys, u_stokes) - GHIA_RE100_U[:, 1])
        err_base_v = np.linalg.norm(GHIA_RE100_V[:, 1])  # v=0 baseline
        err_base = (err_base_u + err_base_v) / 2
        score = max(0.0, min(1.0, (err_base - err) / err_base)) if err_base > 1e-10 else 0.0
        results.append({"valid": True, "err_u": round(float(err_u), 4),
                        "err_v": round(float(err_v), 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_instance": results}
