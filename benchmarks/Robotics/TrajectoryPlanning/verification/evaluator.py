import numpy as np
def _path_length(waypoints):
    pts = np.array(waypoints, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

def _check_collision(p1, p2, obstacles, n_check=20):
    for t in np.linspace(0, 1, n_check):
        p = (1-t)*np.array(p1) + t*np.array(p2)
        for cx, cy, r in obstacles:
            if np.sqrt((p[0]-cx)**2 + (p[1]-cy)**2) < r:
                return True
    return False

def _path_smoothness(waypoints):
    pts = np.array(waypoints, dtype=float)
    if len(pts) < 3: return 0.0
    angles = []
    for i in range(1, len(pts)-1):
        v1 = pts[i]-pts[i-1]; v2 = pts[i+1]-pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10: continue
        cos_a = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
        angles.append(abs(np.arccos(cos_a)))
    return float(np.mean(angles)) if angles else 0.0

SCENARIOS = [
    {"start": (1,1), "goal": (9,9), "obstacles": [(3,3,1),(5,5,1.5),(7,3,0.8),(4,7,1.2)], "bounds": (0,0,10,10)},
    {"start": (0.5,5), "goal": (9.5,5), "obstacles": [(3,5,1.5),(6,3,1),(6,7,1),(8,5,0.7)], "bounds": (0,0,10,10)},
]

def evaluate(plan_trajectory):
    results = []
    for sc in SCENARIOS:
        try:
            wps = plan_trajectory(sc["start"], sc["goal"], sc["obstacles"], sc["bounds"])
            wps = [np.array(w, dtype=float) for w in wps]
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if len(wps) < 2:
            results.append({"valid": False, "reason": "need >=2 waypoints", "score": 0.0}); continue
        # Check collisions
        collision = False
        for i in range(len(wps)-1):
            if _check_collision(wps[i], wps[i+1], sc["obstacles"]):
                collision = True; break
        if collision:
            results.append({"valid": False, "reason": "collision", "score": 0.0}); continue
        length = _path_length(wps)
        direct = np.linalg.norm(np.array(sc["goal"])-np.array(sc["start"]))
        baseline_len = direct  # straight line (collides, so score 0 for baseline)
        optimal_len = direct * 1.3  # rough estimate of optimal detour
        smoothness = _path_smoothness(wps)
        length_score = max(0.0, min(1.0, 2.0 - length / max(optimal_len, 0.1)))
        smooth_score = max(0.0, 1.0 - smoothness / np.pi)
        score = 0.7 * length_score + 0.3 * smooth_score
        results.append({"valid": True, "length": round(length, 4), "score": float(max(0,min(1,score)))})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)), "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid"))/len(results), "per_scenario": results}
