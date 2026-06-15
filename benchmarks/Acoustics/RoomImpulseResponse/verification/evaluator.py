"""Oracle for RoomImpulseResponse — image-source method reference."""
import numpy as np

C = 343.0

def image_source_rir(room, src, mic, fs, order, absorb):
    Lx, Ly, Lz = room
    max_t = (order + 1) * max(room) * 2 / C
    n = int(fs * max_t) + 1
    h = np.zeros(n)
    for nx in range(-order, order + 1):
        for ny in range(-order, order + 1):
            for nz in range(-order, order + 1):
                for sx_sign in [1, -1]:
                    for sy_sign in [1, -1]:
                        for sz_sign in [1, -1]:
                            img = [nx * 2 * Lx + sx_sign * src[0],
                                   ny * 2 * Ly + sy_sign * src[1],
                                   nz * 2 * Lz + sz_sign * src[2]]
                            d = np.sqrt(sum((img[i] - mic[i])**2 for i in range(3)))
                            if d < 0.001: continue
                            refl = abs(nx) + abs(ny) + abs(nz)
                            if sx_sign < 0: refl += 1
                            if sy_sign < 0: refl += 1
                            if sz_sign < 0: refl += 1
                            # Simplified: count wall hits
                            n_walls = abs(nx) + abs(ny) + abs(nz)
                            amp = (1 - absorb) ** n_walls / d
                            idx = int(d / C * fs)
                            if 0 <= idx < n:
                                h[idx] += amp
    return h

SCENARIOS = [
    {"room": (5.0, 4.0, 3.0), "src": (1.0, 1.0, 1.5), "mic": (3.5, 2.5, 1.2),
     "fs": 8000, "max_order": 6, "absorb": 0.3},
    {"room": (8.0, 6.0, 3.5), "src": (2.0, 1.5, 1.7), "mic": (6.0, 4.0, 1.5),
     "fs": 8000, "max_order": 8, "absorb": 0.4},
]

def evaluate(compute_rir):
    results = []
    for sc in SCENARIOS:
        h_ref = image_source_rir(sc["room"], sc["src"], sc["mic"], sc["fs"], 15, sc["absorb"])
        h_base = image_source_rir(sc["room"], sc["src"], sc["mic"], sc["fs"], 0, sc["absorb"])
        try:
            h = np.asarray(compute_rir(sc["room"], sc["src"], sc["mic"],
                                        sc["fs"], sc["max_order"], sc["absorb"]), dtype=float)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        # Align lengths
        L = min(len(h_ref), len(h))
        if L < 10:
            results.append({"valid": False, "reason": "too short", "score": 0.0}); continue
        h_ref_t = h_ref[:L]; h_t = h[:L]; h_base_t = h_base[:L]
        err = np.linalg.norm(h_ref_t - h_t)
        err_base = np.linalg.norm(h_ref_t - h_base_t)
        ref_norm = np.linalg.norm(h_ref_t)
        if ref_norm < 1e-10:
            results.append({"valid": True, "score": 0.0}); continue
        snr = 20 * np.log10(ref_norm / max(err, 1e-15))
        snr_base = 20 * np.log10(ref_norm / max(err_base, 1e-15))
        snr_ceil = 60.0  # practical ceiling
        score = max(0.0, min(1.0, (snr - snr_base) / (snr_ceil - snr_base))) if snr_ceil > snr_base else 0.0
        results.append({"valid": True, "snr": round(snr, 2), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_scenario": results}
