import numpy as np
def _effectiveness_ntu(NTU, Cr, n_passes):
    if Cr < 1e-10: return 1 - np.exp(-NTU)
    if abs(Cr - 1.0) < 1e-10: return NTU / (1 + NTU)
    if n_passes == 1:
        return (1 - np.exp(-NTU*(1-Cr))) / (1 - Cr*np.exp(-NTU*(1-Cr)))
    e1 = _effectiveness_ntu(NTU/n_passes, Cr, 1)
    return ((1-e1*Cr)**n_passes - (1-e1)**n_passes) / ((1-e1*Cr)**n_passes - Cr*(1-e1)**n_passes)

SCENARIOS = [
    {"T_hot": 150, "T_cold": 20, "m_hot": 2.0, "m_cold": 3.0, "cp_hot": 1000, "cp_cold": 4180, "U": 500, "max_area": 10.0},
    {"T_hot": 300, "T_cold": 50, "m_hot": 1.5, "m_cold": 2.0, "cp_hot": 2000, "cp_cold": 1000, "U": 800, "max_area": 5.0},
]

def evaluate(design_exchanger):
    results = []
    for sc in SCENARIOS:
        try:
            area, n_passes = design_exchanger(sc["T_hot"], sc["T_cold"], sc["m_hot"], sc["m_cold"],
                                               sc["cp_hot"], sc["cp_cold"], sc["U"], sc["max_area"])
            area = float(area); n_passes = max(1, int(n_passes))
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if area <= 0 or area > sc["max_area"] * 1.01:
            results.append({"valid": False, "reason": "area out of range", "score": 0.0}); continue
        C_hot = sc["m_hot"] * sc["cp_hot"]; C_cold = sc["m_cold"] * sc["cp_cold"]
        C_min = min(C_hot, C_cold); C_max = max(C_hot, C_cold)
        Cr = C_min / C_max; NTU = sc["U"] * area / C_min
        eff = _effectiveness_ntu(NTU, Cr, n_passes)
        base_eff = _effectiveness_ntu(sc["U"]*sc["max_area"]*0.3/C_min, Cr, 1)
        max_eff = 0.95
        score = max(0.0, min(1.0, (eff - base_eff) / (max_eff - base_eff))) if max_eff > base_eff else 0.0
        results.append({"valid": True, "effectiveness": round(eff, 4), "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)), "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid"))/len(results), "per_scenario": results}
