"""Frozen oracle for ProteinLatticeHP (hidden from the agent).

The HP (Hydrophobic-Polar) lattice model is a simplified model of protein folding:
a sequence of H (hydrophobic) and P (polar) residues is placed on a 2D square lattice
as a self-avoiding walk. The energy = -(number of non-bonded H-H contacts). Finding the
minimum energy conformation is NP-hard. Known optimal energies for standard benchmark
sequences from Dill (1985) and Unger & Moult (1993) are embedded below.

The agent's task is to implement a folding algorithm; the oracle counts valid H-H contacts.
"""

from __future__ import annotations

import numpy as np

# Standard HP benchmark sequences with known optimal energies (2D square lattice).
# Sources: Dill (1985), Unger & Moult (1993), Yue & Dill (1995).
SEQUENCES = [
    {"id": "HP20", "seq": "HPHPPHHPHPPHPHHPPHPH", "len": 20, "optimal_energy": -9,
     "note": "Unger & Moult 1993"},
    {"id": "HP25", "seq": "HPHPPHHPHPPHPHHPPHPHPHPPHH", "len": 25, "optimal_energy": -8,
     "note": "classic 25-mer"},
    {"id": "HP36", "seq": "PPPHHPPHHPPPPPHHHHHHHPPHHPPPPHHPPHPP", "len": 36,
     "optimal_energy": -14, "note": "Dill benchmark"},
]

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # R, L, U, D


def score_fold(seq: str, coords: list) -> dict:
    n = len(seq)
    try:
        pts = [(int(x), int(y)) for x, y in coords]
    except Exception as exc:
        return {"valid": False, "reason": f"bad coords: {exc}", "energy": 0}
    if len(pts) != n:
        return {"valid": False, "reason": f"expected {n} coords, got {len(pts)}", "energy": 0}
    # Check self-avoiding
    if len(set(pts)) != n:
        return {"valid": False, "reason": "self-intersection", "energy": 0}
    # Check connectivity (each consecutive pair must be lattice-adjacent)
    for i in range(n - 1):
        dx = abs(pts[i + 1][0] - pts[i][0])
        dy = abs(pts[i + 1][1] - pts[i][1])
        if dx + dy != 1:
            return {"valid": False, "reason": f"non-adjacent at {i}->{i+1}", "energy": 0}
    # Count H-H contacts (non-bonded, lattice-adjacent)
    pos_set = {p: i for i, p in enumerate(pts)}
    contacts = 0
    for i in range(n):
        if seq[i] != "H":
            continue
        x, y = pts[i]
        for dx, dy in DIRS:
            nb = (x + dx, y + dy)
            if nb in pos_set:
                j = pos_set[nb]
                if j > i + 1 and seq[j] == "H":  # non-bonded (j != i+1), both H
                    contacts += 1
    energy = -contacts
    return {"valid": True, "energy": energy, "contacts": contacts}


def evaluate(fold_protein) -> dict:
    results = []
    for entry in SEQUENCES:
        seq = entry["seq"]
        opt_e = entry["optimal_energy"]
        try:
            coords = fold_protein(seq)
        except Exception as exc:
            results.append({"id": entry["id"], "valid": False, "reason": str(exc), "score": 0.0})
            continue
        r = score_fold(seq, coords)
        if not r["valid"]:
            results.append({"id": entry["id"], "valid": False, "reason": r["reason"], "score": 0.0})
            continue
        # Score: fraction of optimal contacts achieved (0 = no contacts, 1 = optimal)
        if opt_e == 0:
            score = 1.0
        else:
            score = r["energy"] / opt_e  # both negative, so this is in [0,1]
            score = float(min(1.0, max(0.0, score)))
        results.append({"id": entry["id"], "valid": True, "energy": r["energy"],
                        "optimal": opt_e, "score": score})
    scores = [r["score"] for r in results]
    n_valid = sum(1 for r in results if r.get("valid"))
    return {
        "combined_score": float(np.mean(scores)) if scores else 0.0,
        "valid": 1.0 if n_valid == len(SEQUENCES) else 0.0,
        "feasibility_rate": n_valid / len(SEQUENCES),
        "per_sequence": results,
    }
