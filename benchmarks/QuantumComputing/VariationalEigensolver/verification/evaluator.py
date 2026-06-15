"""Oracle for VariationalEigensolver."""
import numpy as np

def _make_hamiltonian(n_qubits, seed):
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    H = rng.standard_normal((dim, dim))
    H = (H + H.T) / 2  # symmetric real
    # Add diagonal dominance for physical realism
    H += np.diag(rng.uniform(0, 5, dim))
    return H

INSTANCES = [
    {"n_qubits": 4, "n_layers": 3, "seed": 42},
    {"n_qubits": 5, "n_layers": 4, "seed": 7},
    {"n_qubits": 6, "n_layers": 5, "seed": 13},
]

def evaluate(find_ground_state):
    results = []
    for inst in INSTANCES:
        H = _make_hamiltonian(inst["n_qubits"], inst["seed"])
        e_exact = float(np.linalg.eigvalsh(H)[0])
        # Random baseline energy
        rng = np.random.default_rng(0)
        dim = 2 ** inst["n_qubits"]
        psi_rand = rng.standard_normal(dim) + 0j; psi_rand /= np.linalg.norm(psi_rand)
        e_rand = float(np.real(psi_rand.conj() @ H @ psi_rand))
        try:
            psi = np.asarray(find_ground_state(H.copy(), inst["n_qubits"], inst["n_layers"]), dtype=complex)
        except Exception as e:
            results.append({"valid": False, "reason": str(e), "score": 0.0}); continue
        if psi.shape != (dim,):
            results.append({"valid": False, "reason": "bad shape", "score": 0.0}); continue
        psi = psi / np.linalg.norm(psi)  # renormalize
        e_found = float(np.real(psi.conj() @ H @ psi))
        gap = e_rand - e_exact
        score = max(0.0, min(1.0, (e_rand - e_found) / gap)) if gap > 1e-10 else 0.0
        results.append({"valid": True, "energy": round(e_found, 6), "exact": round(e_exact, 6),
                        "score": float(score)})
    scores = [r["score"] for r in results]
    return {"combined_score": float(np.mean(scores)),
            "valid": 1.0 if all(r.get("valid") for r in results) else 0.0,
            "feasibility_rate": sum(1 for r in results if r.get("valid")) / len(results),
            "per_instance": results}
