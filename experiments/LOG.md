# Frontier-Science — Experiment Log

All runs use the local **keyless GPT-5.5** (Azure managed-identity proxy, Responses API).
That endpoint is configured only in the git-ignored `conf/llm/local.yaml`; the repo ships a
neutral `openai_compatible.example.yaml`.

Run python: `/home/azureuser/.local/bin/python3.10` (numpy/scipy/requests/yaml/openai present).

---

## 2026-06-11 — v0 harness + LennardJonesCluster

**Harness built**: `frontier_science/` package — LLM client (chat + responses wires),
config resolver, task-spec loader (black-box contract), subprocess evaluator,
OpenEvolve-lite `evolve` loop, registry, CLI (`list` / `eval` / `run` / `smoke`).

**Endpoint smoke**: `python -m frontier_science smoke` → `FS_SMOKE_OK` (gpt-5.5, responses wire). OK.

**Task: Chemistry/LennardJonesCluster**
- Oracle: LJ energy (reduced units), normalized vs Cambridge Cluster Database global minima.
- Test sizes: N ∈ {7, 13, 19} (global minima −16.505384, −44.326801, −72.659782).
- Score: `combined_score = mean_n clip(E_found(n)/E_min(n), 0, 1)`; non-interacting gas → ~0, global minima → 1.
- **Baseline (`solution.py`, random gas)**: combined_score = **0.0767** (per-size 0.148 / 0.037 / 0.045), valid=1.0.

**Evolve run (GPT-5.5, budget 6)**: baseline **0.0767 → best 1.000** (4/6 iters accepted).
Trajectory: 0.077 → 0.989 (iter1) → 0.993 (iter3) → 1.000 (iter6).
Winning program (`runs/20260611_181348/best_program.py`, 388 lines) is **legitimate**: analytic
LJ energy+gradient, L-BFGS-B local minimization, FCC + icosahedral seed geometries, and
basin-hopping with random surface-atom relocation. No oracle access, no hardcoded final
coordinates — it reaches the catalogued global minima for N=7/13/19 by real optimization.

**Finding (benchmark design)**: with famous small sizes {7,13,19} a strong model saturates to
1.0 via a correct basin-hopping implementation. To keep the task discriminative, future
versions should add hard non-icosahedral sizes (e.g. N=38 truncated octahedron, N=75/98) where
global optimization is genuinely difficult. Tracked as a v0.1 hardening item.

**Status**: end-to-end loop proven (harness + black-box contract + keyless GPT-5.5 + continuous
reward all functioning).

---

## 2026-06-11 — two more domains added (Physics, Scientific Computing)

**Physics/SpinGlassGroundState**: Sherrington–Kirkpatrick Ising ground state. Instances
N∈{16,18,20} (seed 0); exact ground states found by full enumeration (Gray-code brute force,
cross-checked against naive `itertools` enumeration for N=16: −6.985473 ✓) and embedded as the
normalization ceiling. Score = mean over instances of clip((E_allup − E_found)/(E_allup − E_min), 0, 1).
- **Baseline (best-of-3 random)**: combined_score = **0.146**.
- Evolve (GPT-5.5, budget 5): _pending — see below._

**ScientificComputing/PoissonSolver2D**: −∇²u=f on (0,1)², Dirichlet, hidden multi-mode
manufactured solution (single-mode shortcut `u=f/(2π²)` verified wrong: err 2.98). Score =
log-scaled error reduction between Jacobi-50 (E=0.820) and 4th-order Mehrstellen (E=1.48e-6);
2nd-order direct solve ≈ 0.48, 4th-order/spectral → 1.0.
- **Baseline (Jacobi 50 sweeps)**: combined_score ≈ **0.0**.
- Evolve (GPT-5.5, budget 5): _pending — see below._

### Baseline leaderboard (v0, initial programs)

| Task | Domain | Baseline combined_score |
|---|---|---|
| LennardJonesCluster | Chemistry | 0.0767 |
| SpinGlassGroundState | Physics | 0.1459 |
| PoissonSolver2D | ScientificComputing | ~0.0 |
