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

### Evolve results (GPT-5.5, keyless, budget 5–6)

| Task | Baseline | Best | Iters to best | Winning method (verified legitimate) |
|---|---|---|---|---|
| LennardJonesCluster | 0.0767 | **1.000** | 6 | analytic LJ energy+grad, L-BFGS-B, FCC/icosahedral seeds, basin-hopping |
| SpinGlassGroundState | 0.1459 | **1.000** | 1 | greedy + tabu search + spectral-eigenvector starts + iterated local search (248 lines) |
| PoissonSolver2D | ~0.0 | **1.000** | 1 | spectral solve — divide each mode by its eigenvalue (27 lines) |

All three winners are genuine, correct scientific optimizers with **no oracle access and no
hardcoded answers** (checked by reading each `best_program.py`).

### KEY FINDING — v0 instances saturate for a frontier model

GPT-5.5 drives **all three tasks to combined_score ≈ 1.0**, usually in one iteration, by
writing the textbook-strong algorithm (basin-hopping / tabu+spectral ILS / spectral solver).
The harness, black-box contract, keyless GPT-5.5 loop, and continuous reward are fully proven
end-to-end and across three domains — but the current *instances* are too easy to discriminate
strong models. This is the central design lesson for turning the proof-of-concept into a real
benchmark.

### v0.1 hardening plan (make scores live in (0,1) for frontier models)

- **LennardJonesCluster**: add hard non-icosahedral sizes — N=38 (FCC truncated octahedron),
  N=75/76/77 (Marks decahedra), N=98 — where basin-hopping needs many restarts; cap per-size
  wall-clock so a single L-BFGS pass cannot reach the global minimum.
- **SpinGlassGroundState**: scale to N=40–80 where exact ground states are unknown; normalize
  against a strong reference (e.g. best of long parallel tempering) and allow scores >1 to be
  reported (uncapped) so genuine improvements over the reference are visible; add many seeded
  instances to reduce variance.
- **PoissonSolver2D**: switch to an accuracy×cost objective with a FLOP/wall-clock budget, and
  a non-separable RHS (no clean modal spectral shortcut), so higher-order *and* efficient
  solvers are rewarded rather than an exact modal reconstruction.
- Cross-cutting: report sample-efficiency (AUC of best-score vs eval count), not just final
  best, so the metric stays continuous even when the ceiling is reachable.

These are tracked as the next implementation step; v0 is committed as the working foundation.

---

## 2026-06-12 — flagship tier (T3): MatrixMultiplicationRank + CapSet

Two deterministic, CPU-cheap, zero-asset flagship tasks scored **uncapped** vs. best-known
(reach SoTA = 1.0, beat it > 1.0). See `docs/difficulty_and_flagship_plan.md`.

**Algorithm/MatrixMultiplicationRank** — agent emits a rank-R bilinear decomposition of the
matmul tensor; oracle verifies exactness (tensor reconstruction to 1e-7 + random integer
matrices) and returns R. Sizes 2×2×2 / 3×3×3 / 4×4×4, anchored at best-known 7 / 23 / **48**
(AlphaEvolve 2025; recursive Strassen=49 → 0.9375 on 4×4).
- Baseline (naive schoolbook): **0.0**.
- Evolve (GPT-5.5, budget 6): baseline 0 → **0.979** (iter 3). It reproduced Strassen (7),
  **Laderman's 23-mult 3×3**, and recursive Strassen (49) — all *verified exact* — but did not
  reach the 48 frontier. Headroom to ≥1.0 remains open.

**Mathematics/CapSet** — agent builds large cap sets in Z₃ⁿ (no 3 distinct collinear); oracle
verifies and returns |S|. Dims 4/5/6 anchored at the proven maxima 20/45/112; baseline = {0,1}ⁿ
(size 2ⁿ).
- Baseline ({0,1}ⁿ): **0.0**.
- Evolve (GPT-5.5, budget 6): baseline 0 → **0.657** (does NOT saturate). Per-dim: n=4 size **20**
  (= proven max, score 1.0), n=5 size **40** (vs 45, 0.615), n=6 size **81** (vs 112, 0.354), via
  product constructions of small caps. First 3 iters scored 0 (invalid/no improvement) before it
  found a working construction.

### KEY FINDING — uncapped SoTA-relative scoring on open-frontier problems resists saturation

Unlike the v0 tasks (all → 1.0), the flagship tasks land a frontier model in (0,1):
**MatrixMultiplicationRank 0.979, CapSet 0.657.** CapSet is the model unsaturable task — GPT-5.5
matches the dim-4 optimum but falls well short on dims 5–6, and beating the known maximum on
n ≥ 7 (score > 1.0) is a genuine research frontier. This validates the difficulty-ladder thesis:
keep an easy on-ramp (v0) but anchor the hard tier on problems whose best-known value is a live
frontier, scored uncapped so there is always headroom.

### Combined leaderboard (GPT-5.5, keyless evolve)

| Task | Tier | Baseline | GPT-5.5 best | Saturates? |
|---|---|---|---|---|
| ScientificComputing/PoissonSolver2D | T0 | ~0.0 | 1.000 | yes (1 iter) |
| Physics/SpinGlassGroundState | T0 | 0.146 | 1.000 | yes (1 iter) |
| Chemistry/LennardJonesCluster | T0 | 0.077 | 1.000 | yes (6 iters) |
| Algorithm/MatrixMultiplicationRank | T3 | 0.0 | **0.979** | near (48 frontier open) |
| Mathematics/CapSet | T3 | 0.0 | **0.657** | **no** (dims 5–6 + n≥7 open) |
