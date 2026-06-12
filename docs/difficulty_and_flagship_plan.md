# Frontier-Science — Difficulty Ladder & Flagship Tasks (plan)

> Why this doc: the v0 tasks (LJ / SpinGlass / Poisson) were all driven to `combined_score ≈ 1.0`
> by GPT-5.5 in 1–6 iterations, because each has a **known, reachable optimum** that a textbook
> algorithm attains. To be a real benchmark we need a difficulty ladder that keeps an easy
> on-ramp **and** tops out in problems that are genuinely hard to grind but carry AlphaFold-tier
> real-world impact. This plans that ladder and the scoring changes it requires.

---

## 1. The core fix: where "hard to grind" comes from

A task is saturable iff its optimum is **known and reachable within budget**. So the hard tier
must use problems whose **best-known value is a live research frontier** (unknown true optimum),
and must be scored so that improvement never bottoms out:

- **Uncapped, SoTA-relative score.** Define `progress = (score − baseline) / (sota_ref − baseline)`.
  Onramp tasks **clip to [0,1]** (calibration). Frontier tasks **do NOT clip above 1**: reaching
  published SoTA = 1.0, and *beating* it (the actual research goal) shows as >1.0. There is always
  headroom, so a frontier model cannot "finish" the task.
- **Sample-efficiency, not just final best.** Report the AUC of best-score vs. eval count, so the
  metric stays continuous even when a ceiling is eventually reachable.
- **Multi-fidelity.** A fast proxy oracle for iteration + an exact oracle for the final score.
- **Feasibility accounting.** `feasibility_rate` / `constraint_violations` for physically- or
  mathematically-constrained designs.
- **Anti-memorization (critical for famous problems).** Famous best-known solutions (e.g.
  AlphaEvolve's 48-multiplication 4×4 scheme) may sit in the model's training data. Defenses:
  (a) score only *improvement beyond* the published best; (b) use held-out instances/sizes whose
  best solution is **not** published; (c) require a verified artifact (a decomposition, a config),
  not a citation — the frozen oracle only credits what it can verify.

---

## 2. The difficulty ladder

| Tier | Intent | Optimum | Scoring | Example tasks |
|---|---|---|---|---|
| **T0 Onramp (Easy)** | calibration / pipeline smoke | known, reachable | clip [0,1] | v0: LJ N≤19, SpinGlass N≤20, Poisson |
| **T1 Medium** | needs real domain strategy; SoTA known but not trivially hit | known SoTA gap | clip [0,1] | LJ N=38/75/98 (capped time); larger spin glass; TDC molecular optimization; RNA design (Eterna100) |
| **T2 Hard** | real models as oracles; improvement genuinely hard | unknown / weak ceiling | uncapped SoTA-relative | protein ddG (Tsuboyama); MatBench-Discovery stability; Open-Catalyst adsorption |
| **T3 Flagship (Grand Challenge)** | AlphaFold / AlphaEvolve-tier; beating SoTA = a real result | unknown, moving frontier | uncapped, + sample-efficiency | matrix-mult tensor rank; cap set; de-novo binder design; stellarator design |

T0/T1 are the "简单题" on-ramp; T2/T3 are the "很难刷但有现实意义" core. The leaderboard's headline
is Average Rank across tasks, with per-tier and per-domain breakdowns.

---

## 3. Flagship (T3) tasks — grounded in 2025–2026 SoTA

Two families: **(A) deterministic, CPU-cheap, AlphaEvolve-lineage** (cheapest to build, zero
assets, impossible to fully solve) and **(B) GPU-oracle, AlphaFold-lineage / real-world** (heavier,
maximum impact). Build A first for impact-per-effort, then B.

### A1 — `Algorithm/MatrixMultiplicationRank`  ★ recommended first flagship
- **Problem**: find a bilinear algorithm multiplying two `n×n` matrices in as few scalar
  multiplications as possible (a low-rank decomposition of the matmul tensor).
- **Oracle (deterministic, CPU-µs–ms)**: the candidate emits a rank-`R` tensor decomposition
  `(U, V, W)`; the oracle (i) verifies exactness symbolically/over random integer & finite-field
  matrices that `Σ_r (Uᵣ·a)(Vᵣ·b) Wᵣ == a·b`, (ii) returns `R` (the multiplication count).
  Score = `(R_baseline − R) / (R_baseline − R_sota)`, uncapped.
- **Frontier / SoTA**: Strassen 4×4 = 49 (1969, unbeaten 56 yrs) → **AlphaEvolve 48 (2025)**, 47 over
  GF(2); 5×5 = 96. **Open at 46–47 for 4×4**, and many `m×n×p` shapes are wide open.
- **Why unsaturable**: tensor-rank minimization is NP-hard; exponentially many inequivalent
  algorithms; lower bounds for 4×4 unknown. Beating the best-known is a publishable result.
- **Anti-memorization**: credit only verified decompositions; headline on **non-published shapes**
  (e.g. `3×4×5`, `4×4×5`) so recalling the famous 48 doesn't win.
- **Assets/compute**: none; pure numpy + a finite-field check. Reference: DeepMind `alphatensor`
  repo (verified solutions + nonequivalence notebooks), AlphaEvolve (2025).

### A2 — `Mathematics/CapSet`
- **Problem**: largest subset of `ℤ₃ⁿ` with no 3 elements summing to 0 (mod 3).
- **Oracle (deterministic, CPU-seconds)**: verify the no-3-term-AP property by triple/scan check;
  return `|S|`. Score uncapped vs. best-known `|S|`.
- **Frontier**: FunSearch raised dim-8 to **512** and the asymptotic lower bound to `2.2203ⁿ`
  (2023); **dim 10–11 exact sizes are open**. Reference: DeepMind `funsearch` (`cap_set.ipynb`).
- **Why unsaturable**: constructions are an open combinatorial frontier; verifier is trivial but
  *finding* large sets is not.

> A3 (stretch): `Geometry/KissingNumber` (AlphaEvolve raised dim-11 to 593; dim-13 to 1154 via
> PackingStar 2025). Needs rational-coordinate handling for a clean deterministic verifier — tier-3b.

### B1 — `ComputationalBiology/DeNovoBinderDesign`  ★ the AlphaFold-tier flagship
- **Problem**: design a protein sequence/backbone that binds a given target epitope.
- **Oracle (GPU, ~30–60 GPU-s/eval)**: fold the designed binder–target complex with an open model
  (AlphaFold2-Multimer or **Boltz-2**, MIT) and score the **interface** (ipTM / pAE-interface /
  ipSAE; Boltz-2 also predicts affinity). Frozen weights, deterministic given sequence.
- **Frontier / SoTA**: BindCraft (Nature 2025), **BoltzGen ~66% success**, AlphaProteo (DeepMind
  2024, 3–300× affinity). Adaptyv-Bio runs live binder competitions scored by exactly these metrics.
- **Why unsaturable**: interface confidence on novel complexes is far from 1; many hard targets
  (e.g. TNFα) resist even SoTA; sequence space is ~10¹⁰⁰. Direct drug-discovery impact.
- **Multi-fidelity**: ESMFold/pLDDT fast filter → AF2-Multimer/Boltz-2 exact. Test set = a fixed
  panel of diverse targets (Adaptyv targets + custom hard epitopes); baseline = BindCraft default.
- **Assets/compute**: AF2/Boltz weights (single 24–40GB GPU). Heaviest task; highest prestige.

### B2 — `ComputationalBiology/ProteinStabilityDDG`  (cheaper bio flagship)
- **Problem**: mutate a protein to maximize folding stability (ΔΔG).
- **Oracle (GPU, ~1–5 GPU-s/eval)**: ESMFold structure + a ddG predictor trained on the
  **Tsuboyama 2023 mega-scale** set (272k mutations, 298 domains); held-out proteins for test.
  Continuous kcal/mol; frontier predictor Spearman ≈ 0.7 (not saturated).
- **Why a good T2/T3 bridge**: open oracle, light GPU, real (therapeutics/enzyme engineering),
  combinatorial epistasis keeps it hard.

### B3 — `MaterialsScience/StableCrystalDiscovery`  (CPU-friendly, real impact)
- **Problem**: propose a crystal (composition + structure) that is thermodynamically stable.
- **Oracle (CPU 5–50 ms/structure, or 1 GPU)**: a universal MLIP (MACE-MP-0 / EquiformerV2 via
  `fairchem`) gives formation energy → **energy above the convex hull**; lower is more stable.
  Score uncapped vs. a strong reference; feasibility = on-hull fraction.
- **Frontier**: MatBench-Discovery leaderboard; GNoME found 2.2M stable structures, **736
  experimentally synthesized** — real discovery, effectively infinite search space.
- **Why a strong pick**: CPU-runnable, mature open oracle, climate/energy/battery impact.

> B4 (physics-prestige, slower): `Fusion/StellaratorOptimization` — SIMSOPT/DESC quasi-symmetry +
> coil-complexity objectives; ConStellaration benchmark (2025, 158k configs). Deterministic physics,
> CPU-minutes/eval, unknown optimum, fusion-energy impact. Tier-3b once budgeting is in place.

---

## 4. Feasibility & build order

| Task | Tier | Oracle | Assets | Compute/eval | Build effort | Impact |
|---|---|---|---|---|---|---|
| MatrixMultiplicationRank | T3 | exact verifier | none | CPU ms | **low** | very high (AlphaEvolve) |
| CapSet | T3 | exact verifier | none | CPU sec | low | high (FunSearch) |
| LJ N=38/75/98, larger SpinGlass | T1 | analytic | none | CPU sec–min | low | medium (calibration) |
| StableCrystalDiscovery | T3 | MLIP (fairchem) | MLIP weights | CPU 10–50 ms | medium | very high (GNoME) |
| ProteinStabilityDDG | T2/T3 | ESMFold + ddG | ESMFold + Tsuboyama | 1–5 GPU-s | medium | high |
| DeNovoBinderDesign | T3 | AF2-Multimer / Boltz-2 | AF2/Boltz weights | 30–60 GPU-s | high | very high (AlphaFold) |
| StellaratorOptimization | T3b | SIMSOPT/DESC | physics codes | CPU min | high | very high (fusion) |

**Recommended order**
1. **MatrixMultiplicationRank + CapSet** — deterministic, zero assets, CPU-cheap, genuinely
   unsaturable, AlphaEvolve-tier. Highest impact-per-effort; ship next.
2. **Harden T1** (LJ N=38/75/98 with wall-clock caps; SpinGlass N=40–80 with reference-relative
   uncapped scoring) — converts the saturated v0 tasks into a real medium tier.
3. **StableCrystalDiscovery** — first MLIP oracle; CPU-friendly; real-world flagship.
4. **ProteinStabilityDDG → DeNovoBinderDesign** — the AlphaFold-lineage flagships, once a GPU box
   and model weights are wired up (multi-fidelity to control cost).

## 5. Scoring-framework changes to implement alongside

- Add `score_mode: clipped | uncapped` to `metadata.yaml`; evaluator returns `progress` (uncapped
  for T3) plus raw objective and `sota_ref` / `baseline_ref`.
- Add `fast_eval_command` (multi-fidelity) and `eval_budget` (sample-efficiency AUC) to the contract.
- Leaderboard: Average Rank (headline) + per-tier means + Feasibility board + an **"above-SoTA"**
  board listing tasks where any run scored > 1.0 (i.e. beat the published best).
- Every flagship task ships a `references/known_best.md` recording the current best-known value,
  its source, and the date — so the normalization ceiling is auditable and updatable.

## 6. Open questions / risks

- **GPU oracles (binder, ddG)**: need a box with AF2/Boltz/ESMFold weights; start with ddG (light)
  before binder design. Multi-fidelity is mandatory to keep eval cost sane.
- **MLIP licensing/weights** for crystal discovery: confirm `fairchem` model license for redistribution.
- **Citation accuracy**: several SoTA numbers above come from a fast 2026 survey; verify each
  best-known value and arXiv id at task-build time and record it in `references/known_best.md`.
- **Determinism on GPU**: fix seeds / disable nondeterministic kernels so the oracle is reproducible.
