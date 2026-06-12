# Frontier-Science — Goal & Status

## Goal

Build **Frontier-Science**: a multi-discipline benchmark of **open-ended scientific
optimization** problems where the metric is a *continuous, improvable* score produced by a
**frozen, deterministic oracle** (physics/biochem simulators, analytic solutions, datasets) —
not a 0/1 answer. An agent iteratively rewrites one runnable program; each edit is scored;
we measure the best solution found within a budget. See [`plan.md`](plan.md) for the full design.

Initial craft targets the Frontier-Engineering paper's structure: a black-box per-task
contract, a generative-optimization loop, continuous `combined_score`, and a normalized
cross-task leaderboard.

## Design invariants

- **Black-box contract per task** (mirrors Frontier-Engineering):
  agent sees `Task.md` + the editable baseline program + `constraints.txt` + returned
  `metrics.json` only. It never sees the oracle (`verification/`) or eval internals.
- **Frozen deterministic oracle**: same input → same score, no LLM judge, no network.
- **Continuous reward** `combined_score ∈ [0,1]` after per-task normalization
  `(score - baseline) / (reference_sota - baseline)`, clipped.
- **Trajectory > single**: best score within an eval budget.

## LLM access (IMPORTANT — secret hygiene)

- The repo commits **only a generic OpenAI-compatible interface**:
  `frontier_science/conf/llm/openai_compatible.example.yaml` with `base_url`, `api_key`, `model`.
- Our **local keyless GPT-5.5** endpoint is configured in `conf/llm/local.yaml`, which is
  **git-ignored** and never pushed. For our own testing we point the client at the local
  Azure managed-identity proxy (Responses API). Nothing about that endpoint is committed.

## Seed task set (v0, dependency-light: numpy/scipy only)

| Task | Domain | Oracle | Metric |
|---|---|---|---|
| LennardJonesCluster | Chemistry / molecular mechanics | LJ potential energy (analytic) | normalized energy vs known global minimum |
| ABProteinFolding | Computational biology / biophysics | Stillinger AB off-lattice energy | normalized energy vs literature minimum |
| SpinGlassGroundState | Physics / statistical mechanics | SK Ising energy (fixed instance) | normalized energy vs reference |
| PoissonSolver2D | Scientific computing | analytic-solution L2 error × FLOP proxy | accuracy/cost vs baseline |

All four are genuine continuous-optimization problems with documented reference values.
The harness adds tasks with **no harness code change** (new benchmark dir + entry).

## Difficulty ladder & flagship tasks

v0 tasks saturate to ~1.0 for a frontier model. The next phase adds a difficulty ladder
(easy on-ramp → AlphaFold/AlphaEvolve-tier flagships scored uncapped vs. SoTA). See
[`docs/difficulty_and_flagship_plan.md`](docs/difficulty_and_flagship_plan.md). First flagships
to build: `Algorithm/MatrixMultiplicationRank` and `Mathematics/CapSet` (deterministic, CPU-cheap,
zero-asset, unsaturable).

## Status log

See [`experiments/LOG.md`](experiments/LOG.md) for the running experiment record.
