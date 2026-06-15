# Frontier-Science

A multi-discipline benchmark of **open-ended scientific optimization** problems. An agent
iteratively rewrites one runnable program; each edit is scored by a **frozen, deterministic
oracle** (simulator / analytic solution / dataset) that returns a **continuous, improvable**
`combined_score` — not a 0/1 answer. We measure the best solution found within a budget.

It is the science counterpart of
[Frontier-Engineering](https://github.com/EinsiaLab/Frontier-Engineering): same
generative-optimization paradigm and black-box per-task contract, applied to scientific
discovery. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add new tasks.

## Tasks

| Task | Domain | Difficulty | Oracle | Metric |
|---|---|---|---|---|
| `Chemistry/LennardJonesCluster` | molecular mechanics | medium | LJ energy vs Cambridge Cluster DB | mean gap-closed |
| `Physics/SpinGlassGroundState` | statistical mechanics | medium | exact SK ground state | mean gap-closed |
| `ScientificComputing/PoissonSolver2D` | scientific computing | medium | analytic-solution L2 error | log-scaled error reduction |
| `Combinatorics/GraphMaxCut` | combinatorial optimization | hard | exact max-cut (weighted) | mean gap-closed |
| `Biology/ProteinLatticeHP` | biophysics | hard | HP lattice energy (2D) | mean fraction of optimal contacts |
| `Optimization/CirclePacking` | computational geometry | hard | valid packing side length | mean gap from grid to Packomania best |
| `Algorithm/MatrixMultiplicationRank` | algebraic complexity | **flagship** | exact tensor decomposition | uncapped, SoTA-relative (AlphaEvolve) |
| `Mathematics/CapSet` | extremal combinatorics | **flagship** | cap-set verification | uncapped, SoTA-relative (FunSearch) |

All oracles are pure `numpy`/`scipy`, CPU-only, with documented reference values. Flagship
tasks use **uncapped scoring** — reaching published SoTA = 1.0, beating it > 1.0.

## Quickstart

```bash
python -m frontier_science list                       # discover tasks
python -m frontier_science eval --task LennardJonesCluster   # score the baseline program
python -m frontier_science run  --task LennardJonesCluster --budget 10   # evolve with an LLM
python -m frontier_science smoke                      # check the configured LLM endpoint
```

## LLM configuration

The runner uses any **OpenAI-compatible** endpoint. Copy the example and fill in your own:

```bash
cp frontier_science/conf/llm/openai_compatible.example.yaml frontier_science/conf/llm/local.yaml
# edit base_url / api_key / model, or export OPENAI_API_KEY
```

`conf/llm/local.yaml` is git-ignored. Resolution order:
`--llm-config` / `FS_LLM_CONFIG` → `conf/llm/local.yaml` → the committed example.

## Adding a task

Create `benchmarks/<Domain>/<Task>/` with a `frontier_eval/` contract (`metadata.yaml`,
`initial_program.txt`, `candidate_destination.txt`, `eval_command.txt`, `constraints.txt`,
`run_eval.py`), an editable baseline program, and a hidden `verification/` oracle. No harness
change is needed — the task is auto-discovered. Use an existing task as a template.

## Contract (black-box)

The agent sees only `Task.md`, the editable program, `constraints.txt`, and the returned
`metrics.json`. It never sees `verification/` (the oracle) or the eval internals. Each
candidate is run in a subprocess; `combined_score` ranges from 0 (baseline) to 1 (SoTA)
for clipped tasks, and is uncapped (>1 = beat SoTA) for flagship tasks. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the full spec.
