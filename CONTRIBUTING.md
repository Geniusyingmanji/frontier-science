# Contributing to Frontier-Science

We need the community to expand this benchmark across scientific disciplines. We welcome
new optimization tasks via Pull Requests. If you'd like to contribute, follow the standards
and process below.

> **AI-assisted contributions are welcome.** However, please verify all oracle code and
> reference values yourself — do not leave scientific correctness entirely to AI.

---

## Task requirements

Every Frontier-Science task must satisfy **all five** of these:

1. **Continuous, improvable metric.** The oracle returns a numeric `combined_score` that can be
   meaningfully improved — not a binary pass/fail.
2. **Deterministic, frozen oracle.** Same candidate program → same score. No LLM judge, no
   network, no randomness without a fixed seed.
3. **Locally runnable.** Easy/Medium tasks: CPU, < 2 min. Hard/Flagship tasks may use GPU but
   must finish a single evaluation within a few minutes.
4. **Black-box safety.** The agent must not be able to read the oracle code, the test-split
   answers, or the verification internals. The oracle runs in a subprocess on a copy of the
   candidate file.
5. **Scientific significance.** Improvement corresponds to real scientific value (a better
   algorithm, a better molecular design, a lower energy, ...). Provide a citable reference for
   the baseline and the best-known result.

---

## Task directory layout

Each task lives at `benchmarks/<Domain>/<Task>/` and is **auto-discovered** by the harness —
no harness code change is needed.

```
benchmarks/
└── <Domain>/                         # e.g. Chemistry, Physics, Algorithm, Mathematics
    └── <Task>/                       # e.g. LennardJonesCluster, CapSet
        ├── Task.md                   # [Required] Agent-visible task description
        ├── solution.py               # [Required] Weak-but-valid baseline program
        ├── frontier_eval/            # [Required] Black-box evaluation contract
        │   ├── metadata.yaml         # Task metadata (see below)
        │   ├── initial_program.txt   # Points to the baseline file (e.g. "solution.py")
        │   ├── candidate_destination.txt  # File the agent edits (e.g. "solution.py")
        │   ├── eval_command.txt      # Eval command template (see below)
        │   ├── constraints.txt       # Natural-language constraints shown to the agent
        │   ├── agent_files.txt       # Files the agent is allowed to see
        │   ├── readonly_files.txt    # Files the agent must not modify
        │   └── run_eval.py           # Subprocess entry: loads candidate, calls oracle
        ├── verification/             # [Required] Hidden oracle — agent CANNOT see this
        │   └── evaluator.py          # The frozen scoring function
        └── references/               # [Optional] Data, configs, known-best records
            └── known_best.md         # Best-known values + sources (for flagship tasks)
```

### `frontier_eval/metadata.yaml`

```yaml
domain: Chemistry                    # top-level domain directory name
task: LennardJonesCluster            # task directory name
difficulty: medium                   # easy | medium | hard | flagship
tier: T1                             # T0 (on-ramp) | T1 | T2 | T3 (flagship)
oracle_type: analytical              # analytical | physical_sim | dataset_oracle | neural_surrogate
score_mode: clipped                  # clipped (cap at [0,1]) | uncapped (SoTA-relative, >1 = beat SoTA)
gpu_required: false
eval_time_seconds: 5                 # approx wall-clock for one evaluation
science_metric: <name>               # human-readable name of the primary metric
reference_baseline: <description>    # what the initial program does
reference_sota: <description>        # best-known result and its source
citation: "Author, Journal, Year"    # citable reference(s)
```

### `frontier_eval/eval_command.txt`

Use this exact template (the harness substitutes `{python}`, `{candidate}`, `{metrics}`):

```
{python} frontier_eval/run_eval.py --candidate {candidate} --metrics-out {metrics}
```

### `frontier_eval/run_eval.py` (template)

```python
"""Black-box eval entrypoint for <YourTask>."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

INVALID = -1e18
TASK_DIR = Path(__file__).resolve().parent.parent

def _load_callable(path, name):
    spec = importlib.util.spec_from_file_location("fs_candidate", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    metrics = {"combined_score": INVALID, "valid": 0.0}
    try:
        sys.path.insert(0, str(TASK_DIR / "verification"))
        import evaluator as oracle
        entry = _load_callable(Path(args.candidate).resolve(), "YOUR_ENTRYPOINT")
        result = oracle.evaluate(entry)
        metrics.update(result); metrics["raw_score"] = result.get("combined_score")
    except Exception as exc:
        metrics["error_message"] = f"{type(exc).__name__}: {exc}"
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps({k: metrics.get(k) for k in ("combined_score", "valid")}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

Replace `YOUR_ENTRYPOINT` with the function name the agent must implement (e.g. `build_cluster`,
`solve`, `build_capset`).

### `verification/evaluator.py` contract

Your oracle must define an `evaluate(candidate_callable)` function that returns a dict with
**at least**:

```python
{
    "combined_score": float,   # the primary metric (higher is better; -1e18 on failure)
    "valid": float,            # 1.0 if the candidate produced a legal result, else 0.0
}
```

Optional fields: `feasibility_rate`, `constraint_violations`, `beat_sota`, `per_size`,
`raw_score`, etc.

---

## Scoring modes

| Mode | When to use | Score range |
|---|---|---|
| `clipped` | The optimum is known and reachable (easy/medium tasks) | `[0, 1]` |
| `uncapped` | The best-known value is a live research frontier (flagship tasks) | `[0, ∞)` — reaching SoTA = 1.0, beating it > 1.0 |

For `uncapped` tasks, also provide `references/known_best.md` documenting the current
best-known value, its source, and the date.

---

## Baseline program (`solution.py`)

- Must be **weak but valid**: it runs, the oracle accepts it, and it scores near 0.
- Keep the function signature and output contract that the oracle expects.
- Use only `numpy` and `scipy` (for CPU tasks). Document any extra dependencies in a
  `verification/requirements.txt`.

---

## Checklist before submitting a PR

- [ ] `python -m frontier_science eval --task <Domain>/<Task>` runs and returns a valid
      `metrics.json` with `combined_score` near 0 for the baseline.
- [ ] `python -m frontier_science list` shows the new task with correct domain/difficulty.
- [ ] The oracle is deterministic (run twice, get the same score).
- [ ] The agent files (`Task.md`, `solution.py`, `constraints.txt`) do not leak the oracle
      implementation or the answer.
- [ ] No absolute paths, no `.env` files, no API keys, no `__pycache__`, no large data files.
- [ ] `metadata.yaml` is complete (all fields filled in).
- [ ] For flagship (`uncapped`) tasks: `references/known_best.md` exists with sourced values.

---

## Contribution process

1. **Fork** this repository and **clone** your fork.
2. **Create a branch**: `feat/<Domain>/<Task>` (e.g. `feat/Biology/RNAInverseFolding`).
3. **Add your task** following the directory layout above. Use an existing task (e.g.
   `Chemistry/LennardJonesCluster` for clipped, `Mathematics/CapSet` for uncapped) as a
   template.
4. **Test locally**:
   ```bash
   python -m frontier_science eval --task <Domain>/<Task>        # baseline score
   python -m frontier_science run  --task <Domain>/<Task> --budget 3  # quick evolve smoke
   ```
5. **Submit a Pull Request** to `main`. In the PR description, include:
   - Scientific background (1–2 sentences).
   - Oracle details (what it computes, dependencies, compute cost).
   - Baseline score and reference SoTA.
6. **Review**: maintainers will check oracle correctness, black-box safety, and scoring
   calibration before merging.

---

## LLM configuration (for testing)

The harness uses any OpenAI-compatible endpoint. Copy the example config and fill in your own:

```bash
cp frontier_science/conf/llm/openai_compatible.example.yaml frontier_science/conf/llm/local.yaml
# edit base_url / api_key / model
```

`local.yaml` is git-ignored. Never commit API keys.

---

> Questions? Open an Issue to discuss your task idea before writing code.
