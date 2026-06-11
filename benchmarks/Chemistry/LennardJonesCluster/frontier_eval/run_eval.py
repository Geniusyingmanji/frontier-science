"""Black-box eval entrypoint for LennardJonesCluster.

Loads the candidate's build_cluster in a fresh namespace, scores it with the frozen oracle
in ../verification/evaluator.py, and writes metrics.json. The candidate never imports the
oracle. Run as:
    python frontier_eval/run_eval.py --candidate <file.py> --metrics-out <metrics.json>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

INVALID = -1e18
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent


def _load_callable(path: Path, name: str):
    spec = importlib.util.spec_from_file_location("fs_candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()

    metrics_out = Path(args.metrics_out)
    metrics: dict = {"combined_score": INVALID, "valid": 0.0}
    try:
        sys.path.insert(0, str(TASK_DIR / "verification"))
        import evaluator as oracle  # type: ignore

        build_cluster = _load_callable(Path(args.candidate).resolve(), "build_cluster")
        result = oracle.evaluate(build_cluster)
        metrics.update(result)
        metrics["raw_score"] = result.get("combined_score")
    except Exception as exc:  # noqa: BLE001
        metrics["error_message"] = f"{type(exc).__name__}: {exc}"

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: metrics.get(k) for k in ("combined_score", "valid", "feasibility_rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
