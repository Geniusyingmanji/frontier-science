"""Black-box eval entrypoint for SpinGlassGroundState."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

INVALID = -1e18
HERE = Path(__file__).resolve().parent
TASK_DIR = HERE.parent

def _load_callable(path: Path, name: str):
    spec = importlib.util.spec_from_file_location("fs_candidate", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return getattr(mod, name)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--metrics-out", required=True)
    args = ap.parse_args()
    metrics = {"combined_score": INVALID, "valid": 0.0}
    try:
        sys.path.insert(0, str(TASK_DIR / "verification"))
        import evaluator as oracle  # type: ignore
        solve = _load_callable(Path(args.candidate).resolve(), "solve")
        result = oracle.evaluate(solve)
        metrics.update(result); metrics["raw_score"] = result.get("combined_score")
    except Exception as exc:  # noqa: BLE001
        metrics["error_message"] = f"{type(exc).__name__}: {exc}"
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: metrics.get(k) for k in ("combined_score","valid","feasibility_rate")}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
