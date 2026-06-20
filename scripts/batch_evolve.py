"""Batch evolve: run GPT-5.5 evolve on all tasks with a small budget, save results."""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frontier_science.registry import list_tasks
from frontier_science.config import load_llm_client
from frontier_science.algorithms.evolve import evolve

BUDGET = 3
TIMEOUT = 120.0
OUTFILE = Path(__file__).resolve().parent.parent / "experiments" / "batch_evolve_results.json"

def main():
    llm = load_llm_client()
    specs = list_tasks()
    print(f"Running evolve budget={BUDGET} on {len(specs)} tasks with model={llm.config.model}")
    results = []
    for i, spec in enumerate(specs):
        print(f"\n[{i+1}/{len(specs)}] {spec.task_id}...", flush=True)
        t0 = time.time()
        try:
            res = evolve(spec, llm, budget=BUDGET, timeout_s=TIMEOUT, log_fn=lambda s: print(f"  {s}"))
            entry = {
                "task": res.task_id,
                "baseline": res.baseline_score,
                "best": res.best_score,
                "accepted": res.accepted,
                "evaluated": res.evaluated,
                "wall_seconds": round(time.time() - t0, 1),
            }
        except Exception as e:
            entry = {"task": spec.task_id, "error": str(e), "wall_seconds": round(time.time() - t0, 1)}
        results.append(entry)
        print(f"  -> best={entry.get('best', 'ERR')} ({entry['wall_seconds']}s)")
        # Save incrementally
        OUTFILE.parent.mkdir(parents=True, exist_ok=True)
        OUTFILE.write_text(json.dumps(results, indent=2))
    # Summary
    scores = [r["best"] for r in results if "best" in r and r["best"] > -1e10]
    print(f"\n{'='*60}")
    print(f"DONE: {len(results)} tasks, {len(scores)} valid scores")
    print(f"Mean best: {sum(scores)/len(scores):.4f}" if scores else "no valid scores")
    print(f"Results: {OUTFILE}")

if __name__ == "__main__":
    main()
