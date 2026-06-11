"""Frontier-Science CLI.

  python -m frontier_science list
  python -m frontier_science eval  --task LennardJonesCluster [--candidate path.py]
  python -m frontier_science run   --task LennardJonesCluster --budget 10 [--llm-config p.yaml]
  python -m frontier_science smoke  # check the configured LLM endpoint responds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .algorithms.evolve import evolve
from .config import load_llm_client, resolve_llm_config_path
from .evaluate import evaluate_candidate
from .registry import find_task, list_tasks


def _cmd_list(args) -> int:
    specs = list_tasks()
    if not specs:
        print("No tasks found under benchmarks/.")
        return 0
    print(f"{'TASK':45} {'DOMAIN':22} {'DIFF':8} ORACLE")
    for s in specs:
        print(f"{s.task_id:45} {s.domain:22} {s.difficulty:8} {s.metadata.get('oracle_type','-')}")
    return 0


def _cmd_eval(args) -> int:
    spec = find_task(args.task)
    cand = Path(args.candidate).resolve() if args.candidate else spec.initial_program_path
    metrics = evaluate_candidate(spec, cand, timeout_s=args.timeout)
    print(json.dumps(metrics, indent=2))
    return 0


def _cmd_run(args) -> int:
    spec = find_task(args.task)
    llm = load_llm_client(args.llm_config)
    print(f"LLM config: {resolve_llm_config_path(args.llm_config)} "
          f"(wire={llm.config.wire}, model={llm.config.model})", file=sys.stderr)
    res = evolve(spec, llm, budget=args.budget, timeout_s=args.timeout)
    print(json.dumps({"task": res.task_id, "baseline": res.baseline_score,
                      "best": res.best_score, "accepted": res.accepted,
                      "evaluated": res.evaluated}, indent=2))
    return 0


def _cmd_smoke(args) -> int:
    llm = load_llm_client(args.llm_config)
    print(f"Using {resolve_llm_config_path(args.llm_config)} "
          f"(wire={llm.config.wire}, model={llm.config.model})", file=sys.stderr)
    out = llm.complete("Reply with exactly: FS_SMOKE_OK", system="Be terse.")
    print(out.strip())
    return 0 if "FS_SMOKE_OK" in out else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="frontier_science")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(fn=_cmd_list)

    pe = sub.add_parser("eval"); pe.set_defaults(fn=_cmd_eval)
    pe.add_argument("--task", required=True); pe.add_argument("--candidate", default=None)
    pe.add_argument("--timeout", type=float, default=300.0)

    pr = sub.add_parser("run"); pr.set_defaults(fn=_cmd_run)
    pr.add_argument("--task", required=True); pr.add_argument("--budget", type=int, default=10)
    pr.add_argument("--timeout", type=float, default=300.0); pr.add_argument("--llm-config", default=None)

    ps = sub.add_parser("smoke"); ps.set_defaults(fn=_cmd_smoke)
    ps.add_argument("--llm-config", default=None)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
