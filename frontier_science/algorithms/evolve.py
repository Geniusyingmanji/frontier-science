"""Generative-optimization loop (OpenEvolve-lite).

Faithful to the Frontier-Engineering paradigm: keep the best runnable program, ask the
LLM to propose an improved full rewrite of the editable file, evaluate it with the frozen
oracle, accept on strict improvement of ``combined_score`` among valid candidates. The
agent only ever sees the task text, the current best program, and the returned metrics.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..evaluate import evaluate_candidate, INVALID_SCORE
from ..llm import LLMClient
from ..spec import TaskSpec

SYSTEM_PROMPT = (
    "You are an expert computational scientist improving a Python program that solves a "
    "scientific optimization problem. You will be given the task, the current best program, "
    "and its measured metrics. Return ONE improved, complete, self-contained Python file. "
    "Keep the required entrypoint/signature and output contract intact. Optimize the reported "
    "combined_score. Respond with exactly one fenced ```python code block and nothing else."
)

_CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> Optional[str]:
    matches = _CODE_RE.findall(text or "")
    if matches:
        return max(matches, key=len).strip()
    stripped = (text or "").strip()
    # Fallback: looks like raw code (no prose) if it imports / defs early.
    if stripped and ("import " in stripped[:200] or "def " in stripped[:200]):
        return stripped
    return None


@dataclass
class EvolveResult:
    task_id: str
    best_score: float
    baseline_score: float
    best_program: str
    history: list[dict] = field(default_factory=list)
    accepted: int = 0
    evaluated: int = 0


def _build_prompt(spec: TaskSpec, program: str, metrics: dict) -> str:
    shown = {k: metrics[k] for k in (
        "combined_score", "valid", "feasibility_rate", "constraint_violations",
        "raw_score", "error_message") if k in metrics}
    return (
        f"{spec.agent_visible_text()}\n\n"
        f"## Current best program (`{spec.candidate_destination}`)\n"
        f"```python\n{program}\n```\n\n"
        f"## Its measured metrics\n```json\n{json.dumps(shown, indent=2)}\n```\n\n"
        "Improve the program to increase `combined_score`. Return one complete "
        "```python``` file implementing the same entrypoint."
    )


def evolve(
    spec: TaskSpec,
    llm: LLMClient,
    budget: int = 10,
    timeout_s: float = 300.0,
    workdir: Optional[Path] = None,
    log_fn: Callable[[str], None] = print,
) -> EvolveResult:
    workdir = Path(workdir or (spec.task_dir / "runs" / time.strftime("%Y%m%d_%H%M%S"))).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    cand_path = workdir / Path(spec.candidate_destination).name

    # Seed with the initial baseline program.
    baseline_src = spec.initial_program_path.read_text(encoding="utf-8")
    cand_path.write_text(baseline_src, encoding="utf-8")
    metrics = evaluate_candidate(spec, cand_path, timeout_s=timeout_s)
    baseline_score = float(metrics.get("combined_score", INVALID_SCORE))
    best_score, best_program, best_metrics = baseline_score, baseline_src, metrics
    log_fn(f"[{spec.task_id}] baseline combined_score={baseline_score:.6f} valid={metrics.get('valid')}")

    result = EvolveResult(spec.task_id, best_score, baseline_score, best_program)
    result.evaluated = 1
    result.history.append({"iter": 0, "score": baseline_score, "accepted": True, "metrics": best_metrics})
    trace = (workdir / "trace.jsonl").open("a", encoding="utf-8")
    trace.write(json.dumps(result.history[-1]) + "\n"); trace.flush()

    for it in range(1, budget + 1):
        try:
            reply = llm.complete(_build_prompt(spec, best_program, best_metrics), system=SYSTEM_PROMPT)
        except Exception as exc:  # noqa: BLE001
            log_fn(f"[{spec.task_id}] iter {it}: LLM error: {exc}")
            result.history.append({"iter": it, "error": str(exc)})
            continue
        code = extract_code(reply)
        if not code:
            log_fn(f"[{spec.task_id}] iter {it}: no code block parsed")
            result.history.append({"iter": it, "error": "no_code", "accepted": False})
            continue

        cand_path.write_text(code, encoding="utf-8")
        m = evaluate_candidate(spec, cand_path, timeout_s=timeout_s)
        result.evaluated += 1
        score = float(m.get("combined_score", INVALID_SCORE))
        valid = float(m.get("valid", 0.0)) >= 1.0 or score > INVALID_SCORE / 2
        accepted = bool(valid and score > best_score)
        if accepted:
            best_score, best_program, best_metrics = score, code, m
            result.best_score, result.best_program = best_score, best_program
            result.accepted += 1
        entry = {"iter": it, "score": score, "best": best_score, "accepted": accepted,
                 "metrics": {k: m.get(k) for k in ("combined_score", "valid", "raw_score", "error_message")}}
        result.history.append(entry)
        trace.write(json.dumps(entry) + "\n"); trace.flush()
        log_fn(f"[{spec.task_id}] iter {it}: score={score:.6f} best={best_score:.6f} "
               f"{'ACCEPT' if accepted else 'reject'}")

    trace.close()
    (workdir / "best_program.py").write_text(best_program, encoding="utf-8")
    (workdir / "summary.json").write_text(json.dumps({
        "task_id": spec.task_id, "baseline_score": baseline_score, "best_score": best_score,
        "evaluated": result.evaluated, "accepted": result.accepted, "budget": budget,
    }, indent=2), encoding="utf-8")
    log_fn(f"[{spec.task_id}] DONE baseline={baseline_score:.6f} -> best={best_score:.6f} "
           f"({result.accepted}/{result.evaluated-1} accepted)  out={workdir}")
    return result
