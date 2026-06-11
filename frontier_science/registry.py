"""Discover tasks under ``benchmarks/`` (any dir containing ``frontier_eval/``)."""

from __future__ import annotations

from pathlib import Path

from .spec import TaskSpec, load_task_spec

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS = REPO_ROOT / "benchmarks"


def discover_task_dirs() -> list[Path]:
    if not BENCHMARKS.is_dir():
        return []
    return sorted(p.parent for p in BENCHMARKS.glob("*/*/frontier_eval") if p.is_dir())


def list_tasks() -> list[TaskSpec]:
    return [load_task_spec(d) for d in discover_task_dirs()]


def find_task(name: str) -> TaskSpec:
    """Match by full id (Domain/Task) or by task name (case-insensitive)."""
    name_l = name.lower().strip("/")
    for spec in list_tasks():
        if spec.task_id.lower() == name_l or spec.task_dir.name.lower() == name_l:
            return spec
    avail = ", ".join(s.task_id for s in list_tasks()) or "(none)"
    raise KeyError(f"Unknown task '{name}'. Available: {avail}")
