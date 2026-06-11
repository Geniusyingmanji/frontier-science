"""Black-box candidate evaluation.

Runs the task's ``frontier_eval/run_eval.py`` in a subprocess on a candidate program
and returns the parsed ``metrics.json``. The agent never imports the oracle directly;
it only ever sees the metrics dict returned here.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .spec import TaskSpec

INVALID_SCORE = -1e18


def evaluate_candidate(spec: TaskSpec, candidate_path: Path, timeout_s: float = 300.0) -> dict[str, Any]:
    candidate_path = Path(candidate_path).resolve()
    with tempfile.TemporaryDirectory(prefix="fs_eval_") as tmp:
        metrics_out = Path(tmp) / "metrics.json"
        cmd_tmpl = spec.eval_command or (
            "{python} frontier_eval/run_eval.py --candidate {candidate} --metrics-out {metrics}"
        )
        cmd = cmd_tmpl.format(
            python=shlex.quote(sys.executable),
            candidate=shlex.quote(str(candidate_path)),
            metrics=shlex.quote(str(metrics_out)),
        )
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(spec.task_dir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"combined_score": INVALID_SCORE, "valid": 0.0, "timeout": 1.0,
                    "error_message": f"eval timeout > {timeout_s}s"}

        if metrics_out.is_file():
            try:
                metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                return {"combined_score": INVALID_SCORE, "valid": 0.0,
                        "error_message": f"bad metrics.json: {exc}"}
            metrics.setdefault("combined_score", INVALID_SCORE)
            return metrics

        return {"combined_score": INVALID_SCORE, "valid": 0.0,
                "error_message": "no metrics.json produced",
                "stderr": (proc.stderr or "")[-2000:]}
