# DoseResponseFitting — Fit Hill equation to dose-response data

## Scientific background
Fit Hill equation to dose-response data. This is a well-studied problem with documented baselines and reference solutions.

## Your task
```python
def fit_dose_response(*args):
    """Implement an optimized solution. See evaluator for the exact input/output contract."""
```

The evaluator provides the problem instance, scores your solution against a reference, and
returns a combined_score in [0,1].

## Rules
- Only edit `solution.py`. Keep the `fit_dose_response()` signature.
- numpy/scipy only. CPU, seconds. No network. Do not read `verification/` or `frontier_eval/`.
