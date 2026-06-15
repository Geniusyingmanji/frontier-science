# WaveguideDesign — Optimize waveguide geometry for mode confinement

## Scientific background
Optimize waveguide geometry for mode confinement. This is a well-studied problem with documented baselines and reference solutions.

## Your task
```python
def design_waveguide(*args):
    """Implement an optimized solution. See evaluator for the exact input/output contract."""
```

The evaluator provides the problem instance, scores your solution against a reference, and
returns a combined_score in [0,1].

## Rules
- Only edit `solution.py`. Keep the `design_waveguide()` signature.
- numpy/scipy only. CPU, seconds. No network. Do not read `verification/` or `frontier_eval/`.
