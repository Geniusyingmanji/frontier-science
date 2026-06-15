# AlloyHardnessOptimization — design a high-hardness alloy composition

## Scientific background

High-entropy alloys (HEAs) contain 5+ principal elements in near-equimolar ratios. Their
vast compositional space offers exceptional mechanical properties (hardness, strength).
Given a surrogate model mapping composition to Vickers hardness (trained on experimental
data), optimize the composition to maximize hardness while maintaining thermodynamic
stability (mixing entropy > R ln 5).

## Your task

```python
def design_alloy(n_elements, element_names, predict_hardness, n_candidates):
    """Return (n_candidates, n_elements) array of compositions (fractions summing to 1).
    predict_hardness(compositions) returns hardness for each."""
```

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
