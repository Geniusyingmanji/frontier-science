# CapSet — find large cap sets in Z_3^n

## Scientific background

A **cap set** is a subset of `Z_3^n` (vectors mod 3) containing no three distinct points on a
line — equivalently, no three distinct `x, y, z` with `x + y + z ≡ 0 (mod 3)`. The maximum
cap-set size is a famous open problem in extremal combinatorics, tied to the cap-set capacity
and to fast matrix multiplication. In 2023, DeepMind's **FunSearch** discovered larger cap sets
than any previously known construction (e.g. size 512 in dimension 8), improving a 20-year-old
asymptotic lower bound.

## Your task

Edit **`solution.py`** so it defines:

```python
def build_capset(n: int) -> list:
    """Return a list of vectors in {0,1,2}^n forming a cap set (no 3 distinct collinear)."""
```

The evaluator calls it for several dimensions, **verifies the cap property**, and reads off the
size `|S|`. Bigger valid caps score higher.

## Scoring

For each dimension, with `baseline = 2^n` (the trivial `{0,1}^n` cap) and `sota` the best known
maximum:

```
score(n) = max(0, (|S| − 2^n) / (sota − 2^n))      # UNCAPPED above
```

So the `{0,1}^n` baseline scores 0, matching the known maximum scores 1.0, and **exceeding it
scores above 1.0**. `combined_score` is the mean over dimensions. Dimensions evaluated: n=4
(max 20), n=5 (max 45), n=6 (max 112). An invalid set (any collinear triple) scores 0 for that
dimension.

## Rules

- Only edit `solution.py`; keep the `build_capset(n)` signature and list-of-vectors output.
- Vectors must have entries in `{0,1,2}` and length `n`; duplicates are de-duplicated.
- `numpy`/stdlib only, CPU, seconds per dimension. Do not read anything under `verification/`
  or `frontier_eval/`.
