# GraphMaxCut — maximize cut weight in a graph

## Scientific background

Given a graph with non-negative edge weights, the **Max-Cut** problem asks for a partition
of vertices into two sets that maximizes the total weight of edges crossing the cut. Max-Cut
is NP-hard and has deep connections to statistical physics (the Ising model), quantum
computing (QAOA), and approximation algorithms (the Goemans–Williamson SDP gives a 0.878
guarantee). It is a core benchmark for combinatorial optimization.

## Your task

Edit **`solution.py`** so it defines:

```python
def solve_maxcut(n: int, W: "np.ndarray") -> "np.ndarray":
    """Given n vertices and the (n, n) symmetric weight matrix W (W[i,j] = edge weight),
    return a length-n binary array in {0, 1} partitioning the vertices."""
```

The evaluator constructs several fixed weighted random graphs, calls your function, and
measures the cut value against the known optimum (found by exact enumeration).

## Scoring

```
score = clip( (cut_found − cut_baseline) / (cut_optimal − cut_baseline), 0, 1 )
```

where `cut_baseline` is the weak random-3 baseline. `combined_score` is the mean over
instances. Graphs: n ∈ {18, 20, 22} with varying density.

## Rules

- Only edit `solution.py`. Keep the `solve_maxcut(n, W)` signature and {0,1} output.
- `numpy` and `scipy` only. CPU, seconds per instance. No network.
- Do not read `verification/` or `frontier_eval/`.
