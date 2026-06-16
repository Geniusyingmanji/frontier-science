# TopologyOptimization — minimize structural compliance of a cantilever beam

## Scientific background

Topology optimization determines the optimal material distribution within a design domain to
maximize stiffness (minimize compliance) under a volume constraint. The SIMP method represents
density ρ(x) ∈ [0,1] with penalization E(ρ) = ρ^p · E₀ to drive toward 0/1. This problem
launched a field (Bendsøe & Kikuchi 1988) and remains the gold standard for structural
optimization benchmarks (Sigmund's "99 line" code, 2001).

## Your task

```python
def optimize_topology(nelx, nely, volfrac):
    \"\"\"Return (nely, nelx) array of material densities in [0.001, 1.0].
    Mean density must be <= volfrac. Domain: cantilever with fixed left edge,
    point load at bottom-right. Lower compliance (stiffer) = higher score.\"\"\"
```

## Scoring

```
score = clip( (C_uniform - C_found) / (C_uniform - C_optimal), 0, 1 )
```
where C_uniform is the compliance with uniform ρ=volfrac everywhere. Optimal topology achieves
~70% compliance reduction. Grid: 60×20, volume fraction 0.4.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU, <30s. Do not read `verification/`.
