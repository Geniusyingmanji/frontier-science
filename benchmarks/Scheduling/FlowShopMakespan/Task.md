# FlowShopMakespan — minimize completion time in a flow shop

## Scientific background
In the permutation flow shop, N jobs must be processed on M machines in the same order.
Each job has known processing times on each machine. The goal is to find the job permutation
minimizing the makespan (completion time of the last job on the last machine). This is
NP-hard for M≥3 (Garey et al. 1976) and the NEH heuristic (1983) remains one of the best
constructive methods after 40 years.

Reference: Nawaz, Enscore & Ham, Omega 11, 91 (1983); Taillard, EJOR 64, 278 (1993).

## Your task
```python
def schedule_flowshop(processing_times, n_jobs, n_machines):
    \"\"\"Given (n_jobs, n_machines) processing time matrix, return a permutation of [0..n-1].\"\"\"
```

## Scoring
`score = clip((makespan_natural - makespan_found) / (makespan_natural - makespan_NEH), 0, 1)`
Natural order → score 0. NEH heuristic → score 1.0. Better than NEH → also 1.0 (clipped).

## Rules
- Only edit `solution.py`. numpy/scipy only. CPU. Do not read `verification/`.
