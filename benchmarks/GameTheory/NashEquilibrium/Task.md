# NashEquilibrium — compute approximate Nash equilibria of bimatrix games

## Scientific background

Computing a Nash Equilibrium (NE) of a two-player game is PPAD-complete — no polynomial-time
algorithm is known. The quality of an approximate NE is measured by **exploitability**: the
maximum any player can gain by unilateral deviation. Games with 30-50 actions have a vast
strategy space (10¹⁵+ pure profiles) where support enumeration is infeasible and local methods
get stuck. Finding low-exploitability solutions underpins algorithmic game theory, mechanism
design, and multi-agent AI.

Reference: Lemke & Howson, SIAM J. Appl. Math. 12, 413 (1964); Porter et al., Games & Econ. Behav. 63, 642 (2008).

## Your task

```python
def find_nash(A, B):
    \"\"\"Given payoff matrices A, B of shape (n, n), return (p, q) where p, q are
    probability vectors (non-negative, sum to 1) forming an approximate Nash equilibrium.\"\"\"
```

## Scoring

```
score = clip( (eps_uniform - eps_found) / eps_uniform, 0, 1 )
```
where eps = total exploitability. Exact NE (eps=0) → score=1.0. Instances: n∈{10, 30, 50}.

## Rules

- Only edit `solution.py`. numpy/scipy only. CPU, seconds. Do not read `verification/`.
