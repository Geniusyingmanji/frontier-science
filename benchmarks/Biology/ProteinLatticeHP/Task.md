# ProteinLatticeHP — fold HP sequences on a 2D lattice

## Scientific background

The **HP lattice model** (Dill, 1985) is a foundational model of protein folding: a protein
is a string of hydrophobic (H) and polar (P) residues placed as a self-avoiding walk on a
2D square lattice. The energy equals the negative count of **non-bonded H–H contacts**
(lattice-adjacent H residues that are not sequence neighbors). Minimizing this energy is
NP-hard and captures the core challenge of the protein folding problem — finding a compact
hydrophobic core. The model launched decades of algorithm development (Monte Carlo, genetic
algorithms, constraint satisfaction, ant colony optimization).

## Your task

Edit **`solution.py`** so it defines:

```python
def fold_protein(seq: str) -> list:
    """Given an HP sequence (e.g. 'HPHPPHHP...'), return a list of (x, y) integer
    coordinates on the 2D square lattice. The fold must be a valid self-avoiding walk
    (consecutive residues lattice-adjacent, no two at the same position)."""
```

The evaluator calls `fold_protein` on several benchmark HP sequences, validates the fold,
counts non-bonded H–H contacts, and compares to the known optimal energy.

## Scoring

```
score = energy_found / energy_optimal       (both ≤ 0; 0 contacts = score 0, optimal = 1.0)
```

`combined_score` is the mean over sequences. Test sequences: HP20 (optimal −9), HP25
(optimal −8), HP36 (optimal −14).

## Rules

- Only edit `solution.py`. Keep the `fold_protein(seq)` signature and `[(x,y), ...]` output.
- The fold must be a valid self-avoiding walk (consecutive residues adjacent, no collisions).
- `numpy`/stdlib only. CPU, seconds per sequence. Do not read `verification/` or `frontier_eval/`.
