"""Initial baseline for ProteinLatticeHP (weak but valid).

Folds the HP sequence as a straight horizontal line on the 2D lattice. This is a valid
self-avoiding walk but produces zero non-bonded H-H contacts (energy = 0), so it scores 0.
Edit this file to implement a real folding algorithm (Monte Carlo, pull moves, genetic
algorithm, beam search, ...).
"""


def fold_protein(seq: str) -> list:
    """Return a list of (x, y) integer coords on the 2D square lattice."""
    return [(i, 0) for i in range(len(seq))]
