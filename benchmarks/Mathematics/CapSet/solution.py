"""Initial baseline for CapSet (the {0,1}^n hypercube).

The set of all 0/1 vectors is always a valid cap set of size 2^n (any three of them summing
to 0 mod 3 must be identical), but it is far from the largest cap. Edit this file to build a
bigger cap — e.g. greedy / randomized-restart search, backtracking, or the classic product
and Hill-cap constructions.
"""

import itertools


def build_capset(n: int):
    """Return a list of vectors in {0,1,2}^n forming a cap set (no 3 distinct collinear)."""
    return [list(v) for v in itertools.product([0, 1], repeat=n)]
