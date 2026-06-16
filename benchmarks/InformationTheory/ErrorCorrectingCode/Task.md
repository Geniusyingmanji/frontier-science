# ErrorCorrectingCode — design binary linear codes with maximum minimum distance

## Scientific background
A binary linear [n, k, d] code encodes k information bits into n-bit codewords with minimum
Hamming distance d, enabling correction of ⌊(d-1)/2⌋ errors. Maximizing d for given (n, k) is
a fundamental problem in coding theory underpinning digital communications, storage, and
quantum error correction. The best codes (Reed-Muller, Golay, BCH) have rich algebraic
structure that random constructions cannot match.

Reference: MacWilliams & Sloane, Theory of Error-Correcting Codes (1977); Grassl, codetables.de.

## Your task
```python
def design_code(n, k):
    \"\"\"Return a generator matrix G of shape (k, n) over GF(2) (entries 0 or 1).
    The code = row space of G. Goal: maximize minimum Hamming distance d.\"\"\"
```

## Scoring
`score = clip((d_found - d_random) / (d_best_known - d_random), 0, 1)` per instance.
Instances: [16,8] (best d=5), [24,12] (best d=8, Golay), [32,16] (best d=8).

## Rules
- Only edit `solution.py`. numpy only. CPU. Do not read `verification/`.
