# CapSet — known-best cap sizes

| dim n | baseline {0,1}^n = 2^n | best known max | status |
|---|---|---|---|
| 4 | 16 | 20 | proven max |
| 5 | 32 | 45 | proven max |
| 6 | 64 | 112 | proven max |
| 7 | 128 | 236 | best known |
| 8 | 256 | 512 | FunSearch 2023 |

`sota_ref` in the evaluator uses these as the score=1.0 anchor. Exceeding a best-known size
(only open for n>=7) scores >1.0. Update this table and `verification/evaluator.py:SIZES`
if a larger verified cap is found, and consider adding n>=7 to push the open frontier.
