# Decision study — grounding for the routing policy

The `ann_router.policy` decision tree is not guessed; it generalises the shipped
brute-force → turbovec routing from the `roitelet` lab and the suite's own
benchmark findings. This note records the evidence and the measured recall on
this machine, so a future change to a threshold is a reviewed, evidence-backed
act (CODING.md rules 14-15).

## Sources

- **roitelet** (`core/personal.py`, `core/pixel_rag.py`): a production two-step
  rule — exact brute force below a per-host crossover, then turbovec (4-bit
  TurboQuant) for a static/large corpus, with an optional AMIPS seam at millions
  of vectors. The crossover is *measured* per host/dimension, not hard-coded.
- **roitelet** `tests/eval/bench_ann_backends.py`: a sweep over 8 backends
  (numpy, turbovec, faiss, hnswlib, qdrant, usearch, annoy, pgvector) that reads
  the brute-force ceiling (the N at which an exact scan exceeds a 10 ms budget)
  as the ANN switch point — the origin of `EXACT_MAX_N`.
- **roitelet** `doc_en/EVALUATION_STUDIES.md` "Study 1": FAISS-HNSW recall
  collapses on near-orthogonal vectors (~0.47 at N=5k) and loses to turbovec on
  disk size and recall — hence FAISS is *not* the default and is reserved for the
  very-large + GPU/batch regime.

## Measured recall@10 on this machine (Apple M2 Max, 2026-08-04)

Fixed synthetic clustered corpus (50 centres, spread 0.4, N=8000, D=128, 50
queries), scored against the exact brute-force ground truth
(`tests/conftest.py`):

| backend | recall@10 | versioned floor |
| --- | --- | --- |
| exact | 1.000 | 1.00 |
| hnsw | 1.000 | 0.90 |
| faiss | 1.000 | 0.90 |
| annoy | 1.000 | 0.90 |
| qdrant | 1.000 | 0.90 |
| turbovec | 0.700 | 0.55 |
| scann | — (no macOS wheel) | 0.90 |
| pgvector | — (needs live server) | 0.90 |

turbovec's floor is deliberately lower: its 2-4 bit TurboQuant is lossy and
data-dependent (0.70 on this clustered set, ~0.85 on sphere-distributed data),
so it wins the *dynamic* branch on mutability, not the *max-recall* branch.

## Environment note

The PyPI `annoy` wheel is miscompiled on macOS/arm64 (returns one neighbour
regardless of `k`) — the same defect that made roitelet's bench report annoy
0.0. The conda-forge `python-annoy` binary is correct and is what produced the
1.000 above. See `INSTALL.md`.
