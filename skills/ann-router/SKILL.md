---
name: ann-router
description: >-
  Select the right approximate-nearest-neighbour (ANN) vector-search backend
  from measured problem criteria — exact / turbovec / HNSW / FAISS / Annoy /
  ScaNN / Qdrant / pgvector — and get a justified, discussable rationale plus a
  ready-to-use index. Use when someone must CHOOSE a vector index / ANN library
  (not serve one), asks "FAISS or HNSW or Annoy?", has a corpus with frequent
  updates, needs metadata-filtered vector search, or wants to benchmark recall.
when_to_use:
  - The user must decide which ANN engine / vector index to use for a given
    corpus size, dimensionality, recall/latency/memory budget, or update pattern.
  - The user is about to hard-code FAISS (or any single engine) and it may be
    the wrong fit (small corpus, churny data, needs filtering/persistence).
  - The user asks to benchmark recall@k of vector backends against exact truth.
triggers:
  - "which ANN / vector index library for N vectors"
  - "FAISS vs HNSW vs Annoy vs ScaNN vs Qdrant vs pgvector"
  - "my corpus has constant inserts/deletes — what index"
  - "metadata-filtered vector search"
  - "brute force or ANN index?"
  - "benchmark vector-search recall"
  - "quel index vectoriel / bibliothèque ANN utiliser"
---

# ann-router

Route to the right ANN vector-search backend from *measured* criteria, instead
of marrying a single library. Part of the AI Helpers suite; the
vector-search sibling of `best-engine-ai-helper`.

## When to reach for this skill

Fire it whenever the task is **choosing** a vector index, not operating one:
sizing questions ("is brute force enough?"), engine comparisons ("FAISS or
HNSW?"), workload constraints (frequent updates, metadata filtering, tight
memory, persistence), or a request to measure recall. Do **not** use it to embed
data or to stand up a database.

## The one thing to run

```bash
ann-router route --n-vectors <N> --dim <D> [--dynamic] [--metadata-filtering] \
                 [--target-recall 0.95] [--memory-budget-gb 2] [--markdown]
```

It prints the chosen backend, the rationale (which criteria drove it), and the
alternatives considered — including any preferred-but-uninstalled engine it fell
back from. Add `--markdown` for a human-readable report.

From Python:

```python
import ann_router as ar
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True))
choice.backend      # 'turbovec'
choice.rationale    # the justification
```

Route + build + search in one step with `ar.auto_index(vectors, criteria)`.

## The decision, in one glance

| criteria | → backend |
| --- | --- |
| `n < 10k` | exact (brute force) |
| frequent updates | turbovec |
| very large + GPU/batch | FAISS (IVF/PQ) |
| persistence + metadata filters | Qdrant / pgvector |
| max recall at scale | ScaNN |
| read-only + tight memory | Annoy |
| stable in-memory (default) | HNSW |

## Progressive disclosure — go deeper only if needed

- **Full criteria & API** → repo `README.md`, `EXAMPLES.md`.
- **Why not just pick FAISS** → `LANDSCAPE.md` / `PAYSAGE.md`.
- **Install / platform caveats** (Apple-Silicon annoy, Linux-only ScaNN,
  pgvector server) → `INSTALL.md`.
- **Tune the policy** → `ann_router/policy.yaml` or `ANN_ROUTER_POLICY`.
- **Benchmark recall** → `ann-router bench --n 5000 --dim 128 -k 10`.
- **Capabilities matrix** → `ann-router capabilities`.

## Install

```bash
pip install ann-router            # core: library + argparse CLI + exact backend
pip install 'ann-router[all]'     # every pip-installable engine + cli + api
```
