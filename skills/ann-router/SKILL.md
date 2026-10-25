---
name: ann-router
description:>-
  Select the right approximate-nearest-neighbour (ANN) vector-search backend
  from measured problem criteria — exact / turbovec / HNSW / FAISS / Annoy /
  Qdrant / pgvector — and get a justified, discussable rationale plus a
  ready-to-use index. Use when someone must CHOOSE a vector index / ANN library
  (not serve one), asks "FAISS or HNSW or Annoy?", has a corpus with frequent
  updates, needs metadata-filtered vector search, or wants to benchmark recall.
  Fires on "which ANN / vector index for N vectors", "FAISS vs HNSW vs Annoy vs
  Qdrant vs pgvector", "my corpus has constant inserts and deletes — what
  index", "metadata-filtered vector search", "brute force or an ANN index?",
  "benchmark vector-search recall", "quel index vectoriel utiliser", and on
  `ann-router`, `ann-router-mcp`, `import ann_router as
  ar`. Do NOT use
  to embed data, stand up a database, serve an existing index, or pick a local
  LLM (best-engine-ai-helper). The exhaustive trigger catalogue is in
  `references/triggers.md`.
license: BSD-3-Clause
metadata:
  author: Warith Harchaoui
  package: ann-router
  suite: ai-helpers
  homepage: https://deraison.ai/ai-helpers
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
choice.backend  # 'turbovec'
choice.rationale  # the justification
```

Route + build + search in one step with `ar.auto_index(vectors, criteria)`.

## The decision, in one glance

| criteria | → backend |
| --- | --- |
| `n < EXACT_MAX_N` | exact (brute force) |
| frequent updates | turbovec |
| very large + GPU/batch | FAISS (IVF/PQ) |
| persistence + metadata filters | Qdrant / pgvector |
| read-only + tight memory | Annoy |
| stable in-memory (default) | HNSW |

Thresholds (`EXACT_MAX_N`, `FAISS_MIN_N`, ...) are being calibrated from
measured data in `bench/` rather than guessed — see `bench/README.md`.

## Progressive disclosure — go deeper only if needed

- **Full command and flag matrix, engine extras, policy thresholds** →
  `references/cli-reference.md`.
- **Library, HTTP API, MCP, configuration files** → `references/surfaces.md`.
- **Exhaustive trigger catalogue (English and French)** →
  `references/triggers.md`.
- **Why not just pick FAISS** → the repo's `LANDSCAPE.md` / `PAYSAGE.md`.
- **Install / platform caveats** (Apple-Silicon annoy, pgvector server)
  → the repo's `INSTALL.md`.
- **Benchmark recall** → `ann-router bench --n 5000 --dim 128 -k 10`.
- **Capabilities matrix** → `ann-router capabilities`.

## Install

```bash
pip install ann-router            # core: library + click CLI + exact backend
pip install 'ann-router[all]'     # every pip-installable engine + cli + api
```
