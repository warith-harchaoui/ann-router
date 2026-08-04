# ann-router

[🇫🇷 LISEZMOI.md](LISEZMOI.md) · 🇬🇧 English

![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Local-first](https://img.shields.io/badge/local--first-yes-brightgreen)

`ann-router` belongs to the **sev7n AI Helpers** suite. It is a *router*: you
describe your approximate-nearest-neighbour (ANN) vector-search problem — in
*measured* terms — and it selects, **justifies**, and can **instantiate** the
right backend engine, instead of marrying you to a single library.

It is the vector-search sibling of
[`best-engine-ai-helper`](https://github.com/warith-harchaoui/best-engine-ai-helper)
(which picks the best local LLM for a machine). Same philosophy: **measure the
criteria → select the engine → return a discussable rationale.**

The engines it routes among:

> **exact (brute force) · turbovec · HNSW (hnswlib) · FAISS (IVF/PQ) · Annoy ·
> ScaNN · Qdrant · pgvector**

Importing the package is cheap and dependency-free — no engine's optional
dependency is loaded at import time, so `import ann_router` works with only
numpy installed, and a backend whose dependency is absent simply reports itself
unavailable while the router routes around it.

## Why route instead of just picking FAISS?

Because the right engine is a *function of the problem*, and the problem changes:
a 5k-vector corpus wants an exact scan (instant, recall 1.0); a corpus with
constant inserts/deletes wants turbovec (O(1) mutation); one needing SQL
`WHERE`-filters wants pgvector; a frozen, memory-tight corpus wants Annoy. Hard
-coding one library gets one of these right and the rest wrong. See
[LANDSCAPE.md](LANDSCAPE.md).

## Install

Core (library + always-on argparse CLI):

```bash
pip install -e ~/os-helper      # suite foundation
pip install -e .                # or: pip install ann-router
```

Add engines as needed (per-backend extras), or everything at once:

```bash
pip install 'ann-router[hnsw]'      # one engine
pip install 'ann-router[all]'       # every pip-installable engine + cli + api
```

Full, platform-specific instructions — including the **Apple Silicon annoy**
caveat and **ScaNN**/**pgvector** notes — are in [INSTALL.md](INSTALL.md).

## Quick start (library)

```python
import numpy as np
import ann_router as ar

# 1. Describe the problem in measured terms.
criteria = ar.Criteria(
    n_vectors=2_000_000, dim=768,
    dynamic=True,              # frequent adds/removes
    target_recall=0.95,
    hardware=ar.detect_hardware(),
)

# 2. Ask which backend — and why.
choice = ar.route(criteria)
print(choice.backend)         # 'turbovec'
print(choice.rationale)       # "corpus receives frequent updates: turbovec offers O(1) ..."

# 3. Or route + build in one call, then search.
vectors = np.random.default_rng(0).standard_normal((5_000, 768)).astype("float32")
index, choice = ar.auto_index(vectors, ar.Criteria(n_vectors=5_000, dim=768))
ids, distances = index.search(vectors[:3], k=10)
```

Every backend speaks the same `ANNIndex` interface:

```python
index.build(vectors, ids=None)
index.add(vectors); index.add_with_ids(vectors, ids); index.remove(ids)
ids, distances = index.search(queries, k)
index.save(path); index.load(path)
Backend.capabilities()        # supports_remove / supports_filter / persistent / needs_gpu ...
```

Operations a backend genuinely cannot do (e.g. `Annoy.remove`) raise a clear
`NotSupported`; a backend whose dependency is missing raises `BackendUnavailable`
with the `pip install` line that fixes it.

## The five doors (one core, five surfaces)

1. **Library** — everything above (`ann_router`).
2. **CLI** — `ann-router` (argparse, always available) and the `ann-router-click`
   twin (`[cli]` extra). Subcommands: `route`, `build`, `search`, `bench`,
   `capabilities`.
3. **HTTP API** — `uvicorn ann_router.api:app` (`[api]` extra): `POST /route`,
   `GET /capabilities`, `GET /bench`.
4. **MCP server** — `python -m ann_router.mcp_server` (`[mcp]` extra): exposes
   `route`, `capabilities`, `bench` as agent tools.
5. **Skill** — `skills/ann-router/SKILL.md`, so an agent knows when to reach for
   the router.

```bash
ann-router route --n-vectors 2000000 --dim 768 --dynamic --markdown
ann-router bench --n 5000 --dim 128 -k 10
ann-router capabilities
```

## How selection works

The decision tree (tunable via `policy.yaml` / `ANN_ROUTER_POLICY`):

| # | If the criteria say… | Route to | Because |
| - | -------------------- | -------- | ------- |
| 1 | `n < 10 000` | **exact** | a brute-force scan is already instant and exact (recall 1.0) |
| 2 | frequent updates | **turbovec** | O(1) add/remove, no rebuild; TurboQuant 2-4 bit (~16×) |
| 3 | very large + GPU/batch | **FAISS** | IVF+PQ scales; GPU batch throughput |
| 4 | persistence + metadata filters | **Qdrant / pgvector** | on-disk HNSW + payload/SQL `WHERE` filtering |
| 5 | max recall at scale | **ScaNN** | anisotropic (score-aware) quantisation |
| 6 | read-only + tight memory | **Annoy** | frozen, memory-mapped, very lean |
| 7 | stable in-memory (default) | **HNSW** | best recall/latency when the index rarely changes |

The router returns not just the name but the **criteria that drove it** and the
**alternatives it considered** (including any preferred-but-uninstalled engine it
fell back from), so the choice is auditable and overridable.

## Criteria (the input spec)

`n_vectors`, `dim`, `target_recall`, `latency_budget_ms`, `memory_budget_gb`,
`dynamic`, `metadata_filtering`, `hardware` (`cpu`/`gpu`/`apple_silicon`,
auto-detectable), `persistence`, `batch_queries`, `metric`
(`cosine`/`l2`/`ip`). Only `n_vectors` and `dim` are required.

## More

- [EXAMPLES.md](EXAMPLES.md) — a runnable cookbook.
- [LANDSCAPE.md](LANDSCAPE.md) — how ann-router compares to just picking one engine.
- [CODING.md](CODING.md) — the coding standard this repo holds itself to.
- [CONTRIBUTING.md](CONTRIBUTING.md) · [CHANGELOG.md](CHANGELOG.md) · [TRIGGERS.md](TRIGGERS.md)

## Author

[Warith HARCHAOUI](https://harchaoui.org/warith), Ph.D.

## Acknowledgements

Built on the shipped brute-force → turbovec routing prototype from the
`roitelet` lab, generalised to eight backends behind one interface.

## License

BSD-3-Clause — see [LICENSE](LICENSE).
