# ann-router Examples

[🇫🇷](https://github.com/warith-harchaoui/ann-router/blob/main/EXEMPLES.md)&nbsp;&nbsp;|&nbsp;&nbsp;[🇬🇧](https://github.com/warith-harchaoui/ann-router/blob/main/EXAMPLES.md)

A self-contained cookbook. Every snippet here was executed against
`ann-router 0.3.0` with the `[all]` extra installed, on an Apple Silicon Mac,
on 2026-10-27, and the commented output is what came back. Where a number is
machine- or data-dependent, it says so.

## Table of Contents

- [Setup](#setup)
- [1. Route: which backend, and why?](#1-route-which-backend-and-why)
- [2. auto_index: route, build and search in one call](#2-auto_index-route-build-and-search-in-one-call)
- [3. Drive a specific backend directly](#3-drive-a-specific-backend-directly)
- [4. Measure recall against the exact ground truth](#4-measure-recall-against-the-exact-ground-truth)
- [5. Metadata-filtered search (Qdrant)](#5-metadata-filtered-search-qdrant)
- [6. Capabilities and availability](#6-capabilities-and-availability)
- [7. Handle the honest failure modes](#7-handle-the-honest-failure-modes)
- [8. Tune the policy without touching code](#8-tune-the-policy-without-touching-code)
- [Command line](#command-line)
- [Troubleshooting](#troubleshooting)

## Setup

```bash
pip install 'ann-router[all]'     # or, from a clone: pip install -e '.[all]'
```

Every Python snippet below assumes these two imports and nothing else:

```python
import numpy as np
import ann_router as ar
```

The one thing worth printing before anything else is which engines this machine
can actually reach — the router will only ever hand back an installed one:

```python
print(ar.__version__)
# 0.3.0
print(ar.available_backends())
# ['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant', 'pgvector']
print(ar.detect_hardware())
# apple_silicon
```

`available_backends()` is the line whose output will differ most on your
machine: with no extras installed it returns `['exact']`, and every routing
snippet below still runs, falling back to the brute-force reference.

## 1. Route: which backend, and why?

```python
# target_recall relaxed below HIGH_RECALL=0.9, turbovec's calibrated ceiling
# (bench/results/calibrated_policy.yaml) -- otherwise the policy skips it.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)
# turbovec
print(choice.rationale)
# corpus receives frequent updates: turbovec offers O(1) add_with_ids/remove(id)
# with no index rebuild, plus TurboQuant 2-4 bit (~16x) compression — graph
# indexes degrade under deletes, so they are avoided here. target_recall=0.85
# is below HIGH_RECALL=0.9, within turbovec's calibrated range.
print([c["backend"] for c in choice.considered])
# ['turbovec', 'hnsw']

# At the house default target_recall=0.95 (>= HIGH_RECALL), the same dynamic
# corpus routes to hnsw instead -- turbovec is not even in the shortlist,
# not because it's uninstalled, but because it can't meet that recall.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True))
print(choice.backend)
# hnsw
print([c["backend"] for c in choice.considered])
# ['hnsw']
```

The router only ever returns an **installed** backend, and `considered` carries
the availability flag it used to decide:

```python
choice = ar.route(ar.Criteria(n_vectors=500_000, dim=768, dynamic=True, target_recall=0.85))
print([(c["backend"], c["available"], c["chosen"]) for c in choice.considered])
# [('turbovec', True, True), ('hnsw', True, False)]
```

If the policy's first pick is *not* installed, the router falls back and says so
in the rationale. You can read that exact wording without uninstalling
anything, by telling one adapter to report itself absent:

```python
from ann_router.backends.turbovec_backend import TurboVecIndex

TurboVecIndex.is_available = classmethod(lambda cls: False)  # pretend the extra is missing

choice = ar.route(ar.Criteria(n_vectors=500_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)
# hnsw
print(choice.rationale)
# Preferred backend 'turbovec' is not installed here, so the router fell back.
# dynamic corpus but turbovec unavailable: HNSW is the working fallback, though
# its tombstone deletes degrade the graph over time.
print([(c["backend"], c["available"], c["chosen"]) for c in choice.considered])
# [('turbovec', False, False), ('hnsw', True, True)]
```

(That monkey-patch exists to read the fallback path, not as a supported API.
Start a fresh interpreter afterwards.)

## 2. auto_index: route, build and search in one call

```python
rng = np.random.default_rng(0)

# Below the calibrated EXACT_MAX_N=1000, the router's advice is a brute scan.
small = rng.standard_normal((800, 128)).astype("float32")
index, choice = ar.auto_index(small, ar.Criteria(n_vectors=800, dim=128))
print(choice.backend)
# exact
ids, distances = index.search(small[:2], k=5)
print(ids.shape)
# (2, 5)
print(int(ids[0, 0]))
# 0        (a point's nearest neighbour is itself)

# Five thousand vectors is already past that threshold, so the same call
# routes to hnsw. "Small corpus" means 10^3 here, not 10^4.
big = rng.standard_normal((5_000, 128)).astype("float32")
index, choice = ar.auto_index(big, ar.Criteria(n_vectors=5_000, dim=128))
print(choice.backend)
# hnsw
```

`auto_index` trusts the array over the criteria: if `n_vectors`/`dim` disagree
with `vectors.shape`, the shape wins — for the routing decision too, not just
for the index it builds.

## 3. Drive a specific backend directly

```python
from ann_router.backends.hnsw import HNSWIndex

vecs = np.random.default_rng(1).standard_normal((10_000, 256)).astype("float32")
idx = HNSWIndex(dim=256, metric="cosine", M=16, ef=64).build(vecs)
ids, dist = idx.search(vecs[:1], k=10)
print(ids.shape, dist.shape)
# (1, 10) (1, 10)

idx.save("/tmp/my.hnsw")
idx2 = HNSWIndex(dim=256).load("/tmp/my.hnsw")
ids2, _ = idx2.search(vecs[:1], k=10)
print(int(ids2[0, 0]) == int(ids[0, 0]))
# True     (the reload answers identically)
```

## 4. Measure recall against the exact ground truth

`ExactIndex` is the reference: a full brute-force scan, recall 1.0 by
construction. Any approximate backend can be scored against it.

```python
from ann_router.backends.exact import ExactIndex
from ann_router.backends.turbovec_backend import TurboVecIndex

rng = np.random.default_rng(7)
corpus = rng.standard_normal((8_000, 128)).astype("float32")
q = rng.standard_normal((50, 128)).astype("float32")
truth, _ = ExactIndex(dim=128).build(corpus).search(q, 10)
pred, _ = TurboVecIndex(dim=128).build(corpus).search(q, 10)
recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth, pred)])
print(round(float(recall), 2))
# 0.85     (data-dependent; TurboQuant is lossy)
```

Or run the built-in bench across every installed backend at once:

```python
from ann_router._core_cli import do_bench

print(do_bench(n=5000, dim=128, k=10)["results"])
# {'exact':    {'recall': 1.0,   'build_s': 0.002, 'query_ms': 0.072},
#  'turbovec': {'recall': 0.736, 'build_s': 0.1,   'query_ms': 0.073},
#  'hnsw':     {'recall': 1.0,   'build_s': 0.174, 'query_ms': 0.011},
#  'faiss':    {'recall': 1.0,   'build_s': 0.02,  'query_ms': 0.028},
#  'annoy':    {'recall': 0.0,   'build_s': 0.088, 'query_ms': 0.015},
#  'qdrant':   {'recall': 1.0,   'build_s': 0.396, 'query_ms': 2.079}}
```

The recall column reproduces exactly (the corpus is seeded); `build_s` and
`query_ms` are one run's wall-clock on one machine and will not. Two things in
that output are the environment rather than the engine, and both are explained
under [Troubleshooting](#troubleshooting): `annoy` at recall 0.0 is the broken
PyPI wheel, and `pgvector` is missing from the dict entirely because no
PostgreSQL was running.

## 5. Metadata-filtered search (Qdrant)

Qdrant is the engine the router reaches for when the question is "nearest
neighbours **where** `lang == 'fr'`". The adapter defaults to Qdrant's embedded
in-memory client, so this snippet needs no server.

```python
from ann_router.backends.qdrant_backend import QdrantIndex

vecs = np.random.default_rng(2).standard_normal((1_000, 64)).astype("float32")
payloads = [{"lang": "fr" if i % 2 else "en"} for i in range(1_000)]
idx = QdrantIndex(dim=64).build(vecs, payloads=payloads)

# nearest neighbours restricted to French documents:
ids, scores = idx.search_filter(vecs[:1], k=5, where={"lang": "fr"})
print(ids.tolist())
# [[361, 869, 409, 555, 85]]
print(all(int(i) % 2 == 1 for i in ids[0]))
# True     (every hit is an odd index, i.e. lang == 'fr')
```

`pgvector` exposes the same `search_filter` against a SQL
`WHERE payload->>...`, but it needs a live DSN (`dsn=...`, or
`ANN_ROUTER_PG_DSN`), so it is not runnable from a bare checkout.

## 6. Capabilities and availability

```python
print(ar.available_backends())
# ['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant', 'pgvector']

caps = ar.all_capabilities()
print(caps["annoy"].supports_remove)
# False    (frozen after build)
print(caps["qdrant"].supports_filter)
# True
print(caps["annoy"].pip_extra)
# annoy    (so the fix is: pip install 'ann-router[annoy]')
```

`all_capabilities()` answers for **every** registered backend, installed or
not — the descriptors are plain data, readable without the dependency, which is
what lets `route()` explain a fallback instead of crashing on an import.

## 7. Handle the honest failure modes

```python
from ann_router.backends.annoy_backend import AnnoyIndex
from ann_router.base import NotSupported

idx = AnnoyIndex(dim=8).build(np.random.rand(100, 8).astype("float32"))
try:
    idx.remove(np.array([1]))
except NotSupported as e:
    print(e)
# annoy: remove() unsupported — rebuild the index without those ids
```

The two exceptions answer deliberately different questions. `NotSupported`
means *this engine cannot do that, ever* — rebuild, or route elsewhere.
`BackendUnavailable` means *this engine is not installed here*, and carries the
`pip install` line that fixes it.

## 8. Tune the policy without touching code

```python
# Raise the exact->ANN crossover to 100k for this call:
choice = ar.route(ar.Criteria(n_vectors=50_000, dim=128), thresholds={"EXACT_MAX_N": 100_000})
print(choice.backend)
# exact
print(choice.rationale)
# n=50,000 < 100,000 (EXACT_MAX_N=100,000 scaled to a 10 ms budget): a
# brute-force scan is already instant and exact (recall 1.0), so approximation
# would only add build cost.

# The same criteria under the shipped policy:
print(ar.route(ar.Criteria(n_vectors=50_000, dim=128)).backend)
# hnsw
```

Or export `ANN_ROUTER_POLICY=/path/to/policy.yaml` to override the shipped
thresholds process-wide. Start from
[`ann_router/policy.yaml`](https://github.com/warith-harchaoui/ann-router/blob/main/ann_router/policy.yaml):
each threshold there carries the calibration note that produced it.

## Command line

One click command, five subcommands: `route`, `build`, `search`, `bench`,
`capabilities`.

```bash
ann-router route --n-vectors 2000000 --dim 768 --dynamic --markdown
```

```markdown
# ann-router decision: `hnsw`

**Rationale.** dynamic corpus, but target_recall=0.95 is at/above
HIGH_RECALL=0.9 — turbovec's calibrated benchmarks consistently undershoot that
recall (see policy.py's HIGH_RECALL docstring), so HNSW's better recall is used
despite the tombstone-delete downside on a churning corpus.

**Recommended config.** `{'metric': 'cosine', 'M': 16, 'ef_construction': 200, 'ef': 64}`

## Considered (priority order)

| backend | available | chosen | reason |
| --- | --- | --- | --- |
| `hnsw` | yes | **<-** | dynamic corpus, but target_recall=0.95 is at/above HIGH_RECALL=0.9 ... |

_policy version 1.2.0_
```

Build an index from a `.npy` file, then search it. `build` writes the index
plus a small JSON sidecar (`my.idx.meta.json`) recording the backend and the
geometry, so `search` can rebuild the right adapter without being told which
one wrote the file:

```bash
python -c "import numpy as np; np.save('c.npy', np.random.rand(20000,128).astype('float32'))"
ann-router build --n-vectors 20000 --dim 128 --vectors c.npy --index my.idx
# {
#   "backend": "hnsw",
#   "rationale": "stable in-memory corpus, high precision wanted: hnswlib gives the
#                 best recall/latency of the in-memory engines when the index rarely
#                 changes (note: it deletes only via tombstones, hence 'stable').",
#   "index_path": "my.idx",
#   "n": 20000,
#   "dim": 128,
#   "build_seconds": 1.4306
# }

python -c "import numpy as np; np.save('q.npy', np.random.rand(3,128).astype('float32'))"
ann-router search --index my.idx --queries q.npy -k 10
# {"ids": [[ ...10 ids... ], ...], "distances": [[0.17006641626358032, ...], ...]}
```

```bash
# Benchmark installed backends against exact ground truth (the §4 numbers)
ann-router bench --n 5000 --dim 128 -k 10

# Availability + capability matrix, merged with the prose catalogue
ann-router capabilities

# -V/--version and -h/--help both work, on the group and on every subcommand
ann-router --version
# ann-router, version 0.3.0
```

The HTTP and MCP doors call the very same core functions, so they cannot
disagree with the CLI: `uvicorn ann_router.api:app` serves `POST /route`,
`GET /capabilities` and `GET /bench`, and `ann-router-mcp` mounts those three
as MCP tools at `http://127.0.0.1:8019/mcp`.

## Troubleshooting

**`annoy` reports recall 0.0.** Not a defect in this package. The PyPI `annoy`
wheel is built from source and, on Apple Silicon, returns one neighbour then
`-1` padding regardless of `k` — which is exactly the 0.0 in §4's bench output.
Replace it with the conda-forge build:

```bash
pip uninstall -y annoy
conda install -c conda-forge python-annoy -y
```

`annoy` also has no numpy-2 ABI, so pin `numpy<2` in any environment that uses
it. Both caveats are detailed in
[INSTALL.md](https://github.com/warith-harchaoui/ann-router/blob/main/INSTALL.md).

**`pgvector` is absent from the bench results.** `PgVectorIndex.is_available()`
only checks that `psycopg` and `pgvector` import; actually using it needs a live
PostgreSQL. Without a DSN it raises `NotSupported: pgvector: no DSN — pass
dsn=... or set ANN_ROUTER_PG_DSN to a live Postgres`, and the bench skips it.
Start one with `docker compose --profile backends up`.

**`BackendUnavailable` on a backend you expected.** The message carries the fix,
e.g. `qdrant-client not installed. Run: pip install 'ann-router[qdrant]'`. The
router does not need you to act on it: it routes around the gap and says so in
the rationale (§1).

**The router keeps answering `hnsw`.** That is the designed default for a stable
in-memory corpus. Read `choice.sensitivity` to see which criterion came closest
to changing the verdict; an empty list means none of them flips it within 25 %.
