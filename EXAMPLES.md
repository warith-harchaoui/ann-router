# ann-router Examples

A self-contained cookbook. Every snippet runs against the installed package
(`pip install -e '.[all]'`); outputs are shown as `# =>` comments.

## 1. Route: which backend, and why?

```python
import ann_router as ar

# target_recall relaxed below HIGH_RECALL=0.9, turbovec's calibrated ceiling
# (bench/results/calibrated_policy.yaml) -- otherwise the policy skips it.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)      # => turbovec
print(choice.rationale)    # => "corpus receives frequent updates: turbovec offers O(1) ..."
print([c["backend"] for c in choice.considered])   # => ['turbovec', 'hnsw']

# At the house default target_recall=0.95 (>= HIGH_RECALL), the same dynamic
# corpus routes to hnsw instead -- turbovec is not even in the shortlist,
# not because it's uninstalled, but because it can't meet that recall.
choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True))
print(choice.backend)      # => hnsw
print([c["backend"] for c in choice.considered])   # => ['hnsw']
```

The router only ever returns an **installed** backend. If the policy's first
pick is not installed, it falls back and says so: on a machine where the
`turbovec` extra isn't installed, a dynamic, recall-relaxed corpus falls back
to `hnsw`:

```python
choice = ar.route(ar.Criteria(n_vectors=500_000, dim=768, dynamic=True, target_recall=0.85))
print(choice.backend)      # => hnsw   (if turbovec is uninstalled here)
print(choice.rationale)    # => "Preferred backend 'turbovec' is not installed here, so ..."
```

## 2. auto_index: route + build + search in one call

```python
import numpy as np, ann_router as ar

rng = np.random.default_rng(0)
vectors = rng.standard_normal((5_000, 128)).astype("float32")

index, choice = ar.auto_index(vectors, ar.Criteria(n_vectors=5_000, dim=128))
print(choice.backend)                      # => exact  (small corpus)
ids, distances = index.search(vectors[:2], k=5)
print(ids.shape)                           # => (2, 5)
print(int(ids[0, 0]))                      # => 0  (a point's nearest neighbour is itself)
```

## 3. Drive a specific backend directly

```python
from ann_router.backends.hnsw import HNSWIndex
import numpy as np

vecs = np.random.default_rng(1).standard_normal((10_000, 256)).astype("float32")
idx = HNSWIndex(dim=256, metric="cosine", M=16, ef=64).build(vecs)
ids, dist = idx.search(vecs[:1], k=10)
idx.save("/tmp/my.hnsw")
idx2 = HNSWIndex(dim=256).load("/tmp/my.hnsw")
```

## 4. Measure recall against the exact ground truth

```python
import numpy as np, ann_router as ar
from ann_router.backends.exact import ExactIndex
from ann_router.backends.turbovec_backend import TurboVecIndex

rng = np.random.default_rng(7)
corpus = rng.standard_normal((8_000, 128)).astype("float32")
q = rng.standard_normal((50, 128)).astype("float32")
truth, _ = ExactIndex(dim=128).build(corpus).search(q, 10)
pred, _ = TurboVecIndex(dim=128).build(corpus).search(q, 10)
recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(truth, pred)])
print(round(float(recall), 2))   # => ~0.85 (data-dependent; TurboQuant is lossy)
```

Or just run the built-in bench across every installed backend:

```python
import ann_router as ar
from ann_router._core_cli import do_bench
print(do_bench(n=5000, dim=128, k=10)["results"])
# => {'exact': {'recall': 1.0, ...}, 'hnsw': {'recall': 1.0, ...}, 'turbovec': {...}, ...}
```

## 5. Metadata-filtered search (Qdrant)

```python
import numpy as np
from ann_router.backends.qdrant_backend import QdrantIndex

vecs = np.random.default_rng(2).standard_normal((1_000, 64)).astype("float32")
payloads = [{"lang": "fr" if i % 2 else "en"} for i in range(1_000)]
idx = QdrantIndex(dim=64).build(vecs, payloads=payloads)

# nearest neighbours restricted to French documents:
ids, scores = idx.search_filter(vecs[:1], k=5, where={"lang": "fr"})
```

`pgvector` exposes the same `search_filter` against a SQL `WHERE payload->>...`.

## 6. Capabilities & availability

```python
import ann_router as ar

print(ar.available_backends())
# => ['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant', 'pgvector']

caps = ar.all_capabilities()
print(caps["annoy"].supports_remove)   # => False (frozen after build)
print(caps["qdrant"].supports_filter)  # => True
```

## 7. Handle the honest failure modes

```python
import numpy as np
from ann_router.backends.annoy_backend import AnnoyIndex
from ann_router.base import NotSupported

idx = AnnoyIndex(dim=8).build(np.random.rand(100, 8).astype("float32"))
try:
    idx.remove(np.array([1]))
except NotSupported as e:
    print(e)   # => "annoy: remove() unsupported ..."
```

## 8. Tune the policy without touching code

```python
import ann_router as ar

# Raise the exact->ANN crossover to 100k for this call:
choice = ar.route(ar.Criteria(n_vectors=50_000, dim=128),
                  thresholds={"EXACT_MAX_N": 100_000})
print(choice.backend)   # => exact
```

Or export `ANN_ROUTER_POLICY=/path/to/policy.yaml` to override the shipped
thresholds process-wide.

## 9. From the command line

```bash
# Decide + explain
ann-router route --n-vectors 2000000 --dim 768 --dynamic --markdown

# Build an index from a .npy file, then search it
python -c "import numpy as np; np.save('c.npy', np.random.rand(20000,128).astype('float32'))"
ann-router build --n-vectors 20000 --dim 128 --vectors c.npy --index my.idx
python -c "import numpy as np; np.save('q.npy', np.random.rand(3,128).astype('float32'))"
ann-router search --index my.idx --queries q.npy -k 10

# Benchmark installed backends vs exact ground truth
ann-router bench --n 5000 --dim 128 -k 10
```
