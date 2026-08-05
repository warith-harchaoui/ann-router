"""Deterministic synthetic corpora and cached exact ground truth.

The calibration harness needs, for a given ``(n, dim, seed)``, a reproducible
vector corpus plus the *exact* k-nearest-neighbour answers for a fixed query set
— the yardstick every approximate backend's recall is scored against.

Design choices that keep a 10M-scale sweep affordable and resumable:

* **Corpora are regenerated, never stored.** ``n * dim * 4`` bytes of float32 is
  cheap to synthesise from a seed and ruinous to keep on disk at 10M, so we hold
  a corpus in RAM only while a cell runs and gate on a memory budget beforehand.
* **Ground truth is cached.** The queries (``q * dim``) and their exact neighbour
  ids (``q * k``) are tiny, so they are written once under ``results/gt/`` and
  reused across every backend and recall target at the same ``(n, dim, ...)``.
* **Clustered data**, not uniform noise — real embeddings live on clusters, and
  ANN recall behaves very differently on clustered vs. uniform data.

Consumes: numpy, faiss (for the exact search; falls back to chunked numpy).
Produces: :func:`make_corpus`, :func:`make_queries`, :func:`ground_truth`.

Author: Warith Harchaoui
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GT_DIR = Path(__file__).resolve().parent / "results" / "gt"


# Number of latent clusters the corpus is drawn around, as a function of size.
# ~sqrt(n) clusters keeps a roughly constant per-cluster population as n grows.
def _n_clusters(n: int) -> int:
    return int(max(8, min(4096, round(n**0.5))))


def _rng(seed: int, salt: int) -> np.random.Generator:
    """Return a generator seeded reproducibly from ``(seed, salt)``."""
    # SeedSequence mixes the two so corpus and query streams never coincide.
    return np.random.default_rng(np.random.SeedSequence([seed, salt]))


def corpus_bytes(n: int, dim: int) -> int:
    """Return the float32 RAM footprint of an ``(n, dim)`` corpus, in bytes."""
    return n * dim * 4


def make_corpus(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Synthesise a deterministic clustered, unit-norm float32 corpus.

    Parameters
    ----------
    n : int
        Number of vectors.
    dim : int
        Dimensionality.
    seed : int, optional
        Base seed; the same ``(n, dim, seed)`` always yields the same array.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, dim)``, dtype float32, each row L2-normalised (so inner
        product equals cosine similarity).
    """
    rng = _rng(seed, salt=1)
    c = _n_clusters(n)
    centres = rng.standard_normal((c, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True) + 1e-12
    assign = rng.integers(0, c, size=n)
    # Tight-ish Gaussian blobs around each centre; 0.15 spread leaves clusters
    # separable enough that approximation is meaningful but not trivial.
    data = centres[assign] + 0.15 * rng.standard_normal((n, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(data, dtype=np.float32)


def make_queries(n: int, dim: int, nq: int, seed: int = 0) -> np.ndarray:
    """Synthesise ``nq`` deterministic unit-norm query vectors.

    Queries are drawn from the same cluster structure as the corpus (so they
    have genuine near neighbours) but on an independent random stream.

    Parameters
    ----------
    n, dim : int
        Corpus shape the queries are meant to probe (drives cluster count).
    nq : int
        Number of queries.
    seed : int, optional
        Base seed.

    Returns
    -------
    numpy.ndarray
        Shape ``(nq, dim)``, float32, unit-norm.
    """
    rng = _rng(seed, salt=2)
    c = _n_clusters(n)
    centres_rng = _rng(seed, salt=1)  # same stream as make_corpus for centres
    centres = centres_rng.standard_normal((c, dim)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True) + 1e-12
    pick = rng.integers(0, c, size=nq)
    q = centres[pick] + 0.15 * rng.standard_normal((nq, dim)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(q, dtype=np.float32)


def _exact_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Return exact cosine top-``k`` neighbour ids, ``(nq, k)``.

    Uses a FAISS flat inner-product index (corpus is unit-norm, so IP == cosine)
    when available, else a chunked numpy scan that never materialises the full
    ``nq x n`` score matrix.
    """
    try:
        import faiss

        index = faiss.IndexFlatIP(corpus.shape[1])
        index.add(corpus)
        _, ids = index.search(queries, k)
        return ids.astype(np.int64)
    except Exception:
        # Chunked fallback: score queries against the corpus in row blocks.
        nq = queries.shape[0]
        out = np.empty((nq, k), dtype=np.int64)
        block = max(1, 4_000_000 // max(1, corpus.shape[0]))  # cap scores in RAM
        for i in range(0, nq, block):
            sims = queries[i : i + block] @ corpus.T
            out[i : i + block] = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        return out


def ground_truth(n: int, dim: int, k: int, nq: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(queries, gt_ids)``, computing and caching the exact answers.

    Parameters
    ----------
    n, dim, k, nq : int
        Corpus size, dimensionality, neighbours per query, and query count.
    seed : int, optional
        Base seed tying corpus, queries, and ground truth together.

    Returns
    -------
    queries : numpy.ndarray
        ``(nq, dim)`` float32 query vectors.
    gt_ids : numpy.ndarray
        ``(nq, k)`` int64 exact neighbour ids for those queries.

    Notes
    -----
    The corpus is built transiently to compute ground truth, then dropped. The
    result is memoised on disk under ``results/gt/`` so subsequent backends and
    recall targets reuse it without recomputation.
    """
    GT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"n{n}_d{dim}_k{k}_q{nq}_s{seed}"
    q_path, gt_path = GT_DIR / f"{tag}.queries.npy", GT_DIR / f"{tag}.gt.npy"
    if q_path.exists() and gt_path.exists():
        return np.load(q_path), np.load(gt_path)

    queries = make_queries(n, dim, nq, seed)
    corpus = make_corpus(n, dim, seed)
    gt = _exact_topk(corpus, queries, k)
    del corpus  # free the big array before returning
    np.save(q_path, queries)
    np.save(gt_path, gt)
    return queries, gt
