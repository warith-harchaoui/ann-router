"""Shared CLI business logic: one core behind both the argparse and click doors.

The house rule (from the sibling repos' CODING.md) is that every surface (the
argparse CLI, the click twin, the FastAPI app, the MCP server) must call the
*same* library functions, never re-implement the decision or index logic. This
module is that shared core: thin functions that take already-parsed, primitive
arguments and return JSON-ready dicts (or persist/read an index), so each front
end is a trivial adapter over these.

Consumes: ``ann_router`` (route/auto_index), numpy, ``os_helper`` (io/logging).
Produces: :func:`do_route`, :func:`do_build`, :func:`do_search`, :func:`do_bench`,
:func:`do_capabilities`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import json
import time

import numpy as np
import os_helper as osh

from .registry import BACKENDS, all_capabilities, available_backends
from .router import auto_index, route, to_markdown
from .spec import Criteria


def do_capabilities() -> dict:
    """Return the availability + capability matrix of every backend.

    Returns
    -------
    dict
        ``{"available": [...], "backends": {name: capability_dict}}``.

    Examples
    --------
    >>> "exact" in do_capabilities()["available"]
    True
    """
    from .config import backend_catalog

    # Merge the machine-readable capability flags (from code) with the prose
    # catalog (from backends.yaml) so the user sees both "what it can do" and
    # "when to reach for it" in one payload.
    catalog = {entry["name"]: entry for entry in backend_catalog()}
    backends = {}
    for name, cap in all_capabilities().items():
        merged = cap.to_dict()
        merged["summary"] = catalog.get(name, {}).get("summary", "")
        merged["when"] = catalog.get(name, {}).get("when", "")
        backends[name] = merged
    return {"available": available_backends(), "backends": backends}


def do_route(criteria: dict, as_markdown: bool = False) -> dict | str:
    """Route a criteria mapping and return the decision (dict or Markdown).

    Parameters
    ----------
    criteria : dict
        Keys matching :class:`ann_router.spec.Criteria` (``n_vectors`` and
        ``dim`` required).
    as_markdown : bool, optional
        Return the human-readable Markdown report instead of the dict.

    Returns
    -------
    dict or str
        The :meth:`BackendChoice.to_dict` payload, or the Markdown string.

    Examples
    --------
    >>> do_route({"n_vectors": 500, "dim": 16})["backend"]
    'exact'
    """
    choice = route(Criteria.from_dict(criteria))
    return to_markdown(choice) if as_markdown else choice.to_dict()


def _load_vectors(path: str) -> np.ndarray:
    """Load an ``.npy`` matrix of vectors, coerced to float32.

    Parameters
    ----------
    path : str
        Path to an ``.npy`` file of shape ``(n, dim)``.

    Returns
    -------
    numpy.ndarray
        The matrix, cast to float32.
    """
    # np.load handles the common .npy case; callers pass a matrix (n, dim).
    return np.load(path).astype(np.float32)


def do_build(
    vectors_path: str, index_path: str, criteria: dict, backend: str | None = None
) -> dict:
    """Route (or force) a backend, build an index from a file, and persist it.

    Parameters
    ----------
    vectors_path : str
        Path to an ``.npy`` corpus of shape ``(n, dim)``.
    index_path : str
        Destination path for the built index (a sidecar ``.meta.json`` is
        written next to it so :func:`do_search` can reload the right backend).
    criteria : dict
        Routing criteria; ``n_vectors``/``dim`` are filled from the array.
    backend : str, optional
        Force a specific backend, bypassing the router.

    Returns
    -------
    dict
        ``{"backend", "rationale", "index_path", "n", "dim", "build_seconds"}``.

    Examples
    --------
    >>> import numpy as np, tempfile, os
    >>> d = tempfile.mkdtemp()
    >>> vp = os.path.join(d, "v.npy"); np.save(vp, np.random.rand(300, 8).astype("float32"))
    >>> out = do_build(vp, os.path.join(d, "idx"), {"n_vectors": 300, "dim": 8})
    >>> out["backend"]
    'exact'
    """
    vectors = _load_vectors(vectors_path)
    # The array is the source of truth for shape; overlay it on the criteria.
    crit = {**criteria, "n_vectors": vectors.shape[0], "dim": vectors.shape[1]}
    c = Criteria.from_dict(crit)
    start = time.perf_counter()
    if backend:
        # Forced path: skip routing but still record why (explicit override).
        cls = BACKENDS[backend]
        if not cls.is_available():
            raise RuntimeError(f"backend '{backend}' is not installed")
        index = cls(dim=vectors.shape[1], metric=c.metric)
        index.build(vectors)
        rationale = f"forced by --backend {backend}"
    else:
        index, choice = auto_index(vectors, c)
        backend = choice.backend
        rationale = choice.rationale
    elapsed = time.perf_counter() - start
    index.save(index_path)
    # Sidecar metadata lets `search` rebuild the same adapter without guessing.
    meta = {"backend": backend, "dim": int(vectors.shape[1]), "metric": c.metric}
    osh.info(f"ann-router: built '{backend}' index at {index_path} in {elapsed:.3f}s")
    with open(index_path + ".meta.json", "w") as fh:
        json.dump(meta, fh)
    return {
        "backend": backend,
        "rationale": rationale,
        "index_path": index_path,
        "n": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
        "build_seconds": round(elapsed, 4),
    }


def do_search(index_path: str, queries_path: str, k: int) -> dict:
    """Reload a persisted index and return the top-``k`` neighbours per query.

    Parameters
    ----------
    index_path : str
        Path passed to :func:`do_build` (its ``.meta.json`` sidecar must exist).
    queries_path : str
        Path to an ``.npy`` query matrix of shape ``(q, dim)``.
    k : int
        Neighbours per query.

    Returns
    -------
    dict
        ``{"backend", "ids": [[...]], "distances": [[...]]}``.
    """
    with open(index_path + ".meta.json") as fh:
        meta = json.load(fh)
    # Reconstruct the exact adapter the index was built with, then load it.
    cls = BACKENDS[meta["backend"]]
    index = cls(dim=meta["dim"], metric=meta["metric"]).load(index_path)
    queries = _load_vectors(queries_path)
    ids, dists = index.search(queries, k)
    return {
        "backend": meta["backend"],
        "ids": ids.tolist(),
        "distances": dists.tolist(),
    }


def do_bench(n: int = 5000, dim: int = 128, k: int = 10, seed: int = 20260804) -> dict:
    """Benchmark every available backend's recall@k against the exact truth.

    Builds a synthetic clustered corpus (realistic embedding-like structure),
    computes exact ground truth, then measures each installed backend's
    recall@k and build/query time — the same methodology as the ``roitelet``
    bench harness this package generalises.

    Parameters
    ----------
    n : int, optional
        Corpus size. Defaults to 5000.
    dim : int, optional
        Dimensionality. Defaults to 128.
    k : int, optional
        Neighbours per query. Defaults to 10.
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``{"config": {...}, "results": {backend: {recall, build_s, query_ms}}}``.

    Examples
    --------
    >>> out = do_bench(n=1200, dim=32, k=5)
    >>> out["results"]["exact"]["recall"]
    1.0
    """
    from .backends.exact import ExactIndex

    rng = np.random.default_rng(seed)
    # Clustered data mimics real embeddings far better than uniform noise, where
    # every method (including exact's "truth") would be near-degenerate.
    centers = rng.standard_normal((max(2, n // 160), dim)).astype(np.float32)
    corpus = (
        centers[rng.integers(0, len(centers), n)] + 0.4 * rng.standard_normal((n, dim))
    ).astype(np.float32)
    queries = (
        centers[rng.integers(0, len(centers), 50)] + 0.4 * rng.standard_normal((50, dim))
    ).astype(np.float32)
    truth, _ = ExactIndex(dim=dim).build(corpus).search(queries, k)

    def recall(ids: np.ndarray) -> float:
        """Mean fraction of the exact top-k that a backend's ``ids`` also returned.

        Parameters
        ----------
        ids : numpy.ndarray
            Shape ``(nq, k)`` a backend's predicted neighbour ids.

        Returns
        -------
        float
            Mean recall@k against ``truth``.
        """
        return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(truth, ids, strict=False)]))

    results: dict[str, dict] = {}
    for name in available_backends():
        cls = BACKENDS[name]
        # pgvector needs a live server; skip it in the offline synthetic bench.
        if name == "pgvector":
            continue
        t0 = time.perf_counter()
        index = cls(dim=dim).build(corpus)
        build_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        ids, _ = index.search(queries, k)
        query_ms = (time.perf_counter() - t1) / len(queries) * 1e3
        results[name] = {
            "recall": round(recall(ids), 3),
            "build_s": round(build_s, 3),
            "query_ms": round(query_ms, 3),
        }
    return {"config": {"n": n, "dim": dim, "k": k, "seed": seed}, "results": results}
