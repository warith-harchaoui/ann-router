"""FAISS backend (IVF + PQ) — scale, quantisation, optional GPU.

FAISS earns its place at the *large volume + batch (+ GPU)* end of the policy:
an IVF coarse quantiser prunes the search to a few cells, and optional Product
Quantisation compresses vectors ~8-16x so billions fit in RAM. The trade is a
training step and lower recall than a graph index at small/medium N — which is
precisely why the router only reaches for FAISS once the corpus is big enough
for those wins to matter, and prefers HNSW/turbovec below that.

The adapter auto-scales ``nlist`` and, for very large corpora, switches from a
flat IVF to IVF-PQ. Ids are handled through an ``IndexIDMap2`` wrapper so
external ids survive.

Consumes: ``faiss`` (optional, ``pip install 'ann-router[faiss]'`` → faiss-cpu).
Produces: :class:`FaissIndex`.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities, NotSupported
from ..spec import MetricName


def _require():
    """Import faiss or raise an actionable :class:`BackendUnavailable`."""
    try:
        import faiss  # local import: faiss must never be needed to import ann_router
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise BackendUnavailable(
            "faiss not installed. Run: pip install 'ann-router[faiss]'  (faiss-cpu)"
        ) from exc
    return faiss


def _metric_flag(faiss, metric: MetricName) -> int:
    """Map our metric name to a FAISS metric constant (cosine == IP on unit norm)."""
    # FAISS has no cosine metric; cosine is inner product over L2-normalised
    # vectors, so we normalise on ingest and use METRIC_INNER_PRODUCT here.
    if metric in ("cosine", "ip"):
        return faiss.METRIC_INNER_PRODUCT
    return faiss.METRIC_L2


class FaissIndex(ANNIndex):
    """FAISS IVF(-PQ) index with id mapping and auto-sized coarse quantiser.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``.
    nlist : int, optional
        Number of IVF cells. Defaults to ``auto`` (``~sqrt(n)*4``, clamped).
    nprobe : int, optional
        Cells probed at query time (recall/latency trade). Defaults to 16.
    use_pq : bool or "auto", optional
        Enable Product Quantisation. ``"auto"`` (default) turns it on above
        ``pq_threshold`` vectors.
    m : int, optional
        PQ sub-quantiser count (must divide ``dim``). Defaults to a divisor
        near ``dim/2``.

    Examples
    --------
    >>> FaissIndex.capabilities().name
    'faiss'
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._nlist = kwargs.get("nlist", "auto")
        self._nprobe = int(kwargs.get("nprobe", 16))
        self._use_pq = kwargs.get("use_pq", "auto")
        self._m = kwargs.get("m")
        self._pq_threshold = int(kwargs.get("pq_threshold", 500_000))
        self._cosine = metric == "cosine"

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the FAISS capability descriptor (GPU-capable, add/remove ok)."""
        return Capabilities(
            name="faiss",
            supports_add=True,
            supports_remove=True,  # via IndexIDMap2.remove_ids
            supports_filter=False,
            persistent=False,
            needs_gpu=False,  # faiss-cpu works; GPU is a bonus, not required
            approximate=True,
            metrics=("cosine", "l2", "ip"),
            pip_extra="faiss",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if faiss is importable.

        Examples
        --------
        >>> isinstance(FaissIndex.is_available(), bool)
        True
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            return False
        return True

    def _prep(self, vectors: np.ndarray) -> np.ndarray:
        """Coerce to float32 and L2-normalise when the metric is cosine."""
        arr = self._as_f32(vectors)
        if self._cosine:
            faiss = _require()
            arr = arr.copy()
            faiss.normalize_L2(arr)  # in-place; copy first so we never mutate caller data
        return arr

    def _pick_pq_m(self) -> int:
        """Choose a PQ sub-quantiser count that divides ``dim``."""
        if self._m:
            return int(self._m)
        # Walk down from dim/2 to the first divisor of dim; PQ requires m | dim.
        for cand in range(max(1, self.dim // 2), 0, -1):
            if self.dim % cand == 0:
                return cand
        return 1

    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> FaissIndex:
        """Train and populate the IVF(-PQ) index."""
        faiss = _require()
        arr = self._prep(vectors)
        n = arr.shape[0]
        # Rule of thumb: nlist ~ 4*sqrt(n), but never more cells than we can
        # train (needs a few points per cell) nor fewer than 1.
        nlist = self._nlist
        if nlist == "auto":
            nlist = int(max(1, min(4 * int(np.sqrt(n)), n // 39 or 1)))
        metric = _metric_flag(faiss, self.metric)
        quantiser = faiss.IndexFlatIP(self.dim) if metric == faiss.METRIC_INNER_PRODUCT \
            else faiss.IndexFlatL2(self.dim)
        want_pq = self._use_pq is True or (self._use_pq == "auto" and n >= self._pq_threshold)
        if want_pq:
            # 8 bits per sub-quantiser is the FAISS default and a good recall
            # trade; m sub-vectors compress each vector to m bytes.
            base = faiss.IndexIVFPQ(quantiser, self.dim, nlist, self._pick_pq_m(), 8, metric)
        else:
            base = faiss.IndexIVFFlat(quantiser, self.dim, nlist, metric)
        base.train(arr)  # IVF needs a training pass to learn the cell centroids
        base.nprobe = self._nprobe
        # IndexIDMap2 lets us attach arbitrary external ids and remove by id.
        self._index = faiss.IndexIDMap2(base)
        labels = np.arange(n) if ids is None else np.asarray(ids)
        self._index.add_with_ids(arr, labels.astype(np.int64))
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors with the next contiguous ids."""
        cur = self._index.ntotal  # type: ignore[union-attr]
        self.add_with_ids(vectors, np.arange(cur, cur + len(vectors)))

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids (index must already be trained)."""
        if self._index is None:
            raise NotSupported("faiss: call build() before add (IVF needs training first)")
        arr = self._prep(vectors)
        self._index.add_with_ids(arr, np.asarray(ids, dtype=np.int64))  # type: ignore[union-attr]

    def remove(self, ids: np.ndarray) -> None:
        """Remove vectors by id via the id map."""
        faiss = _require()
        sel = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
        self._index.remove_ids(sel)  # type: ignore[union-attr]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate top-``k`` neighbours per query."""
        arr = self._prep(queries)
        distances, labels = self._index.search(arr, k)  # type: ignore[union-attr]
        return labels.astype(np.int64), distances.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist via faiss.write_index."""
        faiss = _require()
        faiss.write_index(self._index, path)

    def load(self, path: str) -> FaissIndex:
        """Load an index written by :meth:`save`."""
        faiss = _require()
        self._index = faiss.read_index(path)
        return self
