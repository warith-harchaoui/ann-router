"""HNSW backend (hnswlib) — best recall/latency for a stable, in-memory corpus.

Hierarchical Navigable Small World graphs give the best recall-per-millisecond
of the in-memory engines *when the corpus rarely changes*: the graph is built
once and traversed cheaply. The catch — and the reason the router only picks it
for **static** corpora — is deletion: hnswlib deletes via tombstones
(``mark_deleted``), which degrade the graph over time and are never truly
reclaimed without a rebuild. So this adapter advertises ``supports_remove`` as
tombstone-only and the policy keeps dynamic workloads on turbovec instead.

Consumes: ``hnswlib`` (optional, ``pip install 'ann-router[hnsw]'``).
Produces: :class:`HNSWIndex`.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities
from ..spec import MetricName

# hnswlib names cosine "cosine", euclidean "l2" and inner product "ip"; our
# metric vocabulary already matches, so this map is the identity but explicit.
_SPACE = {"cosine": "cosine", "l2": "l2", "ip": "ip"}


def _require():
    """Import hnswlib or raise an actionable :class:`BackendUnavailable`."""
    try:
        import hnswlib  # local import keeps `import ann_router` dependency-free
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise BackendUnavailable(
            "hnswlib not installed. Run: pip install 'ann-router[hnsw]'"
        ) from exc
    return hnswlib


class HNSWIndex(ANNIndex):
    """hnswlib-backed graph index tuned for high recall on a fixed corpus.

    Build knobs (``M``, ``ef_construction``) and the query knob (``ef``) are
    passed through and default to values that hit ~0.95+ recall on typical
    768-d embeddings. The index is grown to ``max_elements`` lazily and doubled
    on overflow so streaming ``add`` still works within the "stable corpus"
    caveat.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``.
    M : int, optional
        Graph out-degree. Defaults to 16.
    ef_construction : int, optional
        Build-time search width. Defaults to 200.
    ef : int, optional
        Query-time search width (recall/latency trade). Defaults to 64.

    Examples
    --------
    >>> HNSWIndex.capabilities().name
    'hnsw'
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._M = int(kwargs.get("M", 16))
        self._ef_construction = int(kwargs.get("ef_construction", 200))
        self._ef = int(kwargs.get("ef", 64))
        self._capacity = 0  # current max_elements; grown on demand

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the HNSW capability descriptor (remove is tombstone-only)."""
        return Capabilities(
            name="hnsw",
            supports_add=True,
            supports_remove=True,  # tombstones only — see module docstring
            supports_filter=False,
            persistent=False,
            needs_gpu=False,
            approximate=True,
            metrics=("cosine", "l2", "ip"),
            pip_extra="hnsw",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if hnswlib is importable.

        Examples
        --------
        >>> isinstance(HNSWIndex.is_available(), bool)
        True
        """
        try:
            import hnswlib  # noqa: F401
        except ImportError:
            return False
        return True

    def _new_index(self, capacity: int):
        """Allocate a fresh hnswlib index sized for ``capacity`` elements."""
        hnswlib = _require()
        index = hnswlib.Index(space=_SPACE[self.metric], dim=self.dim)
        index.init_index(
            max_elements=capacity, ef_construction=self._ef_construction, M=self._M
        )
        index.set_ef(self._ef)
        self._capacity = capacity
        return index

    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> HNSWIndex:
        """Build the graph from an initial corpus."""
        arr = self._as_f32(vectors)
        n = arr.shape[0]
        # Over-allocate 2x so the first stream of adds does not trigger a resize.
        self._index = self._new_index(max(n * 2, 1024))
        labels = np.arange(n) if ids is None else np.asarray(ids)
        self._index.add_items(arr, labels)  # type: ignore[union-attr]
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors with the next contiguous ids."""
        cur = self._index.get_current_count()  # type: ignore[union-attr]
        self.add_with_ids(vectors, np.arange(cur, cur + len(vectors)))

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids, growing capacity if needed."""
        arr = self._as_f32(vectors)
        need = self._index.get_current_count() + arr.shape[0]  # type: ignore[union-attr]
        if need > self._capacity:
            # hnswlib can resize in place; double to amortise repeated growth.
            self._index.resize_index(max(need, self._capacity * 2))  # type: ignore[union-attr]
            self._capacity = max(need, self._capacity * 2)
        self._index.add_items(arr, np.asarray(ids))  # type: ignore[union-attr]

    def remove(self, ids: np.ndarray) -> None:
        """Tombstone the given ids (graph is not reclaimed — rebuild for that)."""
        for i in np.asarray(ids).tolist():
            # mark_deleted excludes the label from results but keeps the node in
            # the graph; this is the documented HNSW deletion limitation.
            self._index.mark_deleted(int(i))  # type: ignore[union-attr]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate top-``k`` neighbours per query."""
        arr = self._as_f32(queries)
        count = self._index.get_current_count()  # type: ignore[union-attr]
        labels, distances = self._index.knn_query(arr, k=min(k, count))  # type: ignore[union-attr]
        return labels.astype(np.int64), distances.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist the graph via hnswlib's native serialiser."""
        self._index.save_index(path)  # type: ignore[union-attr]

    def load(self, path: str) -> HNSWIndex:
        """Load a graph written by :meth:`save`."""
        hnswlib = _require()
        index = hnswlib.Index(space=_SPACE[self.metric], dim=self.dim)
        index.load_index(path)
        index.set_ef(self._ef)
        self._index = index
        self._capacity = index.get_max_elements()
        return self
