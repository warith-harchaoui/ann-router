"""Qdrant backend: persistent HNSW with first-class metadata filtering.

Qdrant is the router's answer to *persistence + metadata filters*: it stores an
HNSW index plus a JSON payload per point and filters by that payload at query
time, and it survives restarts (embedded on-disk, or a remote server). When a
workload needs "give me the nearest vectors **where** ``lang == 'fr'``", the
in-memory engines cannot help and the policy routes here. This adapter defaults
to the embedded in-memory/on-disk client so tests need no running server, and
exposes a ``search_filter`` extension for the payload path.

Consumes: ``qdrant-client`` (optional, ``pip install 'ann-router[qdrant]'``).
Produces: :class:`QdrantIndex`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities
from ..spec import MetricName

# Qdrant distance enum names keyed by our metric vocabulary.
_DISTANCE = {"cosine": "Cosine", "l2": "Euclid", "ip": "Dot"}


def _require():
    """Import qdrant-client or raise an actionable :class:`BackendUnavailable`."""
    try:
        from qdrant_client import QdrantClient, models  # local import
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise BackendUnavailable(
            "qdrant-client not installed. Run: pip install 'ann-router[qdrant]'"
        ) from exc
    return QdrantClient, models


class QdrantIndex(ANNIndex):
    """Qdrant collection wrapper (embedded by default) with payload filtering.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``.
    location : str, optional
        ``":memory:"`` (default, embedded), a directory path (embedded on-disk),
        or a URL for a remote server.
    collection : str, optional
        Collection name. Defaults to ``"ann_router"``.

    Examples
    --------
    >>> QdrantIndex.capabilities().supports_filter
    True
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._location = str(kwargs.get("location", ":memory:"))
        self._collection = str(kwargs.get("collection", "ann_router"))
        self._payloads: dict[int, dict] = {}  # cached so add() can re-attach metadata

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the Qdrant capability descriptor (persistent + filterable)."""
        return Capabilities(
            name="qdrant",
            supports_add=True,
            supports_remove=True,
            supports_filter=True,  # the distinguishing feature
            persistent=True,
            needs_gpu=False,
            approximate=True,
            metrics=("cosine", "l2", "ip"),
            pip_extra="qdrant",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if qdrant-client is importable.

        Examples
        --------
        >>> isinstance(QdrantIndex.is_available(), bool)
        True
        """
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            return False
        return True

    def _client(self):
        """Return (creating once) the underlying Qdrant client handle."""
        if self._index is None:
            QdrantClient, _ = _require()
            self._index = QdrantClient(location=self._location)
        return self._index

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray | None = None,
        payloads: list[dict] | None = None,
    ) -> QdrantIndex:
        """Create the collection and upsert the initial corpus.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(n, dim)``.
        ids : numpy.ndarray, optional
            Point ids; defaults to ``range(n)``.
        payloads : list of dict, optional
            Per-point metadata for the filtering path.
        """
        QdrantClient, models = _require()
        client = self._client()
        arr = self._as_f32(vectors)
        # Drop any prior collection of this name, then create a fresh one — the
        # explicit two-step replaces the deprecated recreate_collection().
        if client.collection_exists(self._collection):
            client.delete_collection(self._collection)
        client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=self.dim, distance=getattr(models.Distance, _DISTANCE[self.metric].upper())
            ),
        )
        labels = np.arange(arr.shape[0]) if ids is None else np.asarray(ids)
        self._upsert(arr, labels, payloads)
        return self

    def _upsert(self, arr: np.ndarray, labels: np.ndarray, payloads: list[dict] | None) -> None:
        """Upsert a batch of points, caching payloads for later reads.

        Parameters
        ----------
        arr : numpy.ndarray
            Shape ``(m, dim)`` float32 vectors.
        labels : numpy.ndarray
            Shape ``(m,)`` integer point ids.
        payloads : list of dict, optional
            One JSON-serialisable payload per row, or ``None`` for empty payloads.
        """
        _, models = _require()
        points = []
        for row, pid in enumerate(labels.tolist()):
            payload = payloads[row] if payloads else {}
            self._payloads[int(pid)] = payload
            points.append(
                models.PointStruct(id=int(pid), vector=arr[row].tolist(), payload=payload)
            )
        self._index.upsert(collection_name=self._collection, points=points)  # type: ignore[union-attr]

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors with the next contiguous ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        """
        existing = max(self._payloads) + 1 if self._payloads else 0
        self.add_with_ids(vectors, np.arange(existing, existing + len(vectors)))

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids.
        """
        self._upsert(self._as_f32(vectors), np.asarray(ids), None)

    def remove(self, ids: np.ndarray) -> None:
        """Delete points by id.

        Parameters
        ----------
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids to drop.
        """
        _, models = _require()
        self._index.delete(  # type: ignore[union-attr]
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=[int(i) for i in np.asarray(ids)]),
        )
        for i in np.asarray(ids).tolist():
            self._payloads.pop(int(i), None)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate top-``k`` neighbours per query (no filter).

        Parameters
        ----------
        queries : numpy.ndarray
            Shape ``(q, dim)``.
        k : int
            Neighbours per query.

        Returns
        -------
        ids : numpy.ndarray
            Shape ``(q, k)`` neighbour ids.
        distances : numpy.ndarray
            Shape ``(q, k)`` scores.
        """
        return self.search_filter(queries, k, where=None)

    def search_filter(
        self, queries: np.ndarray, k: int, where: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-``k`` neighbours, optionally restricted by a payload filter.

        Parameters
        ----------
        queries : numpy.ndarray
            Shape ``(q, dim)``.
        k : int
            Neighbours per query.
        where : dict, optional
            ``{field: value}`` equality constraints ANDed together. ``None``
            (default) searches the whole collection.

        Returns
        -------
        ids : numpy.ndarray
            Shape ``(q, k)`` (``-1`` pads short rows when a filter is strict).
        distances : numpy.ndarray
            Shape ``(q, k)`` scores.
        """
        _, models = _require()
        arr = self._as_f32(queries)
        # Translate the flat equality dict into Qdrant's Filter/FieldCondition AST.
        flt = None
        if where:
            flt = models.Filter(
                must=[
                    models.FieldCondition(key=key, match=models.MatchValue(value=val))
                    for key, val in where.items()
                ]
            )
        ids_out, dist_out = [], []
        for q in arr:
            res = self._index.query_points(  # type: ignore[union-attr]
                collection_name=self._collection, query=q.tolist(), limit=k, query_filter=flt
            ).points
            row_ids = [int(p.id) for p in res]
            row_d = [float(p.score) for p in res]
            # Pad short result rows so the output stays a rectangular (q, k) array.
            row_ids += [-1] * (k - len(row_ids))
            row_d += [float("inf")] * (k - len(row_d))
            ids_out.append(row_ids)
            dist_out.append(row_d)
        return np.array(ids_out, dtype=np.int64), np.array(dist_out, dtype=np.float32)

    def save(self, path: str) -> None:
        """No-op for embedded on-disk / remote collections (already persistent).

        Parameters
        ----------
        path : str
            Unused — accepted only to satisfy the shared interface.

        Notes
        -----
        Qdrant persists itself when ``location`` is a directory or a server URL;
        the ``:memory:`` client is ephemeral by design. Point ``save`` at a
        directory ``location`` instead of calling this for durability.
        """
        # Nothing to flush explicitly — persistence is a property of `location`.
        return None

    def load(self, path: str) -> QdrantIndex:
        """Reconnect to an on-disk collection at ``path``.

        Parameters
        ----------
        path : str
            The on-disk ``location`` to reconnect to.

        Returns
        -------
        QdrantIndex
            ``self``, reconnected.
        """
        QdrantClient, _ = _require()
        self._location = path
        self._index = QdrantClient(location=path)
        return self
