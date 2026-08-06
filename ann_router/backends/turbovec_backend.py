"""turbovec backend — the dynamic-corpus specialist.

turbovec (Rust + PyO3, ships wheels on PyPI incl. Apple Silicon) is the engine
the router reaches for when the corpus **changes constantly**: it supports O(1)
``add_with_ids`` and ``remove(id)`` with no index rebuild, and its TurboQuant
2-4 bit quantisation gives ~16x compression while keeping recall above
FAISS-PQ. That combination — mutable *and* compact *and* fast on Apple Silicon —
is what earns it the "frequent updates" branch of the policy, ahead of the
graph indexes whose deletes rot the structure.

This is the extracted, packaged form of the brute-force → turbovec routing that
already ships inside the ``roitelet`` prototype's ``core/personal.py``.

Consumes: ``turbovec`` (optional, ``pip install 'ann-router[turbovec]'``).
Produces: :class:`TurboVecIndex`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities
from ..spec import MetricName


def _require():
    """Import turbovec or raise an actionable :class:`BackendUnavailable`."""
    try:
        import turbovec  # local import so `import ann_router` never needs the Rust wheel
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise BackendUnavailable(
            "turbovec not installed. Run: pip install 'ann-router[turbovec]'"
        ) from exc
    return turbovec


class TurboVecIndex(ANNIndex):
    """turbovec ``IdMapIndex``: mutable, quantised, id-native.

    turbovec is id-native (``add_with_ids`` / ``remove(id)``) and returns
    ``(distances, ids)`` from ``search`` — this adapter flips that to the
    package's ``(ids, distances)`` order. Vectors are normalised for cosine so
    the quantiser's inner product matches cosine similarity.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``. turbovec is inner-product /
        cosine oriented; L2 is approximated on normalised vectors.
    bit_width : int, optional
        TurboQuant bit width (2 or 4). Defaults to 4 — the recall/size sweet
        spot measured in the roitelet study.

    Examples
    --------
    >>> TurboVecIndex.capabilities().supports_remove
    True
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._bit_width = int(kwargs.get("bit_width", 4))
        # Cosine == inner product on unit vectors; normalise on the way in.
        self._normalise = metric in ("cosine", "l2")

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the turbovec capability descriptor (fully mutable)."""
        return Capabilities(
            name="turbovec",
            supports_add=True,
            supports_remove=True,  # O(1) remove(id), the reason it exists
            supports_filter=False,  # allowlist exists but is not metadata filtering
            persistent=False,
            needs_gpu=False,
            approximate=True,
            metrics=("cosine", "ip", "l2"),
            pip_extra="turbovec",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if turbovec is importable.

        Examples
        --------
        >>> isinstance(TurboVecIndex.is_available(), bool)
        True
        """
        try:
            import turbovec  # noqa: F401
        except ImportError:
            return False
        return True

    def _prep(self, vectors: np.ndarray) -> np.ndarray:
        """Coerce to float32 and L2-normalise when the metric needs it.

        Parameters
        ----------
        vectors : numpy.ndarray
            Any 2-D array of vectors.

        Returns
        -------
        numpy.ndarray
            Contiguous float32, L2-normalised per row for cosine/l2 metrics.
        """
        arr = self._as_f32(vectors)
        if self._normalise:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> TurboVecIndex:
        """Create the index and insert the initial corpus.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(n, dim)``.
        ids : numpy.ndarray, optional
            Shape ``(n,)``; defaults to ``range(n)``.

        Returns
        -------
        TurboVecIndex
            ``self``.
        """
        turbovec = _require()
        self._index = turbovec.IdMapIndex(dim=self.dim, bit_width=self._bit_width)
        arr = self._prep(vectors)
        labels = np.arange(arr.shape[0]) if ids is None else np.asarray(ids)
        self._index.add_with_ids(arr, labels.astype(np.uint64))
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors with the next contiguous ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        """
        # turbovec has no count getter we rely on, so track the high-water mark
        # from the corpus we have inserted so far via a private counter.
        start = getattr(self, "_next_id", None)
        if start is None:
            start = 0
        self.add_with_ids(vectors, np.arange(start, start + len(vectors)))

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids (O(1), no rebuild).

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids.
        """
        if self._index is None:
            turbovec = _require()
            self._index = turbovec.IdMapIndex(dim=self.dim, bit_width=self._bit_width)
        arr = self._prep(vectors)
        labels = np.asarray(ids, dtype=np.uint64)
        self._index.add_with_ids(arr, labels)  # type: ignore[union-attr]
        # Remember the next free contiguous id for a subsequent bare add().
        self._next_id = int(labels.max()) + 1 if labels.size else getattr(self, "_next_id", 0)

    def remove(self, ids: np.ndarray) -> None:
        """Delete vectors by id — O(1) each, no structural degradation.

        Parameters
        ----------
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids to drop.
        """
        for i in np.asarray(ids, dtype=np.uint64).tolist():
            self._index.remove(int(i))  # type: ignore[union-attr]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate top-``k`` neighbours per query.

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
            Shape ``(q, k)`` distances under the index metric.
        """
        arr = self._prep(queries)
        # turbovec returns (distances, ids); the package contract is (ids, dist),
        # so we swap the two halves of the tuple here.
        distances, ids = self._index.search(arr, k)  # type: ignore[union-attr]
        return np.asarray(ids, dtype=np.int64), np.asarray(distances, dtype=np.float32)

    def save(self, path: str) -> None:
        """Persist via turbovec's native ``write``.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        self._index.write(path)  # type: ignore[union-attr]

    def load(self, path: str) -> TurboVecIndex:
        """Load an index written by :meth:`save`.

        Parameters
        ----------
        path : str
            Source path produced by :meth:`save`.

        Returns
        -------
        TurboVecIndex
            ``self``, populated from disk.
        """
        turbovec = _require()
        self._index = turbovec.IdMapIndex.load(path)
        return self
