"""Exact brute-force backend: the always-available reference implementation.

Pure numpy: no optional dependency, so ``ExactIndex`` is the one backend that
is *always* usable and doubles as the ground truth the router's recall tests
score every approximate engine against. For a corpus under a few tens of
thousands of vectors a vectorised full scan is already sub-millisecond and,
being exact, has recall 1.0 by construction, which is exactly why the policy
routes small problems here instead of paying an index-build cost for nothing.

Consumes: ``ann_router.base`` (the ANNIndex contract), numpy.
Produces: :class:`ExactIndex`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, Capabilities
from ..spec import MetricName


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """Return row-wise L2-normalised copies of ``vectors``.

    Parameters
    ----------
    vectors : numpy.ndarray
        Shape ``(n, dim)`` float32.

    Returns
    -------
    numpy.ndarray
        Same shape, each row unit-norm; zero rows are left as zeros.

    Examples
    --------
    >>> _normalise(np.array([[3.0, 4.0]], dtype=np.float32))
    array([[0.6, 0.8]], dtype=float32)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against division by zero for all-zero rows (norm 0 -> keep as 0).
    norms[norms == 0] = 1.0
    return vectors / norms


class ExactIndex(ANNIndex):
    """Exact k-NN by full matrix multiplication over the stored corpus.

    The corpus is kept as a contiguous float32 matrix; for cosine/inner-product
    metrics search is a single ``queries @ corpus.T`` followed by a top-k
    partition, and for L2 it is the standard ``||q||^2 - 2 q·x + ||x||^2``
    expansion. All operations are exact, so this class defines "truth" for the
    recall benchmarks.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> vecs = rng.standard_normal((100, 16)).astype(np.float32)
    >>> idx = ExactIndex(dim=16).build(vecs)
    >>> ids, dists = idx.search(vecs[:1], k=1)
    >>> int(ids[0, 0])              # nearest neighbour of a point is itself
    0
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        # The corpus matrix and its parallel id vector; both grow with add().
        self._vectors: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self._ids: np.ndarray = np.empty((0,), dtype=np.int64)
        # Cosine is implemented as inner product on pre-normalised vectors, so
        # we remember whether to normalise on ingest and query.
        self._cosine = metric == "cosine"

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the exact backend's capability descriptor.

        Returns
        -------
        Capabilities
            Exact, mutable, in-memory, no GPU, no metadata filter.

        Examples
        --------
        >>> ExactIndex.capabilities().approximate
        False
        """
        return Capabilities(
            name="exact",
            supports_add=True,
            supports_remove=True,
            supports_filter=False,
            persistent=False,
            needs_gpu=False,
            approximate=False,
            metrics=("cosine", "l2", "ip"),
            pip_extra="",  # always available: numpy is a core dependency
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` — numpy is a core dependency, so exact always works.

        Returns
        -------
        bool
            Always ``True``.

        Examples
        --------
        >>> ExactIndex.is_available()
        True
        """
        return True

    def _ingest(self, vectors: np.ndarray) -> np.ndarray:
        """Coerce and (for cosine) normalise vectors on the way into the store.

        Parameters
        ----------
        vectors : numpy.ndarray
            Any 2-D array of vectors.

        Returns
        -------
        numpy.ndarray
            Contiguous float32, unit-norm per row when the metric is cosine.
        """
        arr = self._as_f32(vectors)
        # Cosine similarity == dot product once both sides are unit-norm, so we
        # normalise once at ingest and reuse it for every future query.
        return _normalise(arr) if self._cosine else arr

    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> ExactIndex:
        """Store the initial corpus.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(n, dim)``.
        ids : numpy.ndarray, optional
            Shape ``(n,)``; defaults to ``range(n)``.

        Returns
        -------
        ExactIndex
            ``self``.

        Examples
        --------
        >>> idx = ExactIndex(dim=4).build(np.ones((5, 4), dtype=np.float32))
        >>> idx.search(np.ones((1, 4), dtype=np.float32), k=3)[0].shape
        (1, 3)
        """
        self._vectors = self._ingest(vectors)
        n = self._vectors.shape[0]
        # Default ids are the row positions; explicit ids are copied verbatim.
        self._ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors with the next contiguous ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.

        Examples
        --------
        >>> idx = ExactIndex(dim=4).build(np.ones((2, 4), dtype=np.float32))
        >>> idx.add(np.ones((3, 4), dtype=np.float32))
        >>> idx.size
        5
        """
        # New ids continue past the current maximum so they stay unique.
        start = int(self._ids.max()) + 1 if self._ids.size else 0
        new_ids = np.arange(start, start + len(vectors), dtype=np.int64)
        self.add_with_ids(vectors, new_ids)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        ids : numpy.ndarray
            Shape ``(m,)``.

        Examples
        --------
        >>> idx = ExactIndex(dim=4).build(np.ones((2, 4), dtype=np.float32))
        >>> idx.add_with_ids(np.zeros((1, 4), dtype=np.float32), np.array([99]))
        >>> idx.size
        3
        """
        arr = self._ingest(vectors)
        # Grow both parallel arrays together so row i always matches id i.
        self._vectors = np.vstack([self._vectors, arr]) if self._vectors.size else arr
        self._ids = np.concatenate([self._ids, np.asarray(ids, dtype=np.int64)])

    def remove(self, ids: np.ndarray) -> None:
        """Delete vectors by id (true deletion — the rows are dropped).

        Parameters
        ----------
        ids : numpy.ndarray
            Ids to remove.

        Examples
        --------
        >>> idx = ExactIndex(dim=4).build(np.ones((3, 4), dtype=np.float32))
        >>> idx.remove(np.array([1]))
        >>> idx.size
        2
        """
        # Boolean mask of survivors keeps the two parallel arrays aligned.
        drop = np.isin(self._ids, np.asarray(ids, dtype=np.int64))
        self._vectors = self._vectors[~drop]
        self._ids = self._ids[~drop]

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return exact top-``k`` neighbours per query.

        Parameters
        ----------
        queries : numpy.ndarray
            Shape ``(q, dim)``.
        k : int
            Neighbours per query.

        Returns
        -------
        ids : numpy.ndarray
            Shape ``(q, k)`` neighbour ids (``-1`` where fewer than ``k`` exist).
        distances : numpy.ndarray
            Shape ``(q, k)`` distances (similarity for cosine/ip, euclidean^2 for l2).

        Examples
        --------
        >>> rng = np.random.default_rng(1)
        >>> vecs = rng.standard_normal((50, 8)).astype(np.float32)
        >>> ids, _ = ExactIndex(dim=8).build(vecs).search(vecs[:3], k=5)
        >>> ids.shape
        (3, 5)
        """
        q = self._ingest(queries) if self._cosine else self._as_f32(queries)
        n = self._vectors.shape[0]
        kk = min(k, n)
        if self.metric in ("cosine", "ip"):
            # Higher dot product == closer; negate so "smallest" == "nearest"
            # and we can share the argpartition path with L2.
            sims = q @ self._vectors.T
            order, scores = self._topk(-sims, kk)
            dists = -scores  # report the raw similarity back to the caller
        else:
            # Squared euclidean via the (a-b)^2 expansion; the -2 q·x term is
            # the only per-pair work, the norms are precomputed broadcasts.
            corpus_sq = np.einsum("ij,ij->i", self._vectors, self._vectors)
            query_sq = np.einsum("ij,ij->i", q, q)[:, None]
            d2 = query_sq - 2.0 * (q @ self._vectors.T) + corpus_sq[None, :]
            order, dists = self._topk(d2, kk)
        ids = self._ids[order]
        return self._pad(ids, k), self._pad(dists, k, fill=np.inf)

    def _topk(self, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the indices and values of the ``k`` smallest scores per row.

        Parameters
        ----------
        scores : numpy.ndarray
            Shape ``(q, n)`` per-query, per-candidate scores (lower = closer).
        k : int
            Number of smallest scores to keep per row.

        Returns
        -------
        idx : numpy.ndarray
            Shape ``(q, k)`` column indices of the ``k`` smallest, sorted ascending.
        numpy.ndarray
            Shape ``(q, k)`` the corresponding scores.
        """
        # argpartition is O(n) and gives an unsorted k-set; we then sort only
        # those k, which is the standard fast top-k idiom.
        part = np.argpartition(scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(scores.shape[0])[:, None]
        vals = scores[row, part]
        finish = np.argsort(vals, axis=1)
        idx = part[row, finish]
        return idx, scores[row, idx]

    @property
    def size(self) -> int:
        """Number of vectors currently stored.

        Examples
        --------
        >>> ExactIndex(dim=4).build(np.ones((7, 4), dtype=np.float32)).size
        7
        """
        return int(self._vectors.shape[0])

    def save(self, path: str) -> None:
        """Persist the corpus and ids to a ``.npz`` file.

        Parameters
        ----------
        path : str
            Destination path (``.npz`` appended by numpy if absent).

        Examples
        --------
        >>> import tempfile, os
        >>> idx = ExactIndex(dim=4).build(np.ones((3, 4), dtype=np.float32))
        >>> p = os.path.join(tempfile.mkdtemp(), "exact.npz")
        >>> idx.save(p); os.path.exists(p) or os.path.exists(p + ".npz")
        True
        """
        # np.savez_compressed keeps the two arrays plus the metric so load()
        # can reconstruct an identical index without external context.
        np.savez_compressed(
            path, vectors=self._vectors, ids=self._ids, metric=np.array(self.metric)
        )

    def load(self, path: str) -> ExactIndex:
        """Load a corpus previously written by :meth:`save`.

        Parameters
        ----------
        path : str
            Source ``.npz`` path.

        Returns
        -------
        ExactIndex
            ``self``, populated from disk.
        """
        data = np.load(path if path.endswith(".npz") else path + ".npz", allow_pickle=False)
        # Vectors were already normalised at save time for cosine, so we assign
        # them directly rather than re-ingesting (which would double-normalise).
        self._vectors = data["vectors"].astype(np.float32)
        self._ids = data["ids"].astype(np.int64)
        return self
