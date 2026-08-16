"""Annoy backend: read-only, memory-mapped, very lean.

Annoy (Spotify) builds a forest of random-projection trees, freezes it, and
memory-maps it from disk, so many processes share one on-disk index at almost
zero RAM cost. That frugality is the whole point: the router picks Annoy for a
**read-only corpus under a tight memory budget**. The flip side is rigidity:
once ``build`` is called the index is immutable, so ``add``/``remove`` genuinely
cannot work and honestly raise :class:`NotSupported` rather than pretending.

Consumes: ``annoy`` (optional, ``pip install 'ann-router[annoy]'``).
Produces: :class:`AnnoyIndex`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities, NotSupported
from ..spec import MetricName

# Annoy's metric vocabulary: cosine similarity is served by its "angular" tree;
# euclidean and dot map directly.
_ANNOY_METRIC = {"cosine": "angular", "l2": "euclidean", "ip": "dot"}


def _require():
    """Import annoy or raise an actionable :class:`BackendUnavailable`."""
    try:
        from annoy import AnnoyIndex as _Native  # local import keeps core import clean
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise BackendUnavailable(
            "annoy not installed. Run: pip install 'ann-router[annoy]'"
        ) from exc
    return _Native


class AnnoyIndex(ANNIndex):
    """Annoy random-projection-forest index: build once, then read-only.

    Because Annoy has no external-id concept, this adapter keeps its own
    ``position -> id`` table so ``search`` returns the caller's ids. External
    ids must therefore be supplied at ``build`` time and are fixed thereafter.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"`` (Annoy "angular").
    n_trees : int, optional
        Number of projection trees (more == better recall, larger index).
        Defaults to 50.
    search_k : int, optional
        Nodes inspected at query time (``-1`` == Annoy's ``n_trees * k`` default).

    Examples
    --------
    >>> AnnoyIndex.capabilities().supports_remove
    False
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._n_trees = int(kwargs.get("n_trees", 50))
        self._search_k = int(kwargs.get("search_k", -1))
        self._ids: np.ndarray = np.empty((0,), dtype=np.int64)

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the Annoy capability descriptor (frozen: no add/remove)."""
        return Capabilities(
            name="annoy",
            supports_add=False,  # immutable after build()
            supports_remove=False,  # genuinely impossible — see module docstring
            supports_filter=False,
            persistent=False,
            needs_gpu=False,
            approximate=True,
            metrics=("cosine", "l2", "ip"),
            pip_extra="annoy",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if annoy is importable.

        Examples
        --------
        >>> isinstance(AnnoyIndex.is_available(), bool)
        True
        """
        try:
            from annoy import AnnoyIndex  # noqa: F401
        except ImportError:
            return False
        return True

    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> AnnoyIndex:
        """Build and freeze the projection forest.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(n, dim)``.
        ids : numpy.ndarray, optional
            Shape ``(n,)``; defaults to ``range(n)``.

        Returns
        -------
        AnnoyIndex
            ``self``.
        """
        native = _require()
        arr = self._as_f32(vectors)
        index = native(self.dim, _ANNOY_METRIC[self.metric])
        # Annoy indexes by contiguous position 0..n-1; add each row, then keep a
        # parallel id table so search() can translate positions back to ids.
        for pos in range(arr.shape[0]):
            index.add_item(pos, arr[pos])
        index.build(self._n_trees)
        self._index = index
        self._ids = np.arange(arr.shape[0]) if ids is None else np.asarray(ids, dtype=np.int64)
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Not supported — Annoy is frozen after :meth:`build`.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``. Unused — always raises.

        Raises
        ------
        NotSupported
            Always; the forest is frozen after ``build()``.
        """
        raise NotSupported("annoy: add() unsupported — the forest is frozen after build()")

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Not supported — Annoy is frozen after :meth:`build`.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``. Unused — always raises.
        ids : numpy.ndarray
            Shape ``(m,)``. Unused — always raises.

        Raises
        ------
        NotSupported
            Always; the forest is frozen after ``build()``.
        """
        raise NotSupported("annoy: add_with_ids() unsupported — rebuild the index instead")

    def remove(self, ids: np.ndarray) -> None:
        """Not supported — Annoy cannot delete; rebuild without the ids.

        Parameters
        ----------
        ids : numpy.ndarray
            Shape ``(m,)``. Unused — always raises.

        Raises
        ------
        NotSupported
            Always; Annoy has no delete operation.
        """
        raise NotSupported("annoy: remove() unsupported — rebuild the index without those ids")

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
        arr = self._as_f32(queries)
        out_ids, out_dists = [], []
        for q in arr:
            # get_nns_by_vector returns positions + distances; translate the
            # positions through our id table row by row.
            pos, dist = self._index.get_nns_by_vector(  # type: ignore[union-attr]
                q, k, search_k=self._search_k, include_distances=True
            )
            out_ids.append(self._ids[pos])
            out_dists.append(dist)
        ids = np.array(out_ids, dtype=np.int64)
        dists = np.array(out_dists, dtype=np.float32)
        # A corpus smaller than k gives Annoy fewer than k columns; pad back to
        # the (q, k) contract every backend promises (see ANNIndex.search).
        return self._pad(ids, k), self._pad(dists, k, fill=np.inf)

    def save(self, path: str) -> None:
        """Persist the forest and the id table.

        The Annoy file itself has no room for external ids, so the id table is
        written alongside as ``<path>.ids.npy``.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        self._index.save(path)  # type: ignore[union-attr]
        np.save(path + ".ids.npy", self._ids)

    def load(self, path: str) -> AnnoyIndex:
        """Memory-map a forest written by :meth:`save`.

        Parameters
        ----------
        path : str
            Source path produced by :meth:`save`.

        Returns
        -------
        AnnoyIndex
            ``self``, populated from disk.
        """
        native = _require()
        index = native(self.dim, _ANNOY_METRIC[self.metric])
        index.load(path)  # mmaps the file — this is Annoy's low-RAM superpower
        self._index = index
        self._ids = np.load(path + ".ids.npy")
        return self
