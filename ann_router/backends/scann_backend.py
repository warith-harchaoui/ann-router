"""ScaNN backend — anisotropic quantisation for maximum recall at scale.

Google's ScaNN pairs a learned, anisotropic (score-aware) quantiser with
tree-based partitioning to push recall higher than plain PQ at the same speed,
which is why the router names it for the *max-recall-at-scale* corner. In
practice ScaNN ships wheels only for Linux/x86 (and a narrow set of Python
versions), so on Apple Silicon and many other machines it will simply be
unavailable — this adapter is written so that absence is a clean skip, never an
import error. The build/query calls follow ScaNN's fluent
``builder(...).tree(...).score_ah(...).build()`` API.

Consumes: ``scann`` (optional, ``pip install 'ann-router[scann]'``; Linux/x86 only).
Produces: :class:`ScannIndex`.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities, NotSupported
from ..spec import MetricName


def _require():
    """Import scann or raise an actionable :class:`BackendUnavailable`."""
    try:
        import scann  # local import: scann is Linux/x86-only and often absent
    except ImportError as exc:  # pragma: no cover - absent on macOS/arm64
        raise BackendUnavailable(
            "scann not installed (Linux/x86 only). Run: pip install 'ann-router[scann]'"
        ) from exc
    return scann


class ScannIndex(ANNIndex):
    """ScaNN partition + anisotropic-hash index, rebuilt in bulk.

    ScaNN searchers are immutable once built (like Annoy), so ``add``/``remove``
    raise :class:`NotSupported`; callers with churn should route to turbovec.
    External ids are tracked in a parallel table because ScaNN returns row
    positions.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"`` (ScaNN "dot_product" on
        normalised vectors).
    num_leaves : int, optional
        Partition count. Defaults to ``auto`` (``~sqrt(n)``).
    num_leaves_to_search : int, optional
        Leaves probed per query. Defaults to 100.
    reordering : int, optional
        Exact-reorder candidate count for a final recall boost. Defaults to 100.

    Examples
    --------
    >>> ScannIndex.capabilities().name
    'scann'
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._num_leaves = kwargs.get("num_leaves", "auto")
        self._leaves_to_search = int(kwargs.get("num_leaves_to_search", 100))
        self._reordering = int(kwargs.get("reordering", 100))
        self._ids: np.ndarray = np.empty((0,), dtype=np.int64)

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the ScaNN capability descriptor (immutable after build)."""
        return Capabilities(
            name="scann",
            supports_add=False,
            supports_remove=False,
            supports_filter=False,
            persistent=False,
            needs_gpu=False,
            approximate=True,
            metrics=("cosine", "ip", "l2"),
            pip_extra="scann",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if scann is importable (typically only on Linux/x86).

        Examples
        --------
        >>> isinstance(ScannIndex.is_available(), bool)
        True
        """
        try:
            import scann  # noqa: F401
        except ImportError:
            return False
        return True

    def _prep(self, vectors: np.ndarray) -> np.ndarray:
        """Coerce to float32 and normalise for the dot-product formulation."""
        arr = self._as_f32(vectors)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> ScannIndex:
        """Train the ScaNN searcher on the corpus."""
        scann = _require()
        arr = self._prep(vectors)
        n = arr.shape[0]
        leaves = self._num_leaves
        if leaves == "auto":
            leaves = int(max(1, min(np.sqrt(n), n)))
        # Fluent builder: partition into leaves, score with anisotropic hashing,
        # then exact-reorder the top candidates for the final recall bump.
        builder = scann.scann_ops_pybind.builder(arr, 10, "dot_product")
        self._index = (
            builder.tree(num_leaves=leaves, num_leaves_to_search=self._leaves_to_search)
            .score_ah(2, anisotropic_quantization_threshold=0.2)
            .reorder(self._reordering)
            .build()
        )
        self._ids = np.arange(n) if ids is None else np.asarray(ids, dtype=np.int64)
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Not supported — ScaNN searchers are immutable after build."""
        raise NotSupported("scann: add() unsupported — rebuild the searcher instead")

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Not supported — ScaNN searchers are immutable after build."""
        raise NotSupported("scann: add_with_ids() unsupported — rebuild the searcher instead")

    def remove(self, ids: np.ndarray) -> None:
        """Not supported — ScaNN cannot delete; rebuild without the ids."""
        raise NotSupported("scann: remove() unsupported — rebuild without those ids")

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate top-``k`` neighbours per query."""
        arr = self._prep(queries)
        # search_batched returns (neighbour_positions, distances); translate the
        # positions through the id table.
        neighbours, distances = self._index.search_batched(arr, final_num_neighbors=k)  # type: ignore[union-attr]
        return self._ids[neighbours], distances.astype(np.float32)

    def save(self, path: str) -> None:
        """Persist via ScaNN's directory serialiser plus the id table."""
        import os

        os.makedirs(path, exist_ok=True)
        self._index.serialize(path)  # type: ignore[union-attr]
        np.save(os.path.join(path, "ids.npy"), self._ids)

    def load(self, path: str) -> ScannIndex:
        """Load a searcher written by :meth:`save`."""
        import os

        scann = _require()
        self._index = scann.scann_ops_pybind.load_searcher(path)
        self._ids = np.load(os.path.join(path, "ids.npy"))
        return self
