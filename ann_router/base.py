"""The common ANN index interface every backend implements.

The router's whole value proposition is that eight very different vector-search
engines — from a pure-numpy brute-force scan to Qdrant — can be driven through
*one* small surface. This module defines that surface:

* :class:`Capabilities` — a static descriptor of what a backend can do
  (remove? filter? persist? needs a GPU?), so the router can reason about a
  backend without importing its (possibly absent) dependency.
* :class:`ANNIndex` — the abstract base class with ``build`` / ``add`` /
  ``add_with_ids`` / ``remove`` / ``search`` / ``save`` / ``load``.
* :class:`NotSupported` and :class:`BackendUnavailable` — the two honest
  failure modes: an operation a backend genuinely cannot do (Annoy removes),
  versus a backend whose optional dependency is not installed.

Consumes: ``ann_router.spec`` (metric names).
Produces: the base classes every module in ``ann_router.backends`` subclasses.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import cost
    from .spec import MetricName


class NotSupported(RuntimeError):
    """Raised when a backend genuinely cannot perform a requested operation.

    This is distinct from "not yet implemented": it flags a *fundamental*
    limitation of the engine (e.g. Annoy is frozen after ``build`` and cannot
    add or remove). The router uses the capability descriptor to avoid routing
    a dynamic workload to such a backend, but the guard here is the last line
    of defence for callers who instantiate a backend directly.

    Examples
    --------
    >>> raise NotSupported("annoy: remove() unsupported (frozen index)")
    Traceback (most recent call last):
    ...
    ann_router.base.NotSupported: annoy: remove() unsupported (frozen index)
    """


class BackendUnavailable(ImportError):
    """Raised when a backend's optional dependency is not installed.

    Importing ``ann_router`` must never fail because ``faiss`` or ``turbovec``
    are absent, so backends defer their heavy imports. When a caller actually
    tries to *use* an uninstalled backend, this actionable error names the
    ``pip install`` extra that fixes it.

    Examples
    --------
    >>> raise BackendUnavailable("faiss not installed. Run: pip install 'ann-router[faiss]'")
    Traceback (most recent call last):
    ...
    ann_router.base.BackendUnavailable: faiss not installed. Run: pip install 'ann-router[faiss]'
    """


@dataclass(frozen=True)
class Capabilities:
    """Static description of what a backend can and cannot do.

    The router reads this *without* importing the backend's dependency, so it
    can rank engines even when their libraries are absent. Each flag maps to a
    routing consequence documented inline.

    Parameters
    ----------
    name : str
        The backend identifier (matches :data:`ann_router.spec.BackendName`).
    supports_add : bool
        Can accept vectors after the initial ``build`` (streaming inserts).
    supports_remove : bool
        Can delete vectors by id without a full rebuild.
    supports_filter : bool
        Can restrict a search by structured metadata / payload.
    persistent : bool
        Naturally survives process restarts (a database) as opposed to needing
        an explicit ``save``/``load`` round-trip.
    needs_gpu : bool
        Requires a GPU to be worthwhile (or at all).
    approximate : bool
        Returns approximate neighbours (vs. exact ground truth).
    metrics : tuple of str
        The distance metrics the backend supports.
    pip_extra : str
        The optional-dependency extra that installs it (empty for ``exact``).

    Examples
    --------
    >>> cap = Capabilities(name="exact", supports_add=True, supports_remove=True,
    ...                     supports_filter=False, persistent=False, needs_gpu=False,
    ...                     approximate=False, metrics=("cosine", "l2", "ip"), pip_extra="")
    >>> cap.approximate
    False
    """

    name: str
    supports_add: bool
    supports_remove: bool
    supports_filter: bool
    persistent: bool
    needs_gpu: bool
    approximate: bool
    metrics: tuple[str, ...]
    pip_extra: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the descriptor.

        Returns
        -------
        dict
            One key per flag.

        Examples
        --------
        >>> from ann_router.backends.exact import ExactIndex
        >>> ExactIndex.capabilities().to_dict()["name"]
        'exact'
        """
        return {
            "name": self.name,
            "supports_add": self.supports_add,
            "supports_remove": self.supports_remove,
            "supports_filter": self.supports_filter,
            "persistent": self.persistent,
            "needs_gpu": self.needs_gpu,
            "approximate": self.approximate,
            "metrics": list(self.metrics),
            "pip_extra": self.pip_extra,
        }


class ANNIndex(abc.ABC):
    """Abstract base class for every ANN backend adapter.

    Subclasses wrap one engine behind a uniform surface so the router — and
    downstream code — can build, query, mutate, and persist an index without
    knowing which engine is underneath. Not every backend supports every
    operation; those raise :class:`NotSupported` and advertise the limitation
    via :meth:`capabilities`.

    Parameters
    ----------
    dim : int
        Embedding dimensionality the index is built for.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``.
    **kwargs
        Backend-specific build parameters (e.g. HNSW ``M``).

    Notes
    -----
    Concrete constructors must *not* import their heavy dependency at module
    import time — only inside methods (or a lazily-called checker) — so that
    ``import ann_router`` stays cheap and dependency-free.
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        # Store the shared shape/metric contract; concrete backends stash their
        # own tuning in ``self._params`` and their native handle in ``self._index``.
        self.dim = dim
        self.metric = metric
        self._params = dict(kwargs)
        self._index: object | None = None

    # -- capability / availability -------------------------------------------------

    @classmethod
    @abc.abstractmethod
    def capabilities(cls) -> Capabilities:
        """Return the static capability descriptor for this backend.

        Returns
        -------
        Capabilities
            What the backend can do — readable without importing its dependency.
        """

    @classmethod
    @abc.abstractmethod
    def is_available(cls) -> bool:
        """Return ``True`` if the backend's dependency is importable.

        Returns
        -------
        bool
            ``True`` when the engine can actually be used on this machine.
        """

    # -- lifecycle -----------------------------------------------------------------

    @abc.abstractmethod
    def build(self, vectors: np.ndarray, ids: np.ndarray | None = None) -> ANNIndex:
        """Build the index from an initial batch of vectors.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(n, dim)``, dtype float32 (coerced if needed).
        ids : numpy.ndarray, optional
            Shape ``(n,)`` integer ids. Defaults to ``range(n)``.

        Returns
        -------
        ANNIndex
            ``self``, so calls can be chained.
        """

    @abc.abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Append vectors, assigning them the next contiguous ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.

        Raises
        ------
        NotSupported
            If the backend is frozen after ``build`` (e.g. Annoy).
        """

    @abc.abstractmethod
    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids.

        Raises
        ------
        NotSupported
            If the backend cannot map external ids or is frozen.
        """

    @abc.abstractmethod
    def remove(self, ids: np.ndarray) -> None:
        """Delete vectors by id.

        Parameters
        ----------
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids to drop.

        Raises
        ------
        NotSupported
            If the backend cannot delete (e.g. Annoy) or only tombstones.
        """

    @abc.abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``k`` nearest neighbours of each query.

        Parameters
        ----------
        queries : numpy.ndarray
            Shape ``(q, dim)``.
        k : int
            Number of neighbours per query.

        Returns
        -------
        ids : numpy.ndarray
            Shape ``(q, k)`` integer neighbour ids (``-1`` pads missing slots).
        distances : numpy.ndarray
            Shape ``(q, k)`` distances under the index metric.
        """

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the index to ``path``.

        Parameters
        ----------
        path : str
            Destination file (or directory) path.

        Raises
        ------
        NotSupported
            If the backend cannot serialise itself.
        """

    @abc.abstractmethod
    def load(self, path: str) -> ANNIndex:
        """Load a previously :meth:`save`\\ d index from ``path``.

        Parameters
        ----------
        path : str
            Source path produced by :meth:`save`.

        Returns
        -------
        ANNIndex
            ``self``, populated from disk.
        """

    # -- shared helpers ------------------------------------------------------------

    def _as_f32(self, vectors: np.ndarray) -> np.ndarray:
        """Coerce an array to contiguous ``float32`` of the right width.

        Parameters
        ----------
        vectors : numpy.ndarray
            Any 2-D array of vectors.

        Returns
        -------
        numpy.ndarray
            C-contiguous float32 array shaped ``(n, dim)``.

        Examples
        --------
        >>> from ann_router.backends.exact import ExactIndex
        >>> idx = ExactIndex(dim=3)
        >>> idx._as_f32(np.array([[1, 2, 3]])).dtype
        dtype('float32')
        """
        # ``np.ascontiguousarray`` is a no-op when the array is already laid out
        # correctly, so this is cheap on the hot path yet safe for lists/views.
        arr = np.ascontiguousarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            # Accept a single vector and promote it to a 1-row batch.
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {arr.shape[1]}")
        return arr

    def __repr__(self) -> str:
        """Return a concise, backend-tagged representation.

        Examples
        --------
        >>> from ann_router.backends.exact import ExactIndex
        >>> repr(ExactIndex(dim=8))
        "ExactIndex(dim=8, metric='cosine')"
        """
        return f"{type(self).__name__}(dim={self.dim}, metric={self.metric!r})"
