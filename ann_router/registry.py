"""Backend registry: the single lookup table from name to adapter class.

Every backend adapter is a subclass of :class:`ann_router.base.ANNIndex`; this
module is the one place that maps the seven backend names to those classes and
answers the two questions the router asks about each: *what can it do*
(capabilities, readable without the dependency) and *can it run here* (is the
optional dependency importable). Importing this module, and therefore
``ann_router``, must stay dependency-free, so the adapter modules it imports
all defer their heavy imports.

Consumes: the seven ``ann_router.backends.*`` adapter modules.
Produces: :data:`BACKENDS`, :func:`get_backend`, :func:`available_backends`,
:func:`all_capabilities`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from .backends.annoy_backend import AnnoyIndex
from .backends.exact import ExactIndex
from .backends.faiss_backend import FaissIndex
from .backends.hnsw import HNSWIndex
from .backends.pgvector_backend import PgVectorIndex
from .backends.qdrant_backend import QdrantIndex
from .backends.turbovec_backend import TurboVecIndex
from .base import ANNIndex, Capabilities

# The canonical name -> adapter-class table. Order is the policy's preference
# order for ties, most-specialised first, exact last as the universal fallback.
BACKENDS: dict[str, type[ANNIndex]] = {
    "exact": ExactIndex,
    "turbovec": TurboVecIndex,
    "hnsw": HNSWIndex,
    "faiss": FaissIndex,
    "annoy": AnnoyIndex,
    "qdrant": QdrantIndex,
    "pgvector": PgVectorIndex,
}


def get_backend(name: str) -> type[ANNIndex]:
    """Return the adapter class registered under ``name``.

    Parameters
    ----------
    name : str
        A backend identifier (see :data:`ann_router.spec.BackendName`).

    Returns
    -------
    type[ANNIndex]
        The adapter class (not an instance).

    Raises
    ------
    KeyError
        If ``name`` is not a known backend.

    Examples
    --------
    >>> get_backend("exact").__name__
    'ExactIndex'
    """
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}")
    return BACKENDS[name]


def available_backends() -> list[str]:
    """Return the names of backends whose dependency is importable here.

    Returns
    -------
    list of str
        Sorted-by-preference subset of :data:`BACKENDS` that can actually run.

    Examples
    --------
    >>> "exact" in available_backends()   # numpy is always present
    True
    """
    # Preserve BACKENDS insertion order (preference order) rather than sorting.
    return [name for name, cls in BACKENDS.items() if cls.is_available()]


def all_capabilities() -> dict[str, Capabilities]:
    """Return the capability descriptor of every registered backend.

    Returns
    -------
    dict
        ``{name: Capabilities}`` — readable even for uninstalled backends.

    Examples
    --------
    >>> all_capabilities()["annoy"].supports_remove
    False
    """
    return {name: cls.capabilities() for name, cls in BACKENDS.items()}
