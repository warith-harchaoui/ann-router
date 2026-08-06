"""Backend adapters, one module per ANN engine.

Each module here wraps a single vector-search engine behind the shared
:class:`ann_router.base.ANNIndex` contract. Only :mod:`ann_router.backends.exact`
has no optional dependency; every other module defers its heavy import so that
``import ann_router`` (and importing this package) never fails because an engine
is not installed. Use :mod:`ann_router.registry` to look adapters up by name.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from .annoy_backend import AnnoyIndex
from .exact import ExactIndex
from .faiss_backend import FaissIndex
from .hnsw import HNSWIndex
from .pgvector_backend import PgVectorIndex
from .qdrant_backend import QdrantIndex
from .turbovec_backend import TurboVecIndex

__all__ = [
    "ExactIndex",
    "TurboVecIndex",
    "HNSWIndex",
    "FaissIndex",
    "AnnoyIndex",
    "QdrantIndex",
    "PgVectorIndex",
]
