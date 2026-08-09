"""ann-router — pick the right ANN vector-search backend from measured criteria.

``ann-router`` is a brique in Warith Harchaoui's *AI Helpers* suite. Like
its sibling ``best-engine-ai-helper`` (which picks the best local LLM for a
machine), it is a **router**: you describe the vector-search problem — corpus
size, dimensionality, recall target, latency/memory budgets, update pattern,
metadata-filtering need, hardware, persistence — and it selects, *justifies*, and
can *instantiate* the appropriate engine among:

    exact (brute force) · turbovec · HNSW (hnswlib) · FAISS (IVF/PQ) ·
    Annoy · Qdrant · pgvector

instead of marrying a single library.

Importing this package is cheap and dependency-free: no engine's optional
dependency is imported at import time, so ``import ann_router`` works even with
only numpy installed. A backend whose dependency is absent reports itself
unavailable and the router routes around it, explaining the fallback.

Quick start
-----------
>>> import numpy as np
>>> import ann_router as ar
>>> rng = np.random.default_rng(0)
>>> vecs = rng.standard_normal((500, 64)).astype(np.float32)
>>> choice = ar.route(ar.Criteria(n_vectors=500, dim=64))
>>> choice.backend
'exact'
>>> index, choice = ar.auto_index(vecs, ar.Criteria(n_vectors=500, dim=64))
>>> ids, dists = index.search(vecs[:2], k=5)
>>> ids.shape
(2, 5)

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from .base import ANNIndex, BackendUnavailable, Capabilities, NotSupported
from .config import backend_catalog, hardware_profiles, policy_thresholds
from .detect import detect_hardware, hardware_report
from .policy import POLICY_VERSION, THRESHOLDS, rank_backends
from .registry import all_capabilities, available_backends, get_backend
from .router import auto_index, route, to_markdown
from .spec import BackendChoice, Criteria

__version__ = "0.1.2"
__author__ = "Warith Harchaoui, Ph.D."
__email__ = "warith.harchaoui@deraison.ai"

__all__ = [
    # Problem description + decision result
    "Criteria",
    "BackendChoice",
    # The router surface
    "route",
    "auto_index",
    "to_markdown",
    # Policy (pure, tunable)
    "rank_backends",
    "THRESHOLDS",
    "POLICY_VERSION",
    "policy_thresholds",
    "backend_catalog",
    "hardware_profiles",
    # Backend contract + registry
    "ANNIndex",
    "Capabilities",
    "NotSupported",
    "BackendUnavailable",
    "get_backend",
    "available_backends",
    "all_capabilities",
    # Hardware probing
    "detect_hardware",
    "hardware_report",
    "__version__",
]
