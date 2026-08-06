"""Shared fixtures and versioned recall thresholds for the ann-router suite.

The synthetic corpus here is *clustered* on purpose: real embeddings sit on a
low-dimensional manifold, so uniform noise (where every neighbour is equidistant
and even the exact "truth" is degenerate) would be a dishonest benchmark. The
fixtures build one deterministic clustered dataset and its exact ground truth,
which every backend's recall@k test scores against.

``RECALL_THRESHOLDS`` is the rule-16-style versioned eval contract: one floor per
backend, set comfortably below the values measured on this dataset (Apple
Silicon, 2026-08) so the suite is robust to run-to-run jitter but still catches a
real regression. Bumping a floor is a deliberate, reviewed change.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np
import pytest

from ann_router.backends.exact import ExactIndex

# Locked benchmark geometry — changing any of these invalidates the thresholds.
SEED = 20260804
N_VECTORS = 8_000
DIM = 128
N_QUERIES = 50
K = 10
N_CENTERS = 50
CLUSTER_SPREAD = 0.4

# Versioned recall@k floors. Measured on the fixture below (recall@10):
#   exact 1.000 · hnsw 1.000 · faiss 1.000 · annoy 1.000 · qdrant 1.000 · turbovec 0.700
# turbovec's 2-4 bit TurboQuant is data-dependent, hence the deliberately lower
# floor; the graph/exact engines get a strict 0.90.
RECALL_THRESHOLDS: dict[str, float] = {
    "exact": 1.0,
    "hnsw": 0.90,
    "faiss": 0.90,
    "annoy": 0.90,
    "qdrant": 0.90,
    "pgvector": 0.90,
    "turbovec": 0.55,
}


def _clustered(n: int, rng: np.random.Generator, centers: np.ndarray) -> np.ndarray:
    """Sample ``n`` points from the given cluster centres with fixed spread."""
    # Each point == a random centre plus small Gaussian jitter, giving stable,
    # well-defined nearest neighbours (unlike uniform high-dim noise).
    idx = rng.integers(0, len(centers), n)
    return (centers[idx] + CLUSTER_SPREAD * rng.standard_normal((n, DIM))).astype(np.float32)


@pytest.fixture(scope="session")
def dataset() -> dict:
    """Return the deterministic clustered corpus, queries and exact ground truth.

    Returns
    -------
    dict
        ``{"corpus", "queries", "truth", "dim", "k"}`` — ``truth`` is the exact
        top-k id matrix every backend is scored against.
    """
    rng = np.random.default_rng(SEED)
    centers = rng.standard_normal((N_CENTERS, DIM)).astype(np.float32)
    corpus = _clustered(N_VECTORS, rng, centers)
    queries = _clustered(N_QUERIES, rng, centers)
    # Exact search defines the ground truth all approximate backends aim at.
    truth, _ = ExactIndex(dim=DIM).build(corpus).search(queries, K)
    return {"corpus": corpus, "queries": queries, "truth": truth, "dim": DIM, "k": K}


def recall_at_k(pred_ids: np.ndarray, truth_ids: np.ndarray, k: int) -> float:
    """Return mean recall@k of predicted neighbours against the exact truth.

    Parameters
    ----------
    pred_ids : numpy.ndarray
        Backend output, shape ``(q, k)``.
    truth_ids : numpy.ndarray
        Exact ground truth, shape ``(q, k)``.
    k : int
        Neighbour count.

    Returns
    -------
    float
        Mean over queries of ``|pred ∩ truth| / k``.
    """
    return float(
        np.mean([len(set(a) & set(b)) / k for a, b in zip(pred_ids, truth_ids, strict=False)])
    )
