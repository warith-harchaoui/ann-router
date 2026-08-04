"""Tests for the decision tree — the policy picks the documented backend.

This is the heart of the router's contract: for each representative criteria row
the *policy* (before availability is applied) must prefer the backend the suite's
documentation promises. These rows ARE the decision tree, pinned so a change to
the branch logic is a visible, reviewed diff.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import pytest

from ann_router.policy import EXACT_MAX_N, rank_backends
from ann_router.spec import Criteria


@pytest.mark.parametrize(
    "criteria, expected_top",
    [
        # 1. tiny corpus -> exact (approximation pointless)
        (Criteria(n_vectors=5_000, dim=128), "exact"),
        (Criteria(n_vectors=EXACT_MAX_N - 1, dim=64), "exact"),
        # 2. frequent updates -> turbovec (O(1) add/remove)
        (Criteria(n_vectors=500_000, dim=768, dynamic=True), "turbovec"),
        # 3. very large + GPU/batch -> faiss
        (Criteria(n_vectors=2_000_000, dim=768, hardware="gpu"), "faiss"),
        (Criteria(n_vectors=2_000_000, dim=768, batch_queries=True), "faiss"),
        # 4. persistence + metadata filter -> qdrant, or pgvector when a DB is in place
        (Criteria(n_vectors=200_000, dim=768, metadata_filtering=True), "qdrant"),
        (Criteria(n_vectors=200_000, dim=768, persistence=True), "qdrant"),
        (
            Criteria(n_vectors=200_000, dim=768, metadata_filtering=True,
                     extra={"pg_dsn": "postgres://x"}),
            "pgvector",
        ),
        # 5. max recall at scale -> scann
        (Criteria(n_vectors=2_000_000, dim=768, target_recall=0.99), "scann"),
        # 6. read-only + tight memory -> annoy
        (Criteria(n_vectors=5_000_000, dim=768, memory_budget_gb=1.0), "annoy"),
        # 7. stable in-memory default -> hnsw
        (Criteria(n_vectors=100_000, dim=768), "hnsw"),
    ],
)
def test_policy_picks_expected_backend(criteria: Criteria, expected_top: str) -> None:
    # rank_backends is availability-agnostic, so this tests the pure decision.
    shortlist = rank_backends(criteria)
    assert shortlist[0]["backend"] == expected_top


def test_gpu_needed_for_faiss_branch() -> None:
    # A very large corpus WITHOUT gpu/batch must not fall into the FAISS branch;
    # it should default to the stable in-memory engine instead.
    c = Criteria(n_vectors=2_000_000, dim=768)  # no gpu, no batch, static
    assert rank_backends(c)[0]["backend"] == "hnsw"


def test_dynamic_beats_persistence() -> None:
    # Priority order: a dynamic corpus routes to turbovec even if it also wants
    # persistence/filtering (churn is the dominant constraint).
    c = Criteria(n_vectors=300_000, dim=256, dynamic=True, metadata_filtering=True)
    assert rank_backends(c)[0]["backend"] == "turbovec"


def test_thresholds_are_tunable() -> None:
    # Overriding EXACT_MAX_N must move the exact->ANN crossover.
    c = Criteria(n_vectors=20_000, dim=128)
    assert rank_backends(c)[0]["backend"] == "hnsw"  # default: above 10k
    bumped = rank_backends(c, thresholds={"EXACT_MAX_N": 50_000})
    assert bumped[0]["backend"] == "exact"  # raised crossover keeps it exact


def test_every_shortlist_ends_with_a_universal_fallback() -> None:
    # The last eligible rule must always be one of the always-usable engines so
    # routing can never return an empty shortlist.
    c = Criteria(n_vectors=1_000_000, dim=512)
    tail = rank_backends(c)[-1]["backend"]
    assert tail in {"hnsw", "exact"}
