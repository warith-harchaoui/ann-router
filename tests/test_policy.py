"""Tests for the decision tree — the policy picks the documented backend.

This is the heart of the router's contract: for each representative criteria row
the *policy* (before availability is applied) must prefer the backend the suite's
documentation promises. These rows ARE the decision tree, pinned so a change to
the branch logic is a visible, reviewed diff.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import pytest

from ann_router.policy import EXACT_MAX_N, effective_exact_max_n, rank_backends
from ann_router.spec import Criteria


@pytest.mark.parametrize(
    "criteria, expected_top",
    [
        # 1. tiny corpus -> exact (approximation pointless)
        (Criteria(n_vectors=500, dim=128), "exact"),
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
            Criteria(
                n_vectors=200_000,
                dim=768,
                metadata_filtering=True,
                extra={"pg_dsn": "postgres://x"},
            ),
            "pgvector",
        ),
        # 5. read-only + tight memory -> annoy
        (Criteria(n_vectors=5_000_000, dim=768, memory_budget_gb=1.0), "annoy"),
        # 6. stable in-memory default -> hnsw
        (Criteria(n_vectors=100_000, dim=768), "hnsw"),
    ],
)
def test_policy_picks_expected_backend(criteria: Criteria, expected_top: str) -> None:
    """rank_backends() picks the documented backend for each representative criteria row."""
    # rank_backends is availability-agnostic, so this tests the pure decision.
    shortlist = rank_backends(criteria)
    assert shortlist[0]["backend"] == expected_top


def test_gpu_needed_for_faiss_branch() -> None:
    """A very large corpus WITHOUT gpu/batch falls to hnsw, not faiss."""
    # A very large corpus WITHOUT gpu/batch must not fall into the FAISS branch;
    # it should default to the stable in-memory engine instead.
    c = Criteria(n_vectors=2_000_000, dim=768)  # no gpu, no batch, static
    assert rank_backends(c)[0]["backend"] == "hnsw"


def test_dynamic_beats_persistence() -> None:
    """A dynamic corpus routes to turbovec even when it also wants filtering."""
    # Priority order: a dynamic corpus routes to turbovec even if it also wants
    # persistence/filtering (churn is the dominant constraint).
    c = Criteria(n_vectors=300_000, dim=256, dynamic=True, metadata_filtering=True)
    assert rank_backends(c)[0]["backend"] == "turbovec"


def test_thresholds_are_tunable() -> None:
    """Overriding EXACT_MAX_N moves the exact->ANN crossover."""
    # Overriding EXACT_MAX_N must move the exact->ANN crossover.
    c = Criteria(n_vectors=20_000, dim=128)
    assert rank_backends(c)[0]["backend"] == "hnsw"  # default: above 1k
    bumped = rank_backends(c, thresholds={"EXACT_MAX_N": 50_000})
    assert bumped[0]["backend"] == "exact"  # raised crossover keeps it exact


def test_every_shortlist_ends_with_a_universal_fallback() -> None:
    """The last eligible rule is always one of the always-usable engines."""
    # The last eligible rule must always be one of the always-usable engines so
    # routing can never return an empty shortlist.
    c = Criteria(n_vectors=1_000_000, dim=512)
    tail = rank_backends(c)[-1]["backend"]
    assert tail in {"hnsw", "exact"}


def test_latency_budget_scales_the_exact_crossover() -> None:
    """effective_exact_max_n() scales EXACT_MAX_N proportionally to latency_budget_ms."""
    # A tighter-than-reference budget affords fewer vectors to brute-force; a
    # looser one affords more. Reference is 10 ms (LATENCY_REFERENCE_MS).
    t = {"EXACT_MAX_N": EXACT_MAX_N, "LATENCY_REFERENCE_MS": 10.0}
    tight = Criteria(n_vectors=1, dim=8, latency_budget_ms=1.0)
    loose = Criteria(n_vectors=1, dim=8, latency_budget_ms=100.0)
    default = Criteria(n_vectors=1, dim=8)
    assert effective_exact_max_n(tight, t) == EXACT_MAX_N / 10
    assert effective_exact_max_n(loose, t) == EXACT_MAX_N * 10
    assert effective_exact_max_n(default, t) == EXACT_MAX_N


def test_tight_latency_budget_pushes_a_borderline_corpus_off_exact() -> None:
    """A tightened budget pushes a mid-size n off exact while the default keeps it."""
    # Same n, only the budget changes: n sits just above the tightened
    # crossover but just below the reference one.
    n = int(EXACT_MAX_N * 0.5)
    assert rank_backends(Criteria(n_vectors=n, dim=128))[0]["backend"] == "exact"
    tight = Criteria(n_vectors=n, dim=128, latency_budget_ms=1.0)  # affords EXACT_MAX_N/10
    assert rank_backends(tight)[0]["backend"] != "exact"
