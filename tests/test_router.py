"""Tests for the router orchestration layer (availability + rationale + build).

Where ``test_policy`` checks the pure decision, this checks the router that wraps
it: that it only ever returns an *installed* backend, that it explains a fallback
when the policy's first pick is missing, that ``auto_index`` really builds a
working index, and that the Markdown report is well-formed.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np

from ann_router.registry import BACKENDS, available_backends
from ann_router.router import auto_index, route, to_markdown
from ann_router.spec import Criteria


def test_route_returns_only_available_backend() -> None:
    # Whatever the criteria, the chosen backend must actually be installed here.
    for c in [
        Criteria(n_vectors=1_000, dim=32),
        Criteria(n_vectors=2_000_000, dim=768, target_recall=0.99),  # policy wants scann
        Criteria(n_vectors=500_000, dim=768, dynamic=True),
    ]:
        choice = route(c)
        assert choice.backend in available_backends()


def test_scann_absent_falls_back_with_explanation() -> None:
    # scann has no macOS/arm64 wheel; when the policy prefers it, the router must
    # fall back AND say why in the rationale.
    c = Criteria(n_vectors=2_000_000, dim=768, target_recall=0.99)
    choice = route(c)
    top_considered = choice.considered[0]["backend"]
    if not BACKENDS["scann"].is_available():
        assert top_considered == "scann"
        assert choice.backend != "scann"
        assert "not installed" in choice.rationale


def test_considered_list_marks_exactly_one_choice() -> None:
    choice = route(Criteria(n_vectors=200_000, dim=768, metadata_filtering=True))
    chosen = [row for row in choice.considered if row["chosen"]]
    assert len(chosen) == 1
    assert chosen[0]["backend"] == choice.backend


def test_auto_index_builds_queryable_index() -> None:
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((3_000, 48)).astype(np.float32)
    index, choice = auto_index(vecs, Criteria(n_vectors=3_000, dim=48))
    assert choice.backend == "exact"  # small corpus
    ids, dists = index.search(vecs[:3], k=5)
    assert ids.shape == (3, 5)
    # Nearest neighbour of a corpus point is itself (exact is exact).
    assert ids[0, 0] == 0


def test_auto_index_trusts_the_array_shape() -> None:
    # Even if Criteria.dim is wrong, the built index uses the array's real dim.
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((1_500, 24)).astype(np.float32)
    index, _ = auto_index(vecs, Criteria(n_vectors=1_500, dim=999))
    assert index.dim == 24


def test_markdown_report_is_wellformed() -> None:
    md = to_markdown(route(Criteria(n_vectors=1_000, dim=64)))
    assert md.startswith("# ann-router decision:")
    assert "Rationale" in md
    assert "policy version" in md
