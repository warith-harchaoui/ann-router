"""Tests for the router orchestration layer (availability + rationale + build).

Where ``test_policy`` checks the pure decision, this checks the router that wraps
it: that it only ever returns an *installed* backend, that it explains a fallback
when the policy's first pick is missing, that ``auto_index`` really builds a
working index, and that the Markdown report is well-formed.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np
import pytest

from ann_router.registry import BACKENDS, available_backends, get_backend
from ann_router.router import _recommended_config, auto_index, route, to_markdown
from ann_router.spec import Criteria


def test_route_returns_only_available_backend() -> None:
    # Whatever the criteria, the chosen backend must actually be installed here.
    for c in [
        Criteria(n_vectors=1_000, dim=32),
        Criteria(n_vectors=2_000_000, dim=768, target_recall=0.99),  # high-recall, large-n regime
        Criteria(n_vectors=500_000, dim=768, dynamic=True),
    ]:
        choice = route(c)
        assert choice.backend in available_backends()


def test_unavailable_top_pick_falls_back_with_explanation(monkeypatch) -> None:
    # Force the policy's top pick (hnsw, the static-corpus default) unavailable
    # and confirm the router falls back to the next eligible backend AND says
    # why in the rationale — the "discussable fallback" the house style requires.
    monkeypatch.setattr(BACKENDS["hnsw"], "is_available", classmethod(lambda cls: False))
    c = Criteria(n_vectors=2_000_000, dim=768)
    choice = route(c)
    assert choice.considered[0]["backend"] == "hnsw"
    assert choice.backend != "hnsw"
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


def test_get_backend_rejects_unknown_name() -> None:
    with pytest.raises(KeyError):
        get_backend("does-not-exist")


def test_recommended_config_scales_with_target_recall() -> None:
    # Higher target_recall should push each backend's config toward more work
    # (bigger graph/forest/PQ search), not just carry the metric through.
    lo = Criteria(n_vectors=500_000, dim=768, target_recall=0.90)
    hi = Criteria(n_vectors=500_000, dim=768, target_recall=0.99)
    assert _recommended_config("hnsw", lo)["M"] < _recommended_config("hnsw", hi)["M"]
    assert _recommended_config("annoy", lo)["n_trees"] < _recommended_config("annoy", hi)["n_trees"]
    assert _recommended_config("faiss", lo)["nprobe"] == 16  # static default, still present


def test_recommended_config_carries_dsn_for_db_backends() -> None:
    c = Criteria(n_vectors=500_000, dim=768, extra={"pg_dsn": "postgres://x"})
    assert _recommended_config("pgvector", c)["dsn"] == "postgres://x"
    assert _recommended_config("qdrant", c)["dsn"] == "postgres://x"
    assert "dsn" not in _recommended_config("hnsw", c)
