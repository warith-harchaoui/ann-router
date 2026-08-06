"""Tests for the Criteria / BackendChoice data model.

These pin the invariants the whole router relies on: that criteria validate
their numeric ranges, that they round-trip losslessly through JSON (the CLI/API/
MCP wire format), and that ``from_dict`` is forgiving of unknown keys so older
clients keep working.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import pytest

from ann_router.spec import BackendChoice, Criteria


def test_defaults_match_house_policy() -> None:
    """Criteria's documented defaults (target_recall, budget, metric) hold."""
    # The documented defaults are load-bearing (they define the "common" case).
    c = Criteria(n_vectors=100, dim=8)
    assert c.target_recall == 0.95
    assert c.latency_budget_ms == 10.0
    assert c.metric == "cosine"
    assert c.dynamic is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_vectors": -1, "dim": 8},  # negative corpus
        {"n_vectors": 10, "dim": 0},  # zero dimension
        {"n_vectors": 10, "dim": 8, "target_recall": 1.5},  # recall > 1
        {"n_vectors": 10, "dim": 8, "latency_budget_ms": 0},  # non-positive budget
        {"n_vectors": 10, "dim": 8, "memory_budget_gb": -2.0},  # negative memory
    ],
)
def test_validate_rejects_bad_ranges(kwargs: dict) -> None:
    """validate() raises on out-of-range n_vectors/dim/target_recall/budgets."""
    with pytest.raises(ValueError):
        Criteria(**kwargs).validate()


def test_round_trip_through_dict() -> None:
    """to_dict() / from_dict() are inverses, including dynamic and extra."""
    # to_dict / from_dict must be inverses for the JSON surfaces to be lossless.
    c = Criteria(n_vectors=42, dim=16, dynamic=True, extra={"pg_dsn": "x"})
    again = Criteria.from_dict(c.to_dict())
    assert again == c


def test_from_dict_ignores_unknown_keys() -> None:
    """from_dict() tolerates an unknown key without raising."""
    # Forward-compat: an extra key from a newer/older client must not crash.
    c = Criteria.from_dict({"n_vectors": 10, "dim": 4, "future_flag": True})
    assert c.n_vectors == 10 and c.dim == 4


def test_backend_choice_is_json_ready() -> None:
    """BackendChoice.to_dict() carries the backend name and config through."""
    choice = BackendChoice(backend="exact", rationale="tiny", config={"metric": "cosine"})
    payload = choice.to_dict()
    assert payload["backend"] == "exact"
    assert payload["config"]["metric"] == "cosine"
