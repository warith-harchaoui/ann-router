"""Tests for the YAML config layer (thresholds override, catalogs).

Proves the shipped policy.yaml is genuinely load-bearing: an external override
file changes the routing crossover, and the catalogs stay in sync with the
registry.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from ann_router.config import backend_catalog, hardware_profiles, policy_thresholds
from ann_router.policy import THRESHOLDS, rank_backends
from ann_router.registry import BACKENDS
from ann_router.router import route
from ann_router.spec import Criteria


def test_packaged_thresholds_match_code_defaults() -> None:
    """policy.yaml's thresholds never silently diverge from the code constants."""
    # The shipped policy.yaml must not silently diverge from the code constants.
    packaged = policy_thresholds()
    for key, value in THRESHOLDS.items():
        assert packaged[key] == value


def test_external_override_moves_the_crossover(tmp_path) -> None:
    """A user policy.yaml override raises the exact->ANN crossover end to end."""
    # A user override file must raise the exact->ANN crossover.
    override = tmp_path / "policy.yaml"
    override.write_text("thresholds:\n  EXACT_MAX_N: 100000\n")
    merged = policy_thresholds(path=str(override))
    assert merged["EXACT_MAX_N"] == 100000
    # And the router honours it: 20k vectors stay exact under the raised bound.
    c = Criteria(n_vectors=20_000, dim=128)
    assert rank_backends(c, merged)[0]["backend"] == "exact"


def test_route_uses_env_override(monkeypatch, tmp_path) -> None:
    """route() with no explicit thresholds picks up ANN_ROUTER_POLICY."""
    override = tmp_path / "p.yaml"
    override.write_text("thresholds:\n  EXACT_MAX_N: 100000\n")
    monkeypatch.setenv("ANN_ROUTER_POLICY", str(override))
    # route() with no explicit thresholds should pick up the env override.
    assert route(Criteria(n_vectors=20_000, dim=128)).backend == "exact"


def test_catalog_covers_every_registered_backend() -> None:
    """backend_catalog() names exactly the registered backends, no more, no less."""
    names = {b["name"] for b in backend_catalog()}
    assert names == set(BACKENDS)


def test_hardware_profiles_are_the_three_classes() -> None:
    """hardware_profiles() enumerates exactly cpu/apple_silicon/gpu."""
    assert {p["hardware"] for p in hardware_profiles()} == {"cpu", "apple_silicon", "gpu"}
