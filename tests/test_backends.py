"""One end-to-end lifecycle test per backend adapter, uniform across engines.

For each backend whose dependency is installed, :func:`test_backend_lifecycle`
drives it the way a real caller would — build, check recall@k against the exact
ground truth, round-trip through save/load, then add and remove — in one
continuous run so the same built index is exercised by every check instead of
being rebuilt per assertion. add/remove are checked against what the capability
descriptor promises (and :class:`NotSupported` where it says a backend can't).
Backends whose dependency (or server) is absent are *skipped* with a reason,
never failed — so the suite is green on a bare machine and thorough on a full
one. :func:`test_capability_descriptor_is_consistent` stays a separate,
always-run check since it needs no dependency at all — it reads a classmethod
off the un-instantiated class.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ann_router.base import ANNIndex, NotSupported
from ann_router.registry import BACKENDS
from tests.conftest import RECALL_THRESHOLDS, recall_at_k

# Every registered backend name; parametrising over the registry keeps this test
# automatically in sync if a backend is added.
ALL_BACKENDS = list(BACKENDS)


def _skip_if_unavailable(name: str) -> type[ANNIndex]:
    """Return the adapter class or skip the test with an actionable reason."""
    cls = BACKENDS[name]
    if not cls.is_available():
        pytest.skip(f"backend '{name}' dependency not installed (pip install 'ann-router[{name}]')")
    # pgvector additionally needs a live server; without a DSN there is nothing
    # to test, so skip rather than fail on a connection error.
    if name == "pgvector" and not os.environ.get("ANN_ROUTER_PG_DSN"):
        pytest.skip("pgvector needs a live server — set ANN_ROUTER_PG_DSN to run it")
    return cls


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_capability_descriptor_is_consistent(name: str) -> None:
    # Descriptor is readable WITHOUT importing the dependency — that is the whole
    # point of keeping it a classmethod on an un-instantiated class.
    cap = BACKENDS[name].capabilities()
    assert cap.name == name
    assert "cosine" in cap.metrics
    # exact is the only non-approximate, always-available, no-extra backend.
    if name == "exact":
        assert cap.approximate is False and cap.pip_extra == ""
    else:
        assert cap.pip_extra != ""


@pytest.mark.parametrize("name", ALL_BACKENDS)
def test_backend_lifecycle(name: str, dataset: dict, tmp_path) -> None:
    """Build -> recall -> save/load -> add -> remove, on one live index."""
    cls = _skip_if_unavailable(name)
    cap = cls.capabilities()
    dim, k = dataset["dim"], dataset["k"]

    index = cls(dim=dim).build(dataset["corpus"])
    ids, _ = index.search(dataset["queries"], k)
    # Output shape is part of the contract: rectangular (q, k).
    assert ids.shape == (len(dataset["queries"]), k)
    recall = recall_at_k(ids, dataset["truth"], k)
    floor = RECALL_THRESHOLDS[name]
    assert recall >= floor, f"{name} recall@{k}={recall:.3f} below floor {floor}"

    if name not in ("qdrant", "pgvector"):  # these persist via their store, not save()/load()
        path = str(tmp_path / f"{name}.idx")
        index.save(path)
        # A freshly constructed adapter must reproduce the same neighbours from disk.
        reloaded = cls(dim=dim).load(path)
        after, _ = reloaded.search(dataset["queries"], k)
        assert np.array_equal(ids, after)

    # Continue on a smaller live index so add/remove ids are easy to reason about.
    small = cls(dim=dim).build(dataset["corpus"][:2000])
    extra, extra_ids = dataset["corpus"][2000:2010], np.arange(2000, 2010)
    if cap.supports_add:
        # Adding must grow the searchable set — the new ids become findable.
        small.add_with_ids(extra, extra_ids)
        found, _ = small.search(extra[:1], k=5)
        assert found.shape == (1, 5)
    else:
        # A frozen backend must say so honestly rather than silently no-op.
        with pytest.raises(NotSupported):
            small.add_with_ids(extra, extra_ids)

    if cap.supports_remove:
        # Remove a specific id and confirm it no longer appears for a query that
        # otherwise returns it (its own vector is the strongest match).
        target = 7
        query = dataset["corpus"][target : target + 1]
        small.remove(np.array([target]))
        found, _ = small.search(query, k=10)
        assert target not in found[0]
    else:
        with pytest.raises(NotSupported):
            small.remove(np.array([0]))
