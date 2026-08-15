"""Tests for the FastAPI door — skipped cleanly when the ``[api]`` extra is absent.

The HTTP surface must return the same decisions as the library, so these drive
the app through FastAPI's TestClient and check the route/capabilities endpoints.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client():
    """A ``TestClient`` over a fresh app, skipped when ``[api]`` is absent.

    Returns
    -------
    fastapi.testclient.TestClient
        A client wrapping :func:`ann_router.api.create_app`'s app.
    """
    # fastapi + httpx (TestClient's transport) live behind the [api] extra.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ann_router.api import create_app

    return TestClient(create_app())


def test_capabilities_endpoint(client) -> None:
    """``GET /capabilities`` lists ``exact`` among the available backends."""
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    assert "exact" in resp.json()["available"]


def test_route_endpoint(client) -> None:
    """``POST /route`` on a tiny corpus routes to ``exact``."""
    resp = client.post("/route", json={"n_vectors": 500, "dim": 16})
    assert resp.status_code == 200
    assert resp.json()["backend"] == "exact"


def test_route_endpoint_dynamic(client) -> None:
    """``POST /route`` on a large dynamic corpus routes to a valid, available backend."""
    resp = client.post("/route", json={"n_vectors": 500000, "dim": 768, "dynamic": True})
    # turbovec is installed in the dev env; on a bare machine the fallback (hnsw)
    # is still a valid, available answer.
    assert resp.json()["backend"] in {"turbovec", "hnsw", "exact"}


def test_route_endpoint_rejects_malformed_criteria_with_400_not_500(client) -> None:
    # Criteria.validate()'s ValueError (bad range, or an unrecognised
    # hardware/metric string -- CriteriaModel types both as plain `str`, so
    # nothing rejects a typo before it reaches the dataclass) used to fall
    # through to FastAPI's generic 500, indistinguishable from a real bug.
    resp = client.post("/route", json={"n_vectors": -1, "dim": 16})
    assert resp.status_code == 400
    assert "n_vectors" in resp.json()["detail"]

    resp = client.post("/route", json={"n_vectors": 500, "dim": 16, "metric": "euclidean"})
    assert resp.status_code == 400
    assert "metric" in resp.json()["detail"]
