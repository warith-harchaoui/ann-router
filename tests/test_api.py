"""Tests for the FastAPI door — skipped cleanly when the ``[api]`` extra is absent.

The HTTP surface must return the same decisions as the library, so these drive
the app through FastAPI's TestClient and check the route/capabilities endpoints.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import pytest


@pytest.fixture
def client():
    # fastapi + httpx (TestClient's transport) live behind the [api] extra.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from ann_router.api import create_app

    return TestClient(create_app())


def test_capabilities_endpoint(client) -> None:
    resp = client.get("/capabilities")
    assert resp.status_code == 200
    assert "exact" in resp.json()["available"]


def test_route_endpoint(client) -> None:
    resp = client.post("/route", json={"n_vectors": 500, "dim": 16})
    assert resp.status_code == 200
    assert resp.json()["backend"] == "exact"


def test_route_endpoint_dynamic(client) -> None:
    resp = client.post("/route", json={"n_vectors": 500000, "dim": 768, "dynamic": True})
    # turbovec is installed in the dev env; on a bare machine the fallback (hnsw)
    # is still a valid, available answer.
    assert resp.json()["backend"] in {"turbovec", "hnsw", "exact"}
