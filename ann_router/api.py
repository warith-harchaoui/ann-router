"""FastAPI HTTP surface — the web-API door (optional ``[api]`` extra).

The third of ann-router's five surfaces exposes the router over HTTP so a
service can ask "which backend for this problem?" (and benchmark/inspect
backends) without a Python import. FastAPI + uvicorn live behind the ``[api]``
extra, imported lazily here so the core package never pays for a web stack it
does not use — the same quarantine os-helper applies to its GUI. Every endpoint
delegates to :mod:`ann_router._core_cli`, so the HTTP behaviour matches the CLI
and library exactly.

Each route carries an explicit ``operation_id`` (``route``/``capabilities``/
``bench``) — not just OpenAPI hygiene: :mod:`ann_router.mcp_server` mounts
``fastapi-mcp`` on a copy of this same app and selects exactly these
operation ids as the exposed MCP tools, so the id *is* the tool name an agent
calls. Rename a route here and the MCP door renames with it, automatically.

Run it with::

    pip install 'ann-router[api]'
    uvicorn ann_router.api:app --reload   # or: python -m ann_router.api

Consumes: ``fastapi`` (optional), ``ann_router._core_cli``.
Produces: :data:`app` (the ASGI application), :func:`create_app`.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

from typing import Any

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the HTTP API needs the [api] extra. Run: pip install 'ann-router[api]'"
    ) from exc

from . import _core_cli as core
from .spec import Criteria


class CriteriaModel(BaseModel):
    """Request body mirroring :class:`ann_router.spec.Criteria`.

    Only ``n_vectors`` and ``dim`` are required; the rest carry the same house
    defaults as the dataclass so a minimal request still routes.
    """

    n_vectors: int
    dim: int
    target_recall: float = 0.95
    latency_budget_ms: float = 10.0
    memory_budget_gb: float | None = None
    dynamic: bool = False
    metadata_filtering: bool = False
    hardware: str = "cpu"
    persistence: bool = False
    batch_queries: bool = False
    metric: str = "cosine"
    extra: dict[str, Any] = {}


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Returns
    -------
    FastAPI
        The ASGI app with the ``/route``, ``/capabilities`` and ``/bench``
        routes wired to the shared core.

    Examples
    --------
    >>> app = create_app()  # doctest: +SKIP
    >>> [r.path for r in app.routes]  # doctest: +SKIP
    ['/openapi.json', '/docs', ..., '/route', '/capabilities', '/bench']
    """
    app = FastAPI(
        title="ann-router",
        description="Route to the right ANN vector-search backend from measured criteria.",
        version="0.1.0",
    )

    @app.get("/capabilities", operation_id="capabilities")
    def capabilities() -> dict:
        """Return the availability + capability matrix of every backend."""
        return core.do_capabilities()

    @app.post("/route", operation_id="route")
    def route_endpoint(body: CriteriaModel, markdown: bool = False) -> dict:
        """Route a criteria payload and return the justified decision."""
        # Validate through the dataclass so the HTTP path enforces the same
        # invariants as the library before answering.
        criteria = Criteria.from_dict(body.model_dump()).to_dict()
        result = core.do_route(criteria, as_markdown=markdown)
        return {"markdown": result} if markdown else result

    @app.get("/bench", operation_id="bench")
    def bench(n: int = 5000, dim: int = 128, k: int = 10) -> dict:
        """Benchmark installed backends vs the exact ground truth."""
        return core.do_bench(n=n, dim=dim, k=k)

    return app


# Module-level ASGI app so `uvicorn ann_router.api:app` works out of the box.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8018)
