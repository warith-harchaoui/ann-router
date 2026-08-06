"""MCP server — the agent-tool door (optional ``[mcp]`` extra), via fastapi-mcp.

The fifth surface (paired with the ``skills/ann-router`` skill) lets an AI
agent call the router as Model-Context-Protocol tools. Rather than hand
-writing a parallel set of ``@server.tool()`` wrappers (the previous design,
which duplicated every endpoint's parameter list and docstring against
:mod:`ann_router.api`), this module mounts ``fastapi-mcp`` on a copy of that
same FastAPI app: the three REST operations already tagged with an explicit
``operation_id`` there (``route``/``capabilities``/``bench``) become the three
MCP tools here, automatically, with the same request/response schema. Rename
or extend a route in ``api.py`` and this door follows without a second edit.

This is an architectural change from the pre-fastapi-mcp version: MCP is now
served over **Streamable HTTP** (mounted at ``/mcp`` on a running app), not
stdio — a stdio subprocess client (spawn-and-talk-over-stdin/stdout) is not
what ``fastapi-mcp`` offers; an HTTP-based MCP client pointed at the running
server's ``/mcp`` endpoint is. The upside: the exact same three tools are also
just REST endpoints an agent (or a human with curl) can hit directly.

Run it with::

    pip install 'ann-router[mcp]'
    uvicorn ann_router.mcp_server:app --port 8019   # or: python -m ann_router.mcp_server

MCP endpoint: ``http://127.0.0.1:8019/mcp``. Runs on a different port from
:mod:`ann_router.api`'s 8018 by default so both doors can run side by side.

Consumes: ``fastapi-mcp`` (optional, pulls in ``fastapi``/``mcp``),
:mod:`ann_router.api`.
Produces: :data:`app` (the ASGI application), :func:`build_server`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from fastapi_mcp import FastApiMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the MCP server needs the [mcp] extra. Run: pip install 'ann-router[mcp]'"
    ) from exc

from .api import create_app

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

# The only REST operations exposed as MCP tools — matches the operation_id set
# on each route in ann_router.api.create_app(). An explicit allowlist (rather
# than excluding FastAPI's auto /docs, /openapi.json) so a future route added
# to the API door doesn't silently become an MCP tool without a deliberate opt-in.
_EXPOSED_OPERATIONS = ["route", "capabilities", "bench"]


def build_server(app: FastAPI) -> FastApiMCP:
    """Mount the MCP server (route/capabilities/bench tools) onto ``app``.

    Parameters
    ----------
    app : fastapi.FastAPI
        The app to mount onto — typically a fresh :func:`ann_router.api.create_app`
        instance, so this module's :data:`app` stays independent of
        :mod:`ann_router.api`'s own module-level one.

    Returns
    -------
    fastapi_mcp.FastApiMCP
        The mounted server descriptor (``.name`` defaults to ``app.title``).

    Examples
    --------
    >>> from ann_router.api import create_app
    >>> mcp = build_server(create_app())  # doctest: +SKIP
    >>> mcp.name  # doctest: +SKIP
    'ann-router'
    """
    mcp = FastApiMCP(app, include_operations=_EXPOSED_OPERATIONS)
    mcp.mount_http()
    return mcp


# A fresh app (not ann_router.api's module-level one) with the MCP server
# mounted at /mcp, so `uvicorn ann_router.mcp_server:app` works out of the box.
app = create_app()
build_server(app)


def main() -> None:
    """Run the MCP-mounted app over HTTP (the ``python -m ann_router.mcp_server`` path)."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8019)


if __name__ == "__main__":  # pragma: no cover
    main()
