"""MCP server — the agent-tool door (optional ``[mcp]`` extra).

The fifth surface (paired with the ``skills/ann-router`` skill) lets an AI agent
call the router as Model-Context-Protocol tools: ``route``, ``capabilities`` and
``bench``. The ``mcp`` SDK is optional and imported lazily so the core package
stays dependency-free; every tool delegates to :mod:`ann_router._core_cli`, so an
agent, the CLI and the library all get identical answers.

Run it with::

    pip install 'ann-router[mcp]'
    python -m ann_router.mcp_server        # stdio transport

Consumes: ``mcp`` (optional), ``ann_router._core_cli``.
Produces: :func:`build_server`, :func:`main`.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

from typing import Any

from . import _core_cli as core


def build_server():
    """Construct the MCP server with the three ann-router tools registered.

    Returns
    -------
    mcp.server.fastmcp.FastMCP
        A server exposing ``route``, ``capabilities`` and ``bench``.

    Raises
    ------
    ImportError
        If the ``[mcp]`` extra is not installed.

    Examples
    --------
    >>> server = build_server()  # doctest: +SKIP
    >>> server.name  # doctest: +SKIP
    'ann-router'
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "the MCP server needs the [mcp] extra. Run: pip install 'ann-router[mcp]'"
        ) from exc

    server = FastMCP("ann-router")

    @server.tool()
    def route(
        n_vectors: int,
        dim: int,
        target_recall: float = 0.95,
        latency_budget_ms: float = 10.0,
        memory_budget_gb: float | None = None,
        dynamic: bool = False,
        metadata_filtering: bool = False,
        hardware: str = "cpu",
        persistence: bool = False,
        batch_queries: bool = False,
        metric: str = "cosine",
    ) -> dict[str, Any]:
        """Choose the best ANN backend for the described problem, with rationale."""
        # Forward the flat tool arguments straight into the shared core.
        criteria = {
            "n_vectors": n_vectors,
            "dim": dim,
            "target_recall": target_recall,
            "latency_budget_ms": latency_budget_ms,
            "memory_budget_gb": memory_budget_gb,
            "dynamic": dynamic,
            "metadata_filtering": metadata_filtering,
            "hardware": hardware,
            "persistence": persistence,
            "batch_queries": batch_queries,
            "metric": metric,
        }
        return core.do_route(criteria)

    @server.tool()
    def capabilities() -> dict[str, Any]:
        """List every backend, whether it is installed, and what it can do."""
        return core.do_capabilities()

    @server.tool()
    def bench(n: int = 5000, dim: int = 128, k: int = 10) -> dict[str, Any]:
        """Benchmark installed backends' recall@k against the exact ground truth."""
        return core.do_bench(n=n, dim=dim, k=k)

    return server


def main() -> None:
    """Run the MCP server over stdio (the ``python -m ann_router.mcp_server`` path)."""
    build_server().run()


if __name__ == "__main__":  # pragma: no cover
    main()
