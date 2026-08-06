"""click command-line interface — the optional, ergonomic CLI twin.

Identical behaviour to :mod:`ann_router.cli_argparse`, but built on click for a
nicer help/UX. click is an *optional* dependency (``pip install
'ann-router[cli]'``): the argparse CLI is the always-available one, so importing
this module lazily requires click and fails with an actionable message if it is
missing. Every command delegates to the shared :mod:`ann_router._core_cli`, so
the two CLIs can never drift apart.

Consumes: ``click`` (optional), ``ann_router._core_cli``, ``os_helper``.
Produces: :func:`cli` (the ``ann-router-click`` console-script entry point).

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import json
import logging

try:
    import click
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the click CLI needs the [cli] extra. Run: pip install 'ann-router[cli]' "
        "(the argparse CLI `ann-router` is always available without it)."
    ) from exc

import os_helper as osh

from . import _core_cli as core
from .detect import detect_hardware


def _emit(result: dict | str) -> None:
    """Print a core result as Markdown (str) or indented JSON (dict).

    Parameters
    ----------
    result : dict or str
        A core function's result — a JSON-ready dict, or a Markdown string.
    """
    click.echo(result if isinstance(result, str) else json.dumps(result, indent=2))


def _criteria(**kw: object) -> dict:
    """Assemble a Criteria mapping, auto-detecting hardware when unset.

    Parameters
    ----------
    **kw
        Criteria fields collected from the click options.

    Returns
    -------
    dict
        The same mapping, with ``hardware`` filled in if it was unset.
    """
    kw["hardware"] = kw.get("hardware") or detect_hardware()
    return kw


# Shared decorator stack for the routing criteria, applied to route/build.
def _criteria_options(func):
    """Attach the shared ``Criteria`` click options to a command.

    Parameters
    ----------
    func : Callable
        The click command function to decorate.

    Returns
    -------
    Callable
        ``func``, wrapped with every criteria option.
    """
    # Declared once and reused so route and build expose an identical surface.
    opts = [
        click.option("--n-vectors", type=int, required=True),
        click.option("--dim", type=int, required=True),
        click.option("--target-recall", type=float, default=0.95),
        click.option("--latency-budget-ms", type=float, default=10.0),
        click.option("--memory-budget-gb", type=float, default=None),
        click.option("--dynamic", is_flag=True),
        click.option("--metadata-filtering", is_flag=True),
        click.option(
            "--hardware", type=click.Choice(["cpu", "gpu", "apple_silicon"]), default=None
        ),
        click.option("--persistence", is_flag=True),
        click.option("--batch-queries", is_flag=True),
        click.option("--metric", type=click.Choice(["cosine", "l2", "ip"]), default="cosine"),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


# Every command below passes `help=` explicitly rather than let click derive it
# from the docstring: the docstring is full numpydoc (Parameters section and
# all, per house style), and click would otherwise print that raw RST — dashes,
# "Parameters" header and all — as the command's --help text.


@click.group()
@click.version_option(package_name="ann-router")
@click.option("-v", "--verbose", count=True, help="repeat for more logs")
def cli(verbose: int) -> None:
    """Route to the right ANN vector-search backend from measured criteria.

    Parameters
    ----------
    verbose : int
        ``-v`` repeat count; raises the log level (0=WARNING, 1=INFO, 2+=DEBUG).
    """
    # Same stderr-logging / clean-stdout contract as the argparse twin.
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    osh.init_logging(level=level, stdout=False)


@cli.command("route", help="Choose a backend and print the justified decision.")
@_criteria_options
@click.option("--markdown", is_flag=True, help="human-readable report")
def route_cmd(markdown: bool, **kw: object) -> None:
    """Choose a backend and print the justified decision.

    Parameters
    ----------
    markdown : bool
        Print a human-readable Markdown report instead of JSON.
    **kw
        The rest of the ``Criteria`` fields, from :func:`_criteria_options`.
    """
    _emit(core.do_route(_criteria(**kw), as_markdown=markdown))


@cli.command("build", help="Build and save an index from a vectors file.")
@_criteria_options
@click.option("--vectors", required=True, help="path to .npy corpus (n, dim)")
@click.option("--index", required=True, help="output index path")
@click.option("--backend", default=None, help="force a backend (skip routing)")
def build_cmd(vectors: str, index: str, backend: str | None, **kw: object) -> None:
    """Build and save an index from a vectors file.

    Parameters
    ----------
    vectors : str
        Path to an ``.npy`` corpus of shape ``(n, dim)``.
    index : str
        Destination index path.
    backend : str, optional
        Force a specific backend, bypassing the router.
    **kw
        The rest of the ``Criteria`` fields, from :func:`_criteria_options`.
    """
    _emit(core.do_build(vectors, index, _criteria(**kw), backend))


@cli.command("search", help="Search a saved index with query vectors.")
@click.option("--index", required=True, help="index path from `build`")
@click.option("--queries", required=True, help="path to .npy queries (q, dim)")
@click.option("-k", type=int, default=10)
def search_cmd(index: str, queries: str, k: int) -> None:
    """Search a saved index with query vectors.

    Parameters
    ----------
    index : str
        Path written by ``build``.
    queries : str
        Path to an ``.npy`` query matrix of shape ``(q, dim)``.
    k : int
        Neighbours per query.
    """
    _emit(core.do_search(index, queries, k))


@cli.command("bench", help="Benchmark installed backends vs exact ground truth.")
@click.option("--n", type=int, default=5000)
@click.option("--dim", type=int, default=128)
@click.option("-k", type=int, default=10)
def bench_cmd(n: int, dim: int, k: int) -> None:
    """Benchmark installed backends vs exact ground truth.

    Parameters
    ----------
    n : int
        Synthetic corpus size.
    dim : int
        Embedding dimensionality.
    k : int
        Neighbours per query.
    """
    _emit(core.do_bench(n=n, dim=dim, k=k))


@cli.command("capabilities")
def capabilities_cmd() -> None:
    """List backends, availability and capabilities."""
    _emit(core.do_capabilities())


if __name__ == "__main__":  # pragma: no cover
    cli()
