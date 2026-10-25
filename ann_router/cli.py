"""The ann-router command line, built with click.

The one command-line surface. It used to be the optional twin of an argparse
CLI; the two were kept in lock-step by hand, and a flag naming the parsing
library (``ann-router-click``) described the implementation rather than the
interface. Both are gone: ``click`` is now a core dependency and this module
is the whole CLI. Every command still delegates to the shared
:mod:`ann_router._core_cli`, which is what the API and MCP doors also call,
so no door can answer differently from another.

Consumes: ``click``, ``ann_router._core_cli``, ``os_helper``.
Produces: :func:`main` (the ``ann-router`` console-script entry point,
wrapping :func:`cli`).

Author: Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import logging
import sys

import click
import os_helper as osh

from . import __version__
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


# ``-h`` beside ``--help`` and ``-V`` beside ``--version``: the short forms are
# the ones people type, and the suite spells them the same way everywhere.
# ``-v`` is already --verbose, hence the capital for the version.
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
# Single-sourced from ``__version__``, not from the installed distribution
# metadata: an editable checkout reports whatever was last pip-installed, so
# ``-V`` would disagree with the package it is actually running.
@click.version_option(__version__, "-V", "--version", package_name="ann-router")
@click.option("-v", "--verbose", count=True, help="repeat for more logs")
def cli(verbose: int) -> None:
    """Route to the right ANN vector-search backend from measured criteria.

    \f

    Everything after the form feed is for readers of the source: click
    truncates ``--help`` there. Without it the numpydoc block below is
    rendered to users as one reflowed paragraph of "Parameters ----------".

    Parameters
    ----------
    verbose : int
        ``-v`` repeat count; raises the log level (0=WARNING, 1=INFO, 2+=DEBUG).
    """
    # Diagnostics to stderr, results to stdout: a pipe gets data, not logging.
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


def main() -> None:
    """Console entry point (``ann-router``).

    click's own dispatch only special-cases ``ClickException``/``Abort`` (and
    a broken pipe); a plain library exception (e.g. ``Criteria.validate()``'s
    ``ValueError``) would otherwise propagate as a raw Python traceback
    instead of a clean CLI error. This wraps the whole invocation and
    translates that last case into a one-line stderr message + exit 1 --
    click's own control flow (usage errors, ``--help``, an explicit
    ``sys.exit`` in a subcommand) already raises ``SystemExit``, a
    ``BaseException`` this does not catch, so it passes through untouched.
    """
    try:
        cli()
    except Exception as err:  # noqa: BLE001 - last resort: see docstring
        click.echo(f"Error: {err}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
