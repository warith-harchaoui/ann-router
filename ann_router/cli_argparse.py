"""argparse command-line interface — the always-available CLI door.

This is the second of ann-router's five surfaces (after the library) and, like
os-helper's ``cli_argparse``, it needs no third-party dependency: argparse ships
with Python, so ``ann-router`` works the moment the package is installed. Every
subcommand is a thin shell over :mod:`ann_router._core_cli`; the click twin
(``ann-router-click``) drives the exact same core.

Subcommands: ``route``, ``build``, ``search``, ``bench``, ``capabilities``.

Consumes: ``ann_router._core_cli``, ``os_helper`` (logging to stderr).
Produces: :func:`main` (the ``ann-router`` console-script entry point).

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import os_helper as osh

from . import _core_cli as core


def _add_criteria_flags(p: argparse.ArgumentParser) -> None:
    """Attach the shared ``Criteria`` flags to a subparser.

    Parameters
    ----------
    p : argparse.ArgumentParser
        The subparser to attach flags to (mutated in place).
    """
    # These mirror the Criteria dataclass one-for-one so `route`/`build` accept
    # the full problem description on the command line.
    p.add_argument("--n-vectors", type=int, required=True, help="corpus size")
    p.add_argument("--dim", type=int, required=True, help="embedding dimensionality")
    p.add_argument("--target-recall", type=float, default=0.95)
    p.add_argument("--latency-budget-ms", type=float, default=10.0)
    p.add_argument("--memory-budget-gb", type=float, default=None)
    p.add_argument("--dynamic", action="store_true", help="frequent adds/removes")
    p.add_argument("--metadata-filtering", action="store_true")
    p.add_argument("--hardware", choices=["cpu", "gpu", "apple_silicon"], default=None)
    p.add_argument("--persistence", action="store_true")
    p.add_argument("--batch-queries", action="store_true")
    p.add_argument("--metric", choices=["cosine", "l2", "ip"], default="cosine")


def _criteria_from_args(args: argparse.Namespace) -> dict:
    """Fold parsed argparse flags into a ``Criteria`` mapping.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (from a parser built with :func:`_add_criteria_flags`).

    Returns
    -------
    dict
        A ``Criteria``-shaped mapping.
    """
    # Auto-detect the hardware when the user did not pin it, so routing reflects
    # the real machine (GPU/Apple-Silicon change the decision).
    from .detect import detect_hardware

    return {
        "n_vectors": args.n_vectors,
        "dim": args.dim,
        "target_recall": args.target_recall,
        "latency_budget_ms": args.latency_budget_ms,
        "memory_budget_gb": args.memory_budget_gb,
        "dynamic": args.dynamic,
        "metadata_filtering": args.metadata_filtering,
        "hardware": args.hardware or detect_hardware(),
        "persistence": args.persistence,
        "batch_queries": args.batch_queries,
        "metric": args.metric,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser with every subcommand.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser (used by :func:`main` and by the tests).

    Examples
    --------
    >>> parser = build_parser()
    >>> ns = parser.parse_args(["route", "--n-vectors", "500", "--dim", "16"])
    >>> ns.command
    'route'
    """
    parser = argparse.ArgumentParser(
        prog="ann-router",
        description="Route to the right ANN vector-search backend from measured criteria.",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="repeat for more logs")
    sub = parser.add_subparsers(dest="command", required=True)

    # route: decide + explain, no data needed.
    p_route = sub.add_parser("route", help="choose a backend and print the justified decision")
    _add_criteria_flags(p_route)
    p_route.add_argument("--markdown", action="store_true", help="human-readable report")

    # build: route + build + persist an index from an .npy file.
    p_build = sub.add_parser("build", help="build and save an index from a vectors file")
    _add_criteria_flags(p_build)
    p_build.add_argument("--vectors", required=True, help="path to .npy corpus (n, dim)")
    p_build.add_argument("--index", required=True, help="output index path")
    p_build.add_argument("--backend", default=None, help="force a backend (skip routing)")

    # search: query a persisted index.
    p_search = sub.add_parser("search", help="search a saved index with query vectors")
    p_search.add_argument("--index", required=True, help="index path from `build`")
    p_search.add_argument("--queries", required=True, help="path to .npy queries (q, dim)")
    p_search.add_argument("-k", type=int, default=10)

    # bench: recall/latency of every installed backend on synthetic data.
    p_bench = sub.add_parser("bench", help="benchmark installed backends vs exact ground truth")
    p_bench.add_argument("--n", type=int, default=5000)
    p_bench.add_argument("--dim", type=int, default=128)
    p_bench.add_argument("-k", type=int, default=10)

    # capabilities: the availability + capability matrix.
    sub.add_parser("capabilities", help="list backends, availability and capabilities")
    return parser


def _dispatch(args: argparse.Namespace) -> dict | str:
    """Route a parsed namespace to the matching core function.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments, including ``command``.

    Returns
    -------
    dict or str
        The core function's result (JSON-ready dict, or a Markdown string).
    """
    if args.command == "route":
        return core.do_route(_criteria_from_args(args), as_markdown=args.markdown)
    if args.command == "build":
        return core.do_build(args.vectors, args.index, _criteria_from_args(args), args.backend)
    if args.command == "search":
        return core.do_search(args.index, args.queries, args.k)
    if args.command == "bench":
        return core.do_bench(n=args.n, dim=args.dim, k=args.k)
    if args.command == "capabilities":
        return core.do_capabilities()
    raise ValueError(f"unknown command {args.command!r}")  # pragma: no cover


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``ann-router`` console script.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success).

    Examples
    --------
    >>> main(["route", "--n-vectors", "500", "--dim", "16"])  # doctest: +SKIP
    0
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # Logs go to stderr so stdout stays clean, pipeable JSON (a suite convention).
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    osh.init_logging(level=level, stdout=False)
    try:
        result = _dispatch(args)
    except Exception as err:  # noqa: BLE001 - last resort: a clean CLI error, not a traceback
        print(f"Error: {err}", file=sys.stderr)
        return 1
    # Markdown results print as-is; everything else is emitted as JSON on stdout.
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
