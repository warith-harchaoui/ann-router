"""Calibration harness for ann-router's routing thresholds.

Turns the shipped policy from *guessed* to *measured*: :mod:`bench.harness`
sweeps backends across corpus sizes and records recall/latency into a resumable
YAML store, and :mod:`bench.calibrate` reads that store to derive each threshold
with the evidence that justifies it.

No LLM is involved anywhere in this package: it is pure numeric measurement
(numpy / faiss / hnswlib / annoy / turbovec). Everything runs locally.

Both command lines here are click-built, like ``ann-router`` itself. They share
:class:`VariadicCommand` below, which is what lets a sweep keep saying
``--dims 128 384 768``.
"""

from __future__ import annotations

import os
from collections.abc import Collection, Sequence

import click

# macOS + conda commonly link two OpenMP runtimes (faiss brings one, hnswlib /
# numpy another). When the second `libomp` initialises at first use, the process
# aborts with "OMP: Error #15 ... already initialized". Allowing the duplicate is
# the standard, widely-used workaround; set it here, before any backend loads, so
# `python -m bench.harness ...` runs without needing an env-var prefix. Override
# by exporting KMP_DUPLICATE_LIB_OK yourself if your environment has a single OMP.
#
# click is imported above it on purpose and is not a backend: it pulls in no
# OpenMP-linked extension, so the guarantee this line makes still holds.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _is_value(token: str) -> bool:
    """Return whether ``token`` is a value rather than the next flag.

    A bare ``-`` is a value, and so is a negative number: ``-0.5`` must not be
    mistaken for a one-character short option, or a future threshold flag
    taking a negative value would lose it.

    Parameters
    ----------
    token : str
        One raw argv entry.

    Returns
    -------
    bool
        ``True`` when the token should be consumed as a value.
    """
    if not token.startswith("-") or token == "-":
        return True
    try:
        float(token)
    except ValueError:
        return False
    return True


def expand_variadic(args: Sequence[str], greedy: Collection[str]) -> list[str]:
    """Rewrite ``--flag a b c`` as ``--flag a --flag b --flag c``.

    argparse's ``nargs="+"`` takes every value following one occurrence of the
    flag. click has no equivalent: ``multiple=True`` binds **one** value per
    occurrence, so ``--backends exact hnsw faiss`` would quietly keep only
    ``exact`` and treat the rest as stray positionals. Since the documented
    sweeps spell it that way — ``python -m bench.calibrate --dims 128 384
    768`` in ``bench/README.md`` and ``run_bench.sh`` — the spelling has to
    survive, and silently dropping two thirds of a sweep axis is the kind of
    bug that shows up as a wrong threshold rather than as an error.

    So the argv is normalised before click parses it. Nothing here touches
    click's internals: the rewrite is a pure function over a list of strings,
    applied by :class:`VariadicCommand` through the public
    :meth:`click.Command.parse_args` hook.

    ``--flag=a`` is passed through untouched, which is also what argparse did:
    the ``=`` form never consumed the tokens after it.

    Parameters
    ----------
    args : sequence of str
        The raw arguments click is about to parse.
    greedy : collection of str
        The option spellings that should swallow every value following them,
        e.g. ``{"--dims", "--targets"}``.

    Returns
    -------
    list of str
        The same arguments, with each greedy flag repeated once per value.
    """
    out: list[str] = []
    i = 0
    while i < len(args):
        token = args[i]
        i += 1
        out.append(token)
        if token not in greedy:
            continue
        # Collect the run of values argparse would have taken, then re-emit the
        # flag in front of each one but the first (already appended above). An
        # empty run falls through to click, which reports the missing value the
        # same way argparse reported "expected at least one argument".
        values: list[str] = []
        while i < len(args) and _is_value(args[i]):
            values.append(args[i])
            i += 1
        for position, value in enumerate(values):
            if position:
                out.append(token)
            out.append(value)
    return out


class VariadicCommand(click.Command):
    """A command whose ``multiple=True`` options accept argparse's spelling.

    Every such option on the command is treated as greedy, which is exactly
    the set that used to be declared ``nargs="+"``. See
    :func:`expand_variadic` for why this exists and what it costs.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        """Normalise the greedy flags, then parse as click normally would.

        Parameters
        ----------
        ctx : click.Context
            The context click built for this invocation.
        args : list of str
            The raw arguments, before any rewriting.

        Returns
        -------
        list of str
            Whatever click's own ``parse_args`` leaves unconsumed.
        """
        greedy = {
            opt
            for param in self.params
            if isinstance(param, click.Option) and param.multiple
            for opt in param.opts
        }
        return super().parse_args(ctx, expand_variadic(args, greedy))
