"""Tests for the CLI doors — argparse (always on) and the click twin.

The argparse CLI must work with zero optional dependencies; the click twin is
skipped when the ``[cli]`` extra is absent. Both drive the same core, so a
build/search round-trip through the CLI exercises the whole persistence path end
to end.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ann_router import cli_argparse


def _run(capsys, argv: list[str]) -> str:
    """Drive ``cli_argparse.main(argv)`` and capture its stdout.

    Parameters
    ----------
    capsys : pytest fixture
        Captures stdout/stderr.
    argv : list of str
        Argument vector to pass to ``main``.

    Returns
    -------
    str
        Captured stdout (JSON or Markdown, depending on the command).
    """
    # Drive main() and capture stdout (JSON or Markdown) for assertions.
    assert cli_argparse.main(argv) == 0
    return capsys.readouterr().out


def test_route_json(capsys) -> None:
    """``ann-router route`` on a tiny corpus prints a JSON decision for exact."""
    out = _run(capsys, ["route", "--n-vectors", "500", "--dim", "16"])
    payload = json.loads(out)
    assert payload["backend"] == "exact"


def test_route_markdown(capsys) -> None:
    """``ann-router route --markdown`` prints a Markdown report."""
    out = _run(
        capsys, ["route", "--n-vectors", "500000", "--dim", "768", "--dynamic", "--markdown"]
    )
    assert out.startswith("# ann-router decision:")


def test_capabilities_lists_exact(capsys) -> None:
    """``ann-router capabilities`` lists exact among the available backends."""
    out = _run(capsys, ["capabilities"])
    assert "exact" in json.loads(out)["available"]


def test_bench_runs(capsys) -> None:
    """``ann-router bench`` reports recall 1.0 for the exact backend."""
    out = _run(capsys, ["bench", "--n", "1200", "--dim", "32", "-k", "5"])
    results = json.loads(out)["results"]
    assert results["exact"]["recall"] == 1.0


def test_build_search_round_trip(capsys, tmp_path) -> None:
    """``ann-router build`` then ``search`` round-trips a persisted index."""
    # Persist a corpus, build via the CLI, then search it — the full data path.
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((500, 32)).astype(np.float32)
    vpath = str(tmp_path / "vecs.npy")
    np.save(vpath, vecs)
    ipath = str(tmp_path / "idx")
    build_out = json.loads(
        _run(
            capsys,
            ["build", "--n-vectors", "500", "--dim", "32", "--vectors", vpath, "--index", ipath],
        )
    )
    assert build_out["backend"] == "exact"

    qpath = str(tmp_path / "q.npy")
    np.save(qpath, vecs[:2])
    search_out = json.loads(
        _run(capsys, ["search", "--index", ipath, "--queries", qpath, "-k", "5"])
    )
    assert len(search_out["ids"]) == 2
    assert search_out["ids"][0][0] == 0  # a point's nearest neighbour is itself


def test_click_twin_drives_every_subcommand(tmp_path) -> None:
    """The click CLI exposes route/capabilities/bench/build/search, matching argparse."""
    # The click CLI is optional; when present it must expose the same commands
    # as argparse (route/build/search/bench/capabilities), not just route.
    pytest.importorskip("click")
    from click.testing import CliRunner

    from ann_router import cli_click

    runner = CliRunner()

    def invoke(*args) -> dict:
        """Run a click subcommand and parse its JSON stdout.

        Parameters
        ----------
        *args
            The subcommand and its flags, e.g. ``"route", "--n-vectors", "500"``.

        Returns
        -------
        dict
            The parsed JSON output.
        """
        result = runner.invoke(cli_click.cli, list(args))
        assert result.exit_code == 0, result.output
        return json.loads(result.output)

    assert invoke("route", "--n-vectors", "500", "--dim", "16")["backend"] == "exact"
    assert "exact" in invoke("capabilities")["available"]
    assert (
        invoke("bench", "--n", "1200", "--dim", "32", "-k", "5")["results"]["exact"]["recall"]
        == 1.0
    )

    rng = np.random.default_rng(4)
    vecs = rng.standard_normal((500, 16)).astype(np.float32)
    vpath, ipath = str(tmp_path / "v.npy"), str(tmp_path / "idx")
    np.save(vpath, vecs)
    build_out = invoke(
        "build", "--n-vectors", "500", "--dim", "16", "--vectors", vpath, "--index", ipath
    )
    assert build_out["backend"] == "exact"

    qpath = str(tmp_path / "q.npy")
    np.save(qpath, vecs[:2])
    search_out = invoke("search", "--index", ipath, "--queries", qpath, "-k", "5")
    assert search_out["ids"][0][0] == 0  # a point's nearest neighbour is itself
