"""Tests for the CLI doors — argparse (always on) and the click twin.

The argparse CLI must work with zero optional dependencies; the click twin is
skipped when the ``[cli]`` extra is absent. Both drive the same core, so a
build/search round-trip through the CLI exercises the whole persistence path end
to end.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ann_router import cli_argparse


def _run(capsys, argv: list[str]) -> str:
    # Drive main() and capture stdout (JSON or Markdown) for assertions.
    assert cli_argparse.main(argv) == 0
    return capsys.readouterr().out


def test_route_json(capsys) -> None:
    out = _run(capsys, ["route", "--n-vectors", "500", "--dim", "16"])
    payload = json.loads(out)
    assert payload["backend"] == "exact"


def test_route_markdown(capsys) -> None:
    out = _run(
        capsys, ["route", "--n-vectors", "500000", "--dim", "768", "--dynamic", "--markdown"]
    )
    assert out.startswith("# ann-router decision:")


def test_capabilities_lists_exact(capsys) -> None:
    out = _run(capsys, ["capabilities"])
    assert "exact" in json.loads(out)["available"]


def test_bench_runs(capsys) -> None:
    out = _run(capsys, ["bench", "--n", "1200", "--dim", "32", "-k", "5"])
    results = json.loads(out)["results"]
    assert results["exact"]["recall"] == 1.0


def test_build_search_round_trip(capsys, tmp_path) -> None:
    # Persist a corpus, build via the CLI, then search it — the full data path.
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((2_000, 32)).astype(np.float32)
    vpath = str(tmp_path / "vecs.npy")
    np.save(vpath, vecs)
    ipath = str(tmp_path / "idx")
    build_out = json.loads(
        _run(
            capsys,
            ["build", "--n-vectors", "2000", "--dim", "32", "--vectors", vpath, "--index", ipath],
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
    # The click CLI is optional; when present it must expose the same commands
    # as argparse (route/build/search/bench/capabilities), not just route.
    pytest.importorskip("click")
    from click.testing import CliRunner

    from ann_router import cli_click

    runner = CliRunner()

    def invoke(*args) -> dict:
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
    vecs = rng.standard_normal((1_000, 16)).astype(np.float32)
    vpath, ipath = str(tmp_path / "v.npy"), str(tmp_path / "idx")
    np.save(vpath, vecs)
    build_out = invoke(
        "build", "--n-vectors", "1000", "--dim", "16", "--vectors", vpath, "--index", ipath
    )
    assert build_out["backend"] == "exact"

    qpath = str(tmp_path / "q.npy")
    np.save(qpath, vecs[:2])
    search_out = invoke("search", "--index", ipath, "--queries", qpath, "-k", "5")
    assert search_out["ids"][0][0] == 0  # a point's nearest neighbour is itself
