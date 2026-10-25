"""Tests for the command line — one door now, built with click.

There used to be two: an always-on argparse CLI and an optional click twin,
with every scenario written twice so the pair could be checked for drift. The
twin is the only CLI now, so the duplication is gone and each scenario is
asserted once, against the thing that ships.

A build/search round-trip through the CLI exercises the whole persistence path
end to end, which is why it lives here rather than in a unit test.

Author: Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from click.testing import CliRunner

from ann_router import cli


def _run(argv: list[str]) -> str:
    """Drive the click CLI and return its stdout.

    Parameters
    ----------
    argv : list of str
        The subcommand and its flags, e.g. ``["route", "--n-vectors", "500"]``.

    Returns
    -------
    str
        Captured stdout (JSON or Markdown, depending on the command).
    """
    result = CliRunner().invoke(cli.cli, argv)
    assert result.exit_code == 0, result.output
    return result.output


def test_route_json() -> None:
    """``ann-router route`` on a tiny corpus prints a JSON decision for exact."""
    payload = json.loads(_run(["route", "--n-vectors", "500", "--dim", "16"]))
    assert payload["backend"] == "exact"


def test_route_markdown() -> None:
    """``ann-router route --markdown`` prints a Markdown report."""
    out = _run(["route", "--n-vectors", "500000", "--dim", "768", "--dynamic", "--markdown"])
    assert out.startswith("# ann-router decision:")


def test_capabilities_lists_exact() -> None:
    """``ann-router capabilities`` lists exact among the available backends."""
    assert "exact" in json.loads(_run(["capabilities"]))["available"]


def test_bench_runs() -> None:
    """``ann-router bench`` reports recall 1.0 for the exact backend."""
    results = json.loads(_run(["bench", "--n", "1200", "--dim", "32", "-k", "5"]))["results"]
    assert results["exact"]["recall"] == 1.0


def test_build_search_round_trip(tmp_path) -> None:
    """``ann-router build`` then ``search`` round-trips a persisted index."""
    # Persist a corpus, build via the CLI, then search it — the full data path.
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((500, 32)).astype(np.float32)
    vpath = str(tmp_path / "vecs.npy")
    np.save(vpath, vecs)
    ipath = str(tmp_path / "idx")
    build_out = json.loads(
        _run(["build", "--n-vectors", "500", "--dim", "32", "--vectors", vpath, "--index", ipath])
    )
    assert build_out["backend"] == "exact"

    qpath = str(tmp_path / "q.npy")
    np.save(qpath, vecs[:2])
    search_out = json.loads(_run(["search", "--index", ipath, "--queries", qpath, "-k", "5"]))
    assert len(search_out["ids"]) == 2
    assert search_out["ids"][0][0] == 0  # a point's nearest neighbour is itself


def test_clean_error_on_invalid_criteria_not_a_traceback(capsys) -> None:
    """A library ValueError leaves the console script, not a raw traceback.

    ``n_vectors < 0`` is unconstrained by the parser, unlike ``--metric`` /
    ``--hardware`` which use a ``click.Choice``, so it reaches
    ``Criteria.validate()`` and used to surface as a Python traceback.

    This drives :func:`cli.main` with a patched ``sys.argv`` rather than
    ``CliRunner``, because ``CliRunner.invoke(cli, ...)`` bypasses ``main()``'s
    own wrapper — which is precisely the layer that turns the exception into one
    line on stderr. Driving the wrapper is the only way to assert it.
    """
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sys, "argv", ["ann-router", "route", "--n-vectors", "-1", "--dim", "16"])
    try:
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 1
    finally:
        monkeypatch.undo()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("Error: ")
    assert "n_vectors" in captured.err
