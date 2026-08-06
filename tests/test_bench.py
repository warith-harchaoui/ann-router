"""Tests for the bench/ calibration harness — datagen, harness, calibrate.

bench/ is dev tooling, not part of the shipped package, but its math is exactly
what the router's thresholds are calibrated from, so a silent bug here would
mean a silently wrong threshold. Every test here uses only the always-available
``exact`` backend (numpy, no optional dependency) and redirects every module's
disk-cache paths to ``tmp_path`` first — nothing here may touch the real
``bench/results/`` store or ground-truth cache.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pytest

from bench import calibrate, datagen, harness


@pytest.fixture(autouse=True)
def _isolated_bench_dirs(tmp_path, monkeypatch):
    """Redirect every bench disk path to ``tmp_path`` for every test in this file."""
    monkeypatch.setattr(datagen, "GT_DIR", tmp_path / "gt")
    monkeypatch.setattr(harness, "RESULTS", tmp_path)
    monkeypatch.setattr(harness, "STORE", tmp_path / "measurements.yaml")


# --------------------------------------------------------------------------- #
# datagen
# --------------------------------------------------------------------------- #


def test_make_corpus_is_deterministic_and_unit_norm() -> None:
    """make_corpus() is seed-deterministic, shape-correct, and unit-norm per row."""
    a = datagen.make_corpus(200, 16, seed=0)
    b = datagen.make_corpus(200, 16, seed=0)
    c = datagen.make_corpus(200, 16, seed=1)
    assert np.array_equal(a, b)  # same seed -> identical corpus
    assert not np.array_equal(a, c)  # different seed -> different corpus
    assert a.shape == (200, 16)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_corpus_bytes_is_exact_float32_footprint() -> None:
    """corpus_bytes() returns the exact n*dim*4 float32 byte count."""
    assert datagen.corpus_bytes(1_000, 128) == 1_000 * 128 * 4


def test_ground_truth_caches_to_disk_and_is_reused(tmp_path) -> None:
    """ground_truth() writes its cache once and reuses it on a repeat call."""
    queries1, gt1 = datagen.ground_truth(n=300, dim=8, k=5, nq=20, seed=0)
    cached = list((tmp_path / "gt").glob("*.npy"))
    assert len(cached) == 2  # queries + gt for this (n, dim, k, nq, seed)
    # A second call with the same key must hit the cache, not recompute.
    queries2, gt2 = datagen.ground_truth(n=300, dim=8, k=5, nq=20, seed=0)
    assert np.array_equal(queries1, queries2)
    assert np.array_equal(gt1, gt2)
    assert gt1.shape == (20, 5)


# --------------------------------------------------------------------------- #
# harness — pure helpers
# --------------------------------------------------------------------------- #


def test_cell_id_is_stable_and_unique() -> None:
    """cell_id() is stable for identical args and distinct when a field differs."""
    a = harness.cell_id("exact", 1000, 128, 10, 0.95, "cosine", 500, 0)
    b = harness.cell_id("exact", 1000, 128, 10, 0.90, "cosine", 500, 0)
    assert a == harness.cell_id("exact", 1000, 128, 10, 0.95, "cosine", 500, 0)
    assert a != b


def test_recall_at_k_counts_exact_overlap() -> None:
    """_recall_at_k() averages the per-query exact-overlap fraction."""
    pred = np.array([[1, 2, 3], [4, 5, 6]])
    gt = np.array([[1, 2, 9], [4, 8, 9]])  # 2/3 hits, then 1/3 hits
    assert harness._recall_at_k(pred, gt, k=3) == pytest.approx((2 + 1) / 6)


def test_coarse_ns_caps_at_max_n() -> None:
    """_coarse_ns() returns an ascending grid capped at max_n."""
    ns = harness._coarse_ns(50_000)
    assert ns[-1] == 50_000
    assert all(n <= 50_000 for n in ns)
    assert ns == sorted(ns)  # ascending


def test_load_store_returns_empty_dict_when_absent() -> None:
    """load_store() returns {} when the store file doesn't exist yet."""
    assert harness.load_store() == {}


def test_save_store_then_load_store_round_trips(tmp_path) -> None:
    """save_store() then load_store() round-trips the exact same mapping."""
    store = {"exact|n1|d1|k1|r0.9|cosine|q1|s0": {"backend": "exact", "status": "ok"}}
    harness.save_store(store)
    assert harness.STORE.exists()
    assert harness.load_store() == store


# --------------------------------------------------------------------------- #
# harness — measure_cell / dry-run, exact backend only (always available)
# --------------------------------------------------------------------------- #


def test_measure_cell_exact_hits_recall_one() -> None:
    """measure_cell() on the exact backend always reports recall 1.0."""
    m = harness.measure_cell("exact", n=500, dim=16, k=5, target=0.95, nq=20, seed=0)
    assert m.status == "ok"
    assert m.achieved_recall == 1.0  # exact is exact by definition
    assert m.met_target is True
    assert m.latency_ms_p50 is not None and m.latency_ms_p50 >= 0


def test_measure_cell_never_writes_disk_without_a_store(tmp_path) -> None:
    """measure_cell() with store=None never touches STORE."""
    harness.measure_cell("exact", n=200, dim=8, k=5, target=0.9, nq=10, seed=0)
    assert not harness.STORE.exists()  # store=None -> in-memory only


def test_measure_cell_skips_unavailable_backend() -> None:
    """measure_cell() reports status='skipped' for an unregistered backend name."""
    m = harness.measure_cell("not-a-real-backend", n=200, dim=8, k=5, target=0.9)
    assert m.status == "skipped"
    assert "unavailable" in m.note


def test_dry_run_passes_for_exact(capsys) -> None:
    """cmd_dryrun() reports 0 FAILED for the always-correct exact backend."""
    args = argparse.Namespace(
        ns=[100, 300],
        dims=[8],
        targets=[0.9],
        k=5,
        backends=["exact"],
        metric="cosine",
        nq=10,
        seed=0,
        allow_fail=False,
    )
    harness.cmd_dryrun(args)  # must not raise (exact always clears the recall floor)
    out = capsys.readouterr().out
    assert "0 FAILED" in out
    assert not harness.STORE.exists()  # dry-run never touches the real store


def test_dry_run_fails_loudly_on_broken_recall(capsys) -> None:
    """cmd_dryrun() FAILs a backend that 'succeeds' with near-zero recall."""
    # A backend that "succeeds" with near-zero recall must FAIL the dry-run,
    # exactly the class of bug (broken annoy build) this mode exists to catch.
    args = argparse.Namespace(
        ns=[100],
        dims=[8],
        targets=[0.9],
        k=5,
        backends=["exact"],
        metric="cosine",
        nq=10,
        seed=0,
        allow_fail=True,
    )

    # exact can't actually fail recall, so patch measure_cell to simulate one that does.
    def _fake_measure_cell(backend, n, d, k, r, metric, nq, seed):
        """Stand in for measure_cell(), always reporting a near-zero recall."""
        return harness.Measurement(
            backend=backend,
            n=n,
            dim=d,
            k=k,
            target_recall=r,
            metric=metric,
            nq=nq,
            seed=seed,
            status="ok",
            achieved_recall=0.01,
        )

    orig = harness.measure_cell
    harness.measure_cell = _fake_measure_cell
    try:
        harness.cmd_dryrun(args)
    finally:
        harness.measure_cell = orig
    out = capsys.readouterr().out
    assert "1 FAILED" in out
    assert "FAIL" in out


def test_fit_predict_crossover_finds_a_plausible_n() -> None:
    """_fit_predict_crossover() finds a crossover inside the measured n range."""
    # Synthetic store: 'slow' backend has flat latency, 'fast' scales up with n —
    # they must cross somewhere inside the measured n range.
    store = {}
    for n in (1_000, 10_000, 100_000):
        store[f"slow|n{n}"] = {
            "status": "ok",
            "backend": "slow",
            "dim": 128,
            "k": 10,
            "metric": "cosine",
            "n": n,
            "latency_ms_p50": 5.0,
        }
        store[f"fast|n{n}"] = {
            "status": "ok",
            "backend": "fast",
            "dim": 128,
            "k": 10,
            "metric": "cosine",
            "n": n,
            "latency_ms_p50": 0.001 * n,
        }
    crossover = harness._fit_predict_crossover(
        store, "fast", "slow", dim=128, k=10, target=0.95, metric="cosine", nq=500, seed=0
    )
    assert crossover is not None
    assert 1_000 <= crossover <= 100_000


def test_cmd_status_reports_empty_then_populated_store(capsys) -> None:
    """cmd_status() reports 'empty' with no store, then real counts once populated."""
    harness.cmd_status(None)  # args is unused by this command
    assert "empty" in capsys.readouterr().out

    harness.save_store({"exact|n1": {"status": "ok", "backend": "exact"}})
    harness.cmd_status(None)
    out = capsys.readouterr().out
    assert "'ok': 1" in out
    assert "'exact': 1" in out


def test_build_parser_exposes_every_subcommand() -> None:
    """build_parser() wires coarse/bisect/status/dry-run, each to a callable func."""
    parser = harness.build_parser()
    for cmd in ("coarse", "bisect", "status", "dry-run"):
        ns = parser.parse_args([cmd] if cmd != "bisect" else [cmd, "--a", "x", "--b", "y"])
        assert ns.cmd == cmd
        assert callable(ns.func)


def test_main_dispatches_status_via_argv(monkeypatch, capsys) -> None:
    """main() parses sys.argv and dispatches to the right subcommand."""
    monkeypatch.setattr(sys, "argv", ["harness.py", "status"])
    harness.main()
    assert "store:" in capsys.readouterr().out


def test_fit_predict_crossover_needs_two_points_per_backend() -> None:
    """_fit_predict_crossover() returns None with fewer than 2 points for a backend."""
    store = {
        "fast|n1": {
            "status": "ok",
            "backend": "fast",
            "dim": 128,
            "k": 10,
            "metric": "cosine",
            "n": 1000,
            "latency_ms_p50": 1.0,
        }
    }
    assert (
        harness._fit_predict_crossover(
            store, "fast", "slow", dim=128, k=10, target=0.95, metric="cosine", nq=500, seed=0
        )
        is None
    )


# --------------------------------------------------------------------------- #
# calibrate — pure threshold-derivation math over a synthetic store
# --------------------------------------------------------------------------- #


def _row(backend, n, dim, target_recall, achieved_recall, p50, met_target=None, **extra):
    """Build one synthetic ``measurements.yaml``-shaped row for calibrate() tests.

    Parameters
    ----------
    backend : str
        Backend name.
    n, dim : int
        Corpus size, dimensionality.
    target_recall, achieved_recall : float
        Target and achieved recall@k.
    p50 : float
        p50 latency in ms.
    met_target : bool, optional
        Overrides the default ``achieved_recall >= target_recall`` derivation.
    **extra
        ``index_bytes``/``raw_bytes`` overrides.

    Returns
    -------
    dict
        A row matching the shape :func:`bench.calibrate.calibrate` expects.
    """
    return {
        "backend": backend,
        "n": n,
        "dim": dim,
        "k": 10,
        "target_recall": target_recall,
        "achieved_recall": achieved_recall,
        "latency_ms_p50": p50,
        "met_target": met_target if met_target is not None else achieved_recall >= target_recall,
        "op_knob": None,
        "index_bytes": extra.get("index_bytes"),
        "raw_bytes": extra.get("raw_bytes", n * dim * 4),
        "status": "ok",
    }


def test_exact_max_n_picks_the_largest_n_within_budget_and_unbeaten() -> None:
    """exact_max_n() picks the largest n within budget with no faster ANN."""
    rows = [
        _row("exact", 1_000, 128, 0.95, 1.0, 1.0),
        _row("exact", 10_000, 128, 0.95, 1.0, 8.0),
        _row("exact", 100_000, 128, 0.95, 1.0, 50.0),  # over budget -> excluded
        _row("hnsw", 10_000, 128, 0.95, 0.96, 20.0),  # slower than exact here
    ]
    result = calibrate.exact_max_n(rows, dim=128, budget_ms=10.0)
    assert result["value"] == 10_000
    assert result["evidence"]  # justified by the rows above, not a guess


def test_exact_max_n_none_when_never_measured() -> None:
    """exact_max_n() returns value=None for a dim with no measured rows."""
    result = calibrate.exact_max_n([], dim=999, budget_ms=10.0)
    assert result["value"] is None
    assert result["evidence"] == []


def test_faiss_min_n_picks_first_crossover() -> None:
    """faiss_min_n() picks the first n where FAISS beats HNSW's p50."""
    rows = [
        _row("faiss", 1_000, 128, 0.95, 0.96, 5.0, met_target=True),
        _row("hnsw", 1_000, 128, 0.95, 0.96, 1.0),
        _row("faiss", 100_000, 128, 0.95, 0.96, 0.5, met_target=True),
        _row("hnsw", 100_000, 128, 0.95, 0.96, 2.0),
    ]
    result = calibrate.faiss_min_n(rows, dim=128, target=0.95)
    assert result["value"] == 100_000  # first n where faiss beats hnsw


def test_high_recall_flags_the_lowest_missed_target() -> None:
    """high_recall() flags the lowest target a quantised backend missed."""
    rows = [
        _row("turbovec", 10_000, 128, 0.90, 0.92, 1.0, met_target=True),
        _row("turbovec", 10_000, 128, 0.98, 0.85, 1.0, met_target=False),
    ]
    result = calibrate.high_recall(rows, dim=128)
    assert result["value"] == 0.98


def test_compression_factors_is_the_middle_sorted_ratio() -> None:
    """compression_factors() returns the sorted[len//2] ratio per backend."""
    rows = [
        _row("turbovec", 1_000, 128, 0.9, 0.9, 1.0, index_bytes=1000, raw_bytes=16000),  # 16x
        _row("turbovec", 2_000, 128, 0.9, 0.9, 1.0, index_bytes=2000, raw_bytes=16000),  # 8x
    ]
    out = calibrate.compression_factors(rows)
    # sorted([8.0, 16.0])[len//2] == sorted([8.0, 16.0])[1] == 16.0
    assert out["turbovec"] == pytest.approx(16.0)


def test_calibrate_assembles_a_full_document_with_repo_relative_source() -> None:
    """calibrate() assembles counts/thresholds and a repo-relative source path."""
    store = {
        "exact|n1000": _row("exact", 1_000, 128, 0.95, 1.0, 1.0),
        "exact|n999_error": {"status": "error", "backend": "exact"},
    }
    doc = calibrate.calibrate(store, dims=[128])
    assert doc["n_cells"] == 2
    assert doc["measured_backends"] == ["exact"]
    assert not doc["source"].startswith("/")  # never an absolute, machine-specific path
    assert doc["source"] == "bench/results/measurements.yaml"
    assert 128 in doc["thresholds"]["EXACT_MAX_N"]


def test_mermaid_decision_tree_renders_every_backend() -> None:
    """mermaid_decision_tree() emits a classDef for every registered backend."""
    doc = calibrate.calibrate({"exact|n1": _row("exact", 1_000, 128, 0.95, 1.0, 1.0)}, dims=[128])
    md = calibrate.mermaid_decision_tree(doc)
    assert "```mermaid" in md
    for backend in ("exact", "turbovec", "faiss", "pgvector", "qdrant", "annoy", "hnsw"):
        assert f"classDef {backend} " in md
