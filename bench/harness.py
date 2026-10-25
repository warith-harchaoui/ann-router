"""Resumable, YAML-backed measurement harness for ANN backend calibration.

The router's thresholds (``EXACT_MAX_N``, ``FAISS_MIN_N``, ...) must be *measured*,
not guessed. This module measures them by, for each cell
``(backend, n, dim, k, target_recall)``:

1. building the index once,
2. sweeping the backend's query-time quality knob (HNSW ``ef``, FAISS ``nprobe``,
   Annoy ``search_k``, turbovec ``bit_width``) up its ladder until the achieved
   recall@k against exact ground truth first meets ``target_recall``,
3. recording the *minimum* latency that hit the target, plus build time, index
   size, and the operating knob.

Every measured cell is appended to ``results/measurements.yaml`` and reused, so a
10M-scale sweep can be done in **pieces over days**: re-running only advances the
cells not yet present. To keep it affordable we go **coarse-then-dichotomic**:

* ``coarse``: a logarithmic ``n`` grid, just to bracket where crossovers live;
* ``bisect``: binary search on ``n`` for one backend-pair crossover, seeded by
  a log-log **interpolation** of the coarse latency curves so we start near the
  answer;
* ``status``: how much of the plan is done.

Nothing here mutates the shipped policy; :mod:`bench.calibrate` reads the YAML
and derives the justified thresholds.

Consumes: :mod:`bench.datagen`, :mod:`ann_router.registry`, ``click``, PyYAML.
Produces: the ``coarse`` / ``bisect`` / ``status`` / ``dry-run`` CLI and
``measure_cell``.

Author: Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import click
import numpy as np
import yaml

from ann_router.registry import available_backends, get_backend

from . import VariadicCommand, datagen

RESULTS = Path(__file__).resolve().parent / "results"
STORE = RESULTS / "measurements.yaml"

# How much RAM a single corpus may occupy before a cell is skipped (not run).
# 96 GiB machine -> 60 GiB leaves head-room for the index and OS.
MEM_BUDGET_BYTES = int(os.environ.get("ANN_BENCH_MEM_GB", "60")) * (1024**3)

# Query-time quality ladders, ascending = slower but higher recall. The sweep
# stops at the first rung that meets the target, so the ladder tops out high
# enough to reach 0.99 on hard cells.
_EF_LADDER = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
_NPROBE_LADDER = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
_SEARCHK_LADDER = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
_BITS_LADDER = [2, 3, 4]  # turbovec: a *build* knob (only 2/3/4 valid), so each rung rebuilds.


def _set_hnsw_ef(idx, v: int) -> None:
    """Set an HNSW index's query-time ``ef`` (search-width) knob.

    Parameters
    ----------
    idx : ann_router.backends.hnsw.HNSWIndex
        A built HNSW index.
    v : int
        The rung of ``_EF_LADDER`` to set.
    """
    idx._index.set_ef(int(v))
    idx._ef = int(v)


def _set_faiss_nprobe(idx, v: int) -> None:
    """Set a FAISS index's query-time ``nprobe`` (cells-probed) knob.

    Parameters
    ----------
    idx : ann_router.backends.faiss_backend.FaissIndex
        A built FAISS index.
    v : int
        The rung of ``_NPROBE_LADDER`` to set.
    """
    import faiss

    try:
        faiss.extract_index_ivf(idx._index).nprobe = int(v)
    except Exception:  # not an IVF index (small n may be flat) — knob is a no-op
        idx._index.nprobe = int(v)


def _set_annoy_searchk(idx, v: int) -> None:
    """Set an Annoy index's query-time ``search_k`` (nodes-inspected) knob.

    Parameters
    ----------
    idx : ann_router.backends.annoy_backend.AnnoyIndex
        A built Annoy index.
    v : int
        The rung of ``_SEARCHK_LADDER`` to set.
    """
    idx._search_k = int(v)


# Per-backend sweep strategy. "query" knobs reuse one built index; "build" knobs
# rebuild per rung. exact has no knob (it *is* the ground truth, recall 1.0).
_SWEEP = {
    "exact": {"kind": "none"},
    "hnsw": {"kind": "query", "ladder": _EF_LADDER, "set": _set_hnsw_ef, "build": {}},
    "faiss": {"kind": "query", "ladder": _NPROBE_LADDER, "set": _set_faiss_nprobe, "build": {}},
    "annoy": {
        "kind": "query",
        "ladder": _SEARCHK_LADDER,
        "set": _set_annoy_searchk,
        "build": {"n_trees": 50},
    },
    "turbovec": {"kind": "build", "ladder": _BITS_LADDER, "param": "bit_width"},
    # qdrant / pgvector: measured only when importable & (for the servers)
    # reachable; handled generically as a single default operating point below.
}


@dataclass
class Measurement:
    """One measured cell — the atomic row of the results store."""

    backend: str
    n: int
    dim: int
    k: int
    target_recall: float
    metric: str
    nq: int
    seed: int
    status: str = "ok"  # ok | skipped | error
    achieved_recall: float | None = None
    met_target: bool | None = None
    op_knob: float | None = None
    build_s: float | None = None
    index_bytes: int | None = None
    raw_bytes: int | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    latency_ms_mean: float | None = None
    qps_batch: float | None = None
    hardware: str = "unknown"
    note: str = ""
    ts: float = field(default_factory=time.time)


def cell_id(
    backend: str, n: int, dim: int, k: int, target: float, metric: str, nq: int, seed: int
) -> str:
    """Return the canonical string key for a cell in the store.

    Parameters
    ----------
    backend : str
        Registry name.
    n, dim, k : int
        Corpus size, dimensionality, neighbours per query.
    target : float
        Target recall@k.
    metric : str
        Distance metric.
    nq, seed : int
        Query count and base seed.

    Returns
    -------
    str
        The cell's canonical key, e.g. ``"exact|n1000|d128|k10|r0.95|cosine|q500|s0"``.
    """
    return f"{backend}|n{n}|d{dim}|k{k}|r{target}|{metric}|q{nq}|s{seed}"


def load_store() -> dict:
    """Load the measurements YAML (``{}`` if absent)."""
    if STORE.exists():
        return yaml.safe_load(STORE.read_text()) or {}
    return {}


def save_store(store: dict) -> None:
    """Atomically write the measurements YAML (crash-safe for day-long runs).

    Parameters
    ----------
    store : dict
        The full measurements mapping to persist.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=RESULTS, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(store, fh, sort_keys=True, allow_unicode=True)
    os.replace(tmp, STORE)


def _recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean fraction of the exact top-k recovered per query (ignores -1 pads).

    Parameters
    ----------
    pred : numpy.ndarray
        Shape ``(nq, k)`` a backend's predicted neighbour ids.
    gt : numpy.ndarray
        Shape ``(nq, k)`` the exact ground-truth neighbour ids.
    k : int
        Neighbours per query.

    Returns
    -------
    float
        Mean recall@k over all queries.
    """
    hits = 0
    for p, g in zip(pred, gt, strict=False):
        hits += len(set(p.tolist()) & set(g[:k].tolist()))
    return hits / (len(gt) * k)


def _latencies_ms(idx, queries: np.ndarray, k: int) -> tuple[float, float, float]:
    """Per-query search latency percentiles over the query set, in ms.

    Parameters
    ----------
    idx : ann_router.base.ANNIndex
        A built index to query.
    queries : numpy.ndarray
        Shape ``(nq, dim)``.
    k : int
        Neighbours per query.

    Returns
    -------
    p50, p95, mean : float
        Latency percentiles/mean, in milliseconds.
    """
    times = np.empty(len(queries))
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        idx.search(q.reshape(1, -1), k)
        times[i] = (time.perf_counter() - t0) * 1e3
    return float(np.percentile(times, 50)), float(np.percentile(times, 95)), float(times.mean())


def _index_bytes(idx) -> int | None:
    """Size on disk of a ``save``-round-trip, a clean proxy for index RAM.

    Parameters
    ----------
    idx : ann_router.base.ANNIndex
        A built index.

    Returns
    -------
    int or None
        Bytes on disk, or ``None`` if ``save`` failed or wrote nothing.
    """
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "idx")
            idx.save(path)
            p = Path(path)
            if p.is_file():
                return p.stat().st_size
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) or None
    except Exception:
        return None


def _detect_hw() -> str:
    """Classify the local hardware, never raising.

    Returns
    -------
    str
        ``"gpu"``/``"apple_silicon"``/``"cpu"``, or ``"unknown"`` on failure.
    """
    try:
        from ann_router.detect import detect_hardware

        return detect_hardware()
    except Exception:
        return "unknown"


def measure_cell(
    backend: str,
    n: int,
    dim: int,
    k: int,
    target: float,
    metric: str = "cosine",
    nq: int = 500,
    seed: int = 0,
    store: dict | None = None,
    force: bool = False,
) -> Measurement:
    """Measure (or fetch from cache) a single calibration cell.

    Parameters
    ----------
    backend : str
        Registry name (``exact``/``turbovec``/``hnsw``/``faiss``/``annoy``/...).
    n, dim, k : int
        Corpus size, dimensionality, neighbours per query.
    target : float
        Recall@k the backend must reach before its latency is recorded.
    metric : str, optional
        Distance metric (default ``"cosine"``).
    nq, seed : int, optional
        Query count and base seed.
    store : dict, optional
        In-memory results store; if given, a cached ``ok`` cell is returned and
        new results are written through to :data:`STORE`.
    force : bool, optional
        Re-measure even if a cached result exists.

    Returns
    -------
    Measurement
        The measured (or cached) row. ``status`` is ``skipped`` when the corpus
        exceeds the RAM budget or the backend is unavailable, ``error`` on an
        unexpected failure — never silently dropped.
    """
    key = cell_id(backend, n, dim, k, target, metric, nq, seed)
    if store is not None and not force and key in store and store[key].get("status") == "ok":
        return Measurement(**store[key])

    base = {
        "backend": backend,
        "n": n,
        "dim": dim,
        "k": k,
        "target_recall": target,
        "metric": metric,
        "nq": nq,
        "seed": seed,
        "raw_bytes": datagen.corpus_bytes(n, dim),
        "hardware": _detect_hw(),
    }

    if backend not in available_backends():
        m = Measurement(**base, status="skipped", note="backend unavailable on this machine")
    elif backend == "pgvector" and not os.environ.get("ANN_ROUTER_PG_DSN"):
        m = Measurement(
            **base,
            status="skipped",
            note="pgvector needs a live server: set ANN_ROUTER_PG_DSN (see bench run script)",
        )
    elif datagen.corpus_bytes(n, dim) > MEM_BUDGET_BYTES:
        gib = datagen.corpus_bytes(n, dim) / (1024**3)
        m = Measurement(
            **base,
            status="skipped",
            note=f"corpus {gib:.1f} GiB exceeds {MEM_BUDGET_BYTES / (1024**3):.0f} GiB budget",
        )
    else:
        try:
            m = _run_cell(base, backend, n, dim, k, target, metric, nq, seed)
        except Exception as exc:  # keep the sweep alive; record the failure
            m = Measurement(**base, status="error", note=f"{type(exc).__name__}: {exc}")

    if store is not None:
        store[key] = asdict(m)
        save_store(store)
    return m


def _run_cell(
    base: dict,
    backend: str,
    n: int,
    dim: int,
    k: int,
    target: float,
    metric: str,
    nq: int,
    seed: int,
) -> Measurement:
    """Build, sweep the quality knob to the target, and record — one cell.

    Parameters
    ----------
    base : dict
        The base :class:`Measurement` fields already known before running.
    backend : str
        Registry name.
    n, dim, k : int
        Corpus size, dimensionality, neighbours per query.
    target : float
        Recall@k target to sweep the knob toward.
    metric : str
        Distance metric.
    nq, seed : int
        Query count and base seed.

    Returns
    -------
    Measurement
        The measured cell, ``status="ok"``.
    """
    queries, gt = datagen.ground_truth(n, dim, k, nq, seed)
    corpus = datagen.make_corpus(n, dim, seed)
    cls = get_backend(backend)
    strat = _SWEEP.get(backend, {"kind": "default"})

    best: dict | None = None  # first operating point meeting the target
    fallback: dict | None = None  # best-recall point if the target is never met

    def record(idx, knob, build_s):
        """Score one operating point and update ``best``/``fallback``.

        Parameters
        ----------
        idx : ann_router.base.ANNIndex
            The built index at this operating point.
        knob : float or None
            The value of the backend's quality knob at this point.
        build_s : float
            Build time in seconds.
        """
        nonlocal best, fallback
        p50, p95, mean = _latencies_ms(idx, queries, k)
        pred, _ = idx.search(queries, k)
        rec = _recall_at_k(pred, gt, k)
        row = {
            "op_knob": knob,
            "achieved_recall": rec,
            "build_s": build_s,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "latency_ms_mean": mean,
            "index_bytes": _index_bytes(idx),
        }
        if fallback is None or rec > fallback["achieved_recall"]:
            fallback = row
        if rec >= target and best is None:
            best = row

    if strat["kind"] == "none":  # exact: one point, recall is 1.0 by definition
        t0 = time.perf_counter()
        idx = cls(dim=dim, metric=metric).build(corpus)
        record(idx, None, time.perf_counter() - t0)
        best = fallback  # exact always "meets" the target

    elif strat["kind"] == "query":  # build once, sweep the query knob
        t0 = time.perf_counter()
        idx = cls(dim=dim, metric=metric, **strat.get("build", {})).build(corpus)
        build_s = time.perf_counter() - t0
        for knob in strat["ladder"]:
            strat["set"](idx, knob)
            record(idx, knob, build_s)
            if best is not None:
                break

    elif strat["kind"] == "build":  # rebuild per rung (e.g. turbovec bit_width)
        for knob in strat["ladder"]:
            t0 = time.perf_counter()
            idx = cls(dim=dim, metric=metric, **{strat["param"]: knob}).build(corpus)
            record(idx, knob, time.perf_counter() - t0)
            if best is not None:
                break

    else:  # generic single default operating point (qdrant/pgvector)
        # Their query knobs aren't exposed through the ANNIndex surface, so we
        # measure one default operating point. `met_target` then honestly reflects
        # whether that default hit the target (record() only sets `best` if so).
        t0 = time.perf_counter()
        idx = cls(dim=dim, metric=metric).build(corpus)
        record(idx, None, time.perf_counter() - t0)

    del corpus
    chosen = best or fallback
    return Measurement(**base, status="ok", met_target=best is not None, **chosen)


# --------------------------------------------------------------------------- #
# CLI: coarse sweep, dichotomic crossover search, status.
# --------------------------------------------------------------------------- #


def _coarse_ns(max_n: int) -> list[int]:
    """A logarithmic n grid (~2 points/decade) up to ``max_n``.

    Parameters
    ----------
    max_n : int
        Largest corpus size to include.

    Returns
    -------
    list of int
        The grid values ``<= max_n``.
    """
    grid = [
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
    ]
    return [n for n in grid if n <= max_n]


def cmd_coarse(
    *,
    dims: Sequence[int],
    targets: Sequence[float],
    k: int,
    max_n: int,
    max_cells: int,
    time_budget: float,
    backends: Sequence[str],
    metric: str,
    nq: int,
    seed: int,
) -> None:
    """Run coarse-grid cells (ascending n), bounded by budget — one 'piece'.

    Parameters
    ----------
    dims, targets : sequence
        The dimensionality and recall-target axes to sweep over.
    k : int
        Neighbours per query.
    max_n : int
        Cap on corpus size for this run.
    max_cells : int
        Stop after this many *new* cells; ``0`` means unbounded.
    time_budget : float
        Stop after this many seconds; ``0`` means unbounded.
    backends : sequence of str
        Which backends to measure; empty means every available one.
    metric : str
        Distance metric.
    nq, seed : int
        Query count and base seed, part of a cell's identity.
    """
    store = load_store()
    backends = list(backends) or available_backends()
    ns = _coarse_ns(max_n)
    cells = [(b, n, d, k, r) for n in ns for d in dims for r in targets for b in backends]
    t_start, done, ran = time.time(), 0, 0
    for b, n, d, k, r in cells:
        key = cell_id(b, n, d, k, r, metric, nq, seed)
        if key in store and store[key].get("status") in ("ok", "skipped"):
            done += 1
            continue
        if max_cells and ran >= max_cells:
            break
        if time_budget and time.time() - t_start > time_budget:
            print(f"[coarse] time budget {time_budget}s reached")
            break
        m = measure_cell(b, n, d, k, r, metric, nq, seed, store=store)
        ran += 1
        print(
            f"[coarse] {b:9s} n={n:>9,} d={d} r={r} -> {m.status} "
            f"recall={m.achieved_recall} p50={m.latency_ms_p50}ms knob={m.op_knob}"
        )
    print(f"[coarse] ran {ran} new cell(s); {done} already present; store={STORE}")


def _fit_predict_crossover(
    store: dict, a: str, b: str, dim: int, k: int, target: float, metric: str, nq: int, seed: int
) -> int | None:
    """Log-log interpolate each backend's p50(n) and predict where they cross.

    Parameters
    ----------
    store : dict
        The full measurements mapping.
    a, b : str
        The two backends to compare (crossover is where ``a`` and ``b``'s
        fitted p50(n) lines meet).
    dim, k : int
        Dimensionality and neighbours-per-query to filter measured points by.
    target : float
        Recall@k target to filter measured points by — the fit compares each
        backend's p50(n) at this *same* operating point, since a backend's
        latency to reach a given recall shifts with the target.
    metric : str
        Distance metric to filter measured points by.
    nq, seed : int
        Query count and base seed to filter measured points by, so both
        curves are fit from the same measurement configuration.

    Returns
    -------
    int or None
        The predicted crossover ``n``, or ``None`` if there aren't enough
        measured points (``< 2``) to fit both backends' curves.
    """

    def pts(backend):
        """Return ``(log(n), log(p50))`` measured points for ``backend``.

        Parameters
        ----------
        backend : str
            The backend to collect points for.

        Returns
        -------
        xs, ys : numpy.ndarray
            Log-corpus-size and log-p50-latency arrays, same length.
        """
        xs, ys = [], []
        for v in store.values():
            if (
                v.get("status") == "ok"
                and v["backend"] == backend
                and v["dim"] == dim
                and v["k"] == k
                and v["target_recall"] == target
                and v["metric"] == metric
                and v["nq"] == nq
                and v["seed"] == seed
                and v.get("latency_ms_p50")
            ):
                xs.append(math.log(v["n"]))
                ys.append(math.log(v["latency_ms_p50"]))
        return np.array(xs), np.array(ys)

    xa, ya = pts(a)
    xb, yb = pts(b)
    if len(xa) < 2 or len(xb) < 2:
        return None
    (sa, ia), (sb, ib) = np.polyfit(xa, ya, 1), np.polyfit(xb, yb, 1)
    if abs(sa - sb) < 1e-9:
        return None
    x_cross = (ib - ia) / (sa - sb)  # ln(n) where the two log-lines meet
    return int(round(math.exp(x_cross)))


def cmd_bisect(
    *,
    a: str,
    b: str,
    dim: int,
    k: int,
    target: float,
    lo: int,
    hi: int,
    rounds: int,
    metric: str,
    nq: int,
    seed: int,
) -> None:
    """Binary-search the n where backend A's p50 crosses backend B's, at target.

    Seeds the bracket from the interpolated prediction, then refines by bisection
    (each step measures both backends at one n; cells are cached). Prints and
    records the crossover.

    Parameters
    ----------
    a, b : str
        The two backends: ``a`` is the one expected to be faster below the
        crossover.
    dim, k : int
        Dimensionality and neighbours per query.
    target : float
        Recall@k target both backends are measured at.
    lo, hi : int
        The initial bracket on corpus size.
    rounds : int
        How many bisection steps to take.
    metric : str
        Distance metric.
    nq, seed : int
        Query count and base seed, part of a cell's identity.
    """
    store = load_store()

    def p50(backend, n):
        """Return backend's measured p50 latency at corpus size ``n``.

        Parameters
        ----------
        backend : str
            The backend to measure.
        n : int
            Corpus size.

        Returns
        -------
        float or None
            p50 latency in ms, or ``None`` if the cell errored/was skipped.
        """
        m = measure_cell(backend, n, dim, k, target, metric, nq, seed, store=store)
        return m.latency_ms_p50 if m.status == "ok" else None

    pred = _fit_predict_crossover(store, a, b, dim, k, target, metric, nq, seed)
    if pred and lo < pred < hi:
        print(f"[bisect] interpolation predicts crossover near n={pred:,}; bracketing it")
        lo, hi = max(lo, pred // 4), min(hi, pred * 4)

    def faster_a(n):
        """Return whether backend A is faster than B at corpus size ``n``.

        Parameters
        ----------
        n : int
            Corpus size to compare at.

        Returns
        -------
        bool or None
            ``True`` if A's p50 < B's p50, ``None`` if either is unmeasurable.
        """
        pa, pb = p50(a, n), p50(b, n)
        if pa is None or pb is None:
            return None
        return pa < pb

    base = faster_a(lo)
    for _ in range(rounds):
        mid = int(round(math.sqrt(lo * hi)))  # geometric midpoint (log scale)
        f = faster_a(mid)
        print(f"[bisect] n={mid:,}: {a} faster? {f}   bracket=[{lo:,}, {hi:,}]")
        if f is None:
            break
        if f == base:
            lo = mid
        else:
            hi = mid
    print(f"[bisect] {a} vs {b} @dim={dim},r={target}: crossover n in [{lo:,}, {hi:,}]")


def cmd_dryrun(
    *,
    ns: Sequence[int],
    dims: Sequence[int],
    targets: Sequence[float],
    k: int,
    backends: Sequence[str],
    metric: str,
    nq: int,
    seed: int,
    allow_fail: bool,
) -> None:
    """Fast, exhaustive-over-backends sanity sweep — catches breakage cheaply.

    Runs every available backend (plus any explicitly requested) on a tiny grid
    (small ``n``/``dim``/``nq``) so the whole thing finishes in seconds, not
    hours. Results are never written to :data:`STORE` — this is a smoke test,
    not a measurement — so it is safe to run before, after, or during a real
    sweep without disturbing cached cells.

    A backend is flagged ``FAIL`` when it crashes (``status == "error"``) or
    when it "succeeds" with recall far below what a trivial corpus should give
    (``< 0.5`` — the threshold that would have caught the annoy/py3.13 wheel
    bug this mode was built to catch: a broken environment silently recording
    near-zero recall as if it were a real measurement). ``skipped`` (missing
    dependency, no DSN, ...) is reported but not a failure.

    Parameters
    ----------
    ns, dims, targets : sequence
        The tiny grid to sweep; each empty sequence falls back to its default
        (``200 2000`` / ``16 128`` / ``0.9``).
    k : int
        Neighbours per query.
    backends : sequence of str
        Which backends to smoke-test; empty means every available one.
    metric : str
        Distance metric.
    nq, seed : int
        Query count and base seed.
    allow_fail : bool
        Exit 0 even when a backend FAILed, for CI triage.

    Raises
    ------
    SystemExit
        If any backend FAILed and ``allow_fail`` is not set.
    """
    ns = list(ns) or [200, 2_000]
    dims = list(dims) or [16, 128]
    targets = list(targets) or [0.9]
    backends = list(backends) or available_backends()

    rows: list[tuple[str, int, int, str, float | None, str]] = []
    for b in sorted(backends):
        for n in ns:
            for d in dims:
                for r in targets:
                    m = measure_cell(b, n, d, k, r, metric, nq, seed)
                    rows.append((b, n, d, m.status, m.achieved_recall, m.note))

    fails, warns = [], []
    print(f"{'backend':10s} {'n':>7s} {'dim':>5s}  status   recall  note")
    for b, n, d, status, recall, note in rows:
        bad = status == "error" or (status == "ok" and (recall or 0.0) < 0.5)
        flag = "FAIL" if bad else ("WARN" if status == "skipped" else "PASS")
        (fails if flag == "FAIL" else warns if flag == "WARN" else []).append((b, n, d))
        recall_s = f"{recall:.3f}" if recall is not None else "   -"
        print(f"{b:10s} {n:>7,d} {d:>5d}  {flag:<4s} {recall_s}  {note}")

    print(
        f"\n[dry-run] {len(rows)} cells, {len(warns)} skipped, {len(fails)} FAILED "
        f"across {len({b for b, *_ in rows})} backend(s)"
    )
    if fails:
        broken = sorted({b for b, *_ in fails})
        print(f"[dry-run] broken backends: {broken}")
        if not allow_fail:
            raise SystemExit(1)


def cmd_status() -> None:
    """Summarise how much of the coarse plan is measured.

    Takes nothing: ``status`` has no flags, and it no longer has to accept an
    unused namespace to share a signature with the other subcommands — each
    click command below passes its own options by name.
    """
    store = load_store()
    by_status: dict[str, int] = {}
    by_backend: dict[str, int] = {}
    for v in store.values():
        by_status[v.get("status", "?")] = by_status.get(v.get("status", "?"), 0) + 1
        if v.get("status") == "ok":
            by_backend[v["backend"]] = by_backend.get(v["backend"], 0) + 1
    print(f"store: {STORE}  ({len(store)} cells)")
    print("by status :", by_status or "empty")
    print("ok/backend:", by_backend or "empty")
    print("available backends:", available_backends())


# ``-h`` beside ``--help``. argparse supplied the short form for free and
# click does not, so a conversion that forgets this line silently takes it
# away; `ann_router/cli.py` spells it the same way.
_CONTEXT: dict[str, object] = {"help_option_names": ["-h", "--help"]}

# Every command below passes `help=` explicitly instead of letting click derive
# it from the docstring, and cuts the docstring with a `\f` besides: the
# docstrings are full numpydoc, which click would otherwise reflow into the
# user's --help, "Parameters ----------" and all. Same convention as
# `ann_router/cli.py`.


@click.group(context_settings=_CONTEXT)
def cli() -> None:
    """ANN calibration harness (resumable)."""


# `multiple=True` plus `cls=VariadicCommand` is what keeps argparse's
# `nargs="+"` spelling alive: `--dims 128 384 768` in one occurrence. Plain
# `multiple=True` would bind only `128` and leave `384 768` as stray
# positionals -- a sweep silently measuring a third of its grid. See
# `bench/expand_variadic`.
@cli.command("coarse", cls=VariadicCommand, help="run a bounded piece of the coarse n-grid sweep")
@click.option("--dims", type=int, multiple=True, default=(128, 384, 768), help="one or more")
@click.option(
    "--targets", type=float, multiple=True, default=(0.90, 0.95, 0.98), help="one or more"
)
@click.option("--k", type=int, default=10)
@click.option("--max-n", type=int, default=1_000_000, help="cap corpus size this run")
@click.option("--max-cells", type=int, default=0, help="stop after N new cells (0=unbounded)")
@click.option("--time-budget", type=float, default=0, help="stop after S seconds (0=unbounded)")
@click.option("--backends", multiple=True, help="one or more")
@click.option("--metric", default="cosine")
@click.option("--nq", type=int, default=500)
@click.option("--seed", type=int, default=0)
def coarse_cmd(**kw: object) -> None:
    """Run a bounded piece of the coarse n-grid sweep.

    \f

    Parameters
    ----------
    **kw
        The options declared above, forwarded verbatim to :func:`cmd_coarse`.
    """
    cmd_coarse(**kw)  # type: ignore[arg-type]


@cli.command("bisect", help="dichotomic crossover search between two backends")
@click.option("--a", required=True, help="backend A (faster below the crossover)")
@click.option("--b", required=True, help="backend B (faster above the crossover)")
@click.option("--dim", type=int, default=768)
@click.option("--k", type=int, default=10)
@click.option("--target", type=float, default=0.95)
@click.option("--lo", type=int, default=1_000)
@click.option("--hi", type=int, default=10_000_000)
@click.option("--rounds", type=int, default=8)
@click.option("--metric", default="cosine")
@click.option("--nq", type=int, default=500)
@click.option("--seed", type=int, default=0)
def bisect_cmd(**kw: object) -> None:
    """Dichotomic crossover search between two backends.

    \f

    Parameters
    ----------
    **kw
        The options declared above, forwarded verbatim to :func:`cmd_bisect`.
    """
    cmd_bisect(**kw)  # type: ignore[arg-type]


@cli.command("status", help="show sweep progress")
def status_cmd() -> None:
    """Show sweep progress."""
    cmd_status()


@cli.command(
    "dry-run",
    cls=VariadicCommand,
    help="fast exhaustive-over-backends smoke test (never touches STORE)",
)
@click.option("--ns", type=int, multiple=True, help="one or more (default: 200 2000)")
@click.option("--dims", type=int, multiple=True, help="one or more (default: 16 128)")
@click.option("--targets", type=float, multiple=True, help="one or more (default: 0.9)")
@click.option("--k", type=int, default=5)
@click.option("--backends", multiple=True, help="one or more")
@click.option("--metric", default="cosine")
@click.option("--nq", type=int, default=50)
@click.option("--seed", type=int, default=0)
@click.option("--allow-fail", is_flag=True, help="exit 0 even if a backend fails (for CI triage)")
def dry_run_cmd(**kw: object) -> None:
    """Fast exhaustive-over-backends smoke test (never touches STORE).

    \f

    Parameters
    ----------
    **kw
        The options declared above, forwarded verbatim to :func:`cmd_dryrun`.
    """
    cmd_dryrun(**kw)  # type: ignore[arg-type]


def main() -> None:
    """Entry point for ``python -m bench.harness``: parse argv and dispatch."""
    cli()


if __name__ == "__main__":
    main()
