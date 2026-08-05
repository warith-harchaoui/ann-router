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

* ``coarse``  — a logarithmic ``n`` grid, just to bracket where crossovers live;
* ``bisect``  — binary search on ``n`` for one backend-pair crossover, seeded by
  a log-log **interpolation** of the coarse latency curves so we start near the
  answer;
* ``status``  — how much of the plan is done.

Nothing here mutates the shipped policy; :mod:`bench.calibrate` reads the YAML
and derives the justified thresholds.

Consumes: :mod:`bench.datagen`, :mod:`ann_router.registry`, PyYAML.
Produces: the ``coarse`` / ``bisect`` / ``status`` CLI and ``measure_cell``.

Author: Warith Harchaoui
"""

from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from ann_router.registry import available_backends, get_backend

from . import datagen

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
_BITS_LADDER = [2, 3, 4, 8]  # turbovec: a *build* knob, so each rung rebuilds.


def _set_hnsw_ef(idx, v: int) -> None:
    idx._index.set_ef(int(v))
    idx._ef = int(v)


def _set_faiss_nprobe(idx, v: int) -> None:
    import faiss

    try:
        faiss.extract_index_ivf(idx._index).nprobe = int(v)
    except Exception:  # not an IVF index (small n may be flat) — knob is a no-op
        idx._index.nprobe = int(v)


def _set_annoy_searchk(idx, v: int) -> None:
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
    # scann / qdrant / pgvector: measured only when importable & (for the servers)
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
    """Return the canonical string key for a cell in the store."""
    return f"{backend}|n{n}|d{dim}|k{k}|r{target}|{metric}|q{nq}|s{seed}"


def load_store() -> dict:
    """Load the measurements YAML (``{}`` if absent)."""
    if STORE.exists():
        return yaml.safe_load(STORE.read_text()) or {}
    return {}


def save_store(store: dict) -> None:
    """Atomically write the measurements YAML (crash-safe for day-long runs)."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=RESULTS, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(store, fh, sort_keys=True, allow_unicode=True)
    os.replace(tmp, STORE)


def _recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean fraction of the exact top-k recovered per query (ignores -1 pads)."""
    hits = 0
    for p, g in zip(pred, gt, strict=False):
        hits += len(set(p.tolist()) & set(g[:k].tolist()))
    return hits / (len(gt) * k)


def _latencies_ms(idx, queries: np.ndarray, k: int) -> tuple[float, float, float]:
    """Per-query search latency percentiles over the query set, in ms."""
    times = np.empty(len(queries))
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        idx.search(q.reshape(1, -1), k)
        times[i] = (time.perf_counter() - t0) * 1e3
    return float(np.percentile(times, 50)), float(np.percentile(times, 95)), float(times.mean())


def _index_bytes(idx) -> int | None:
    """Size on disk of a ``save``-round-trip, a clean proxy for index RAM."""
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
    """Build, sweep the quality knob to the target, and record — one cell."""
    queries, gt = datagen.ground_truth(n, dim, k, nq, seed)
    corpus = datagen.make_corpus(n, dim, seed)
    cls = get_backend(backend)
    strat = _SWEEP.get(backend, {"kind": "default"})

    best: dict | None = None  # first operating point meeting the target
    fallback: dict | None = None  # best-recall point if the target is never met

    def record(idx, knob, build_s):
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

    else:  # generic single default operating point (qdrant/pgvector/scann)
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
    """A logarithmic n grid (~2 points/decade) up to ``max_n``."""
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


def cmd_coarse(args) -> None:
    """Run coarse-grid cells (ascending n), bounded by budget — one 'piece'."""
    store = load_store()
    backends = args.backends or available_backends()
    ns = _coarse_ns(args.max_n)
    cells = [
        (b, n, d, args.k, r) for n in ns for d in args.dims for r in args.targets for b in backends
    ]
    t_start, done, ran = time.time(), 0, 0
    for b, n, d, k, r in cells:
        key = cell_id(b, n, d, k, r, args.metric, args.nq, args.seed)
        if key in store and store[key].get("status") in ("ok", "skipped"):
            done += 1
            continue
        if args.max_cells and ran >= args.max_cells:
            break
        if args.time_budget and time.time() - t_start > args.time_budget:
            print(f"[coarse] time budget {args.time_budget}s reached")
            break
        m = measure_cell(b, n, d, k, r, args.metric, args.nq, args.seed, store=store)
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

    Returns the predicted crossover ``n`` (int) or ``None`` if there aren't
    enough measured points to fit both curves.
    """

    def pts(backend):
        xs, ys = [], []
        for v in store.values():
            if (
                v.get("status") == "ok"
                and v["backend"] == backend
                and v["dim"] == dim
                and v["k"] == k
                and v["metric"] == metric
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


def cmd_bisect(args) -> None:
    """Binary-search the n where backend A's p50 crosses backend B's, at target.

    Seeds the bracket from the interpolated prediction, then refines by bisection
    (each step measures both backends at one n; cells are cached). Prints and
    records the crossover.
    """
    store = load_store()

    def p50(backend, n):
        m = measure_cell(
            backend, n, args.dim, args.k, args.target, args.metric, args.nq, args.seed, store=store
        )
        return m.latency_ms_p50 if m.status == "ok" else None

    pred = _fit_predict_crossover(
        store, args.a, args.b, args.dim, args.k, args.target, args.metric, args.nq, args.seed
    )
    lo, hi = args.lo, args.hi
    if pred and lo < pred < hi:
        print(f"[bisect] interpolation predicts crossover near n={pred:,}; bracketing it")
        lo, hi = max(args.lo, pred // 4), min(args.hi, pred * 4)

    def faster_a(n):  # True when A is faster than B at n (A wins below crossover)
        pa, pb = p50(args.a, n), p50(args.b, n)
        if pa is None or pb is None:
            return None
        return pa < pb

    base = faster_a(lo)
    for _ in range(args.rounds):
        mid = int(round(math.sqrt(lo * hi)))  # geometric midpoint (log scale)
        f = faster_a(mid)
        print(f"[bisect] n={mid:,}: {args.a} faster? {f}   bracket=[{lo:,}, {hi:,}]")
        if f is None:
            break
        if f == base:
            lo = mid
        else:
            hi = mid
    print(
        f"[bisect] {args.a} vs {args.b} @dim={args.dim},r={args.target}: "
        f"crossover n in [{lo:,}, {hi:,}]"
    )


def cmd_status(args) -> None:
    """Summarise how much of the coarse plan is measured."""
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


def build_parser() -> argparse.ArgumentParser:
    """Construct the ``coarse`` / ``bisect`` / ``status`` CLI."""
    p = argparse.ArgumentParser(description="ANN calibration harness (resumable).")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coarse", help="run a bounded piece of the coarse n-grid sweep")
    c.add_argument("--dims", type=int, nargs="+", default=[128, 384, 768])
    c.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.98])
    c.add_argument("--k", type=int, default=10)
    c.add_argument("--max-n", type=int, default=1_000_000, help="cap corpus size this run")
    c.add_argument("--max-cells", type=int, default=0, help="stop after N new cells (0=unbounded)")
    c.add_argument(
        "--time-budget", type=float, default=0, help="stop after S seconds (0=unbounded)"
    )
    c.add_argument("--backends", nargs="+", default=None)
    c.add_argument("--metric", default="cosine")
    c.add_argument("--nq", type=int, default=500)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=cmd_coarse)

    b = sub.add_parser("bisect", help="dichotomic crossover search between two backends")
    b.add_argument("--a", required=True, help="backend A (faster below the crossover)")
    b.add_argument("--b", required=True, help="backend B (faster above the crossover)")
    b.add_argument("--dim", type=int, default=768)
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--target", type=float, default=0.95)
    b.add_argument("--lo", type=int, default=1_000)
    b.add_argument("--hi", type=int, default=10_000_000)
    b.add_argument("--rounds", type=int, default=8)
    b.add_argument("--metric", default="cosine")
    b.add_argument("--nq", type=int, default=500)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_bisect)

    s = sub.add_parser("status", help="show sweep progress")
    s.set_defaults(func=cmd_status)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
