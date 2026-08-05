"""Derive justified routing thresholds from the measured YAML store.

Reads ``results/measurements.yaml`` (produced by :mod:`bench.harness`) and turns
it into ``results/calibrated_policy.yaml`` — the same threshold names the shipped
policy uses, but each value carries the *measured rows that justify it* and the
method used, so nothing is a guess.

Thresholds derived here
-----------------------
* ``EXACT_MAX_N`` — largest ``n`` (per dim) where the exact scan's p50 latency is
  still within the interactive budget **and** no approximate backend is faster:
  below it, approximation only adds build cost.
* ``FAISS_MIN_N`` — smallest ``n`` where FAISS meets the recall target with lower
  p50 latency than HNSW: the "very large volume" regime.
* ``HIGH_RECALL`` — the recall target at/above which a quantised backend
  (turbovec / FAISS-PQ) can no longer meet it, so routing must steer to
  exact/HNSW/ScaNN.
* memory compression factors — ``index_bytes / raw_bytes`` per backend, the
  evidence behind the tight-memory (Annoy / turbovec) branch.

Backends that could not be measured on this machine (e.g. ScaNN — no Apple
Silicon wheel) are reported as ``unmeasured`` with their heuristic retained, so
the gap is explicit rather than silently filled.

Consumes: ``results/measurements.yaml``, PyYAML.
Produces: ``results/calibrated_policy.yaml`` and a stdout summary. With
``--apply`` it would patch ``ann_router/policy.yaml`` (off by default).

Author: Warith Harchaoui
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

RESULTS = Path(__file__).resolve().parent / "results"
STORE = RESULTS / "measurements.yaml"
OUT = RESULTS / "calibrated_policy.yaml"

INTERACTIVE_BUDGET_MS = 10.0  # the latency budget EXACT_MAX_N is defined against


def _ok(store: dict) -> list[dict]:
    return [v for v in store.values() if v.get("status") == "ok"]


def _by(rows, **eq):
    return [r for r in rows if all(r.get(kk) == vv for kk, vv in eq.items())]


def _evidence(r: dict) -> dict:
    """A compact, quotable subset of a measured row for the justification trail."""
    return {
        kk: r.get(kk)
        for kk in (
            "backend",
            "n",
            "dim",
            "k",
            "target_recall",
            "achieved_recall",
            "latency_ms_p50",
            "op_knob",
            "index_bytes",
            "raw_bytes",
        )
    }


def exact_max_n(rows, dim: int, budget_ms: float = INTERACTIVE_BUDGET_MS) -> dict:
    """Largest n where exact is both within budget and not beaten by an ANN.

    Parameters
    ----------
    rows : list of dict
        All ``ok`` measurement rows.
    dim : int
        Dimensionality to calibrate for.
    budget_ms : float, optional
        Interactive per-query budget (default 10 ms).

    Returns
    -------
    dict
        ``{value, method, evidence}`` — ``value`` is ``None`` if exact was never
        measured at this dim.
    """
    ex = sorted(_by(rows, backend="exact", dim=dim), key=lambda r: r["n"])
    best_n, evid = None, []
    for r in ex:
        ann = [
            a
            for a in rows
            if a["dim"] == dim
            and a["backend"] != "exact"
            and a["n"] == r["n"]
            and a.get("latency_ms_p50")
        ]
        faster_ann = [a for a in ann if a["latency_ms_p50"] < r["latency_ms_p50"]]
        if r["latency_ms_p50"] is not None and r["latency_ms_p50"] <= budget_ms and not faster_ann:
            best_n = r["n"]
            evid = [_evidence(r)] + [_evidence(a) for a in ann]
    return {
        "value": best_n,
        "method": f"exact p50<= {budget_ms}ms and no faster ANN, dim={dim}",
        "evidence": evid,
    }


def faiss_min_n(rows, dim: int, target: float) -> dict:
    """Smallest n where FAISS meets the target with lower p50 than HNSW."""
    ns = sorted({r["n"] for r in rows if r["dim"] == dim})
    for n in ns:
        f = _by(rows, backend="faiss", dim=dim, n=n, target_recall=target)
        h = _by(rows, backend="hnsw", dim=dim, n=n, target_recall=target)
        if f and h and f[0].get("met_target") and f[0]["latency_ms_p50"] < h[0]["latency_ms_p50"]:
            return {
                "value": n,
                "method": f"first n where faiss p50<hnsw p50 at recall {target}, dim={dim}",
                "evidence": [_evidence(f[0]), _evidence(h[0])],
            }
    return {
        "value": None,
        "method": f"faiss never beat hnsw at recall {target}, dim={dim}",
        "evidence": [],
    }


def high_recall(rows, dim: int) -> dict:
    """Recall target at/above which quantised backends stop meeting it."""
    caps = []
    for backend in ("turbovec", "faiss"):
        for r in _by(rows, backend=backend, dim=dim):
            if r.get("met_target") is False:
                caps.append((r["target_recall"], _evidence(r)))
    if not caps:
        return {"value": None, "method": "no quantised miss observed", "evidence": []}
    caps.sort(key=lambda t: t[0])
    return {
        "value": caps[0][0],
        "method": f"lowest target a quantised backend missed, dim={dim}",
        "evidence": [c[1] for c in caps[:4]],
    }


def compression_factors(rows) -> dict:
    """Median ``raw_bytes / index_bytes`` per backend — the memory evidence."""
    out = {}
    for backend in sorted({r["backend"] for r in rows}):
        ratios = [
            r["raw_bytes"] / r["index_bytes"]
            for r in _by(rows, backend=backend)
            if r.get("index_bytes") and r.get("raw_bytes")
        ]
        if ratios:
            ratios.sort()
            out[backend] = round(ratios[len(ratios) // 2], 2)
    return out


def calibrate(store: dict, dims: list[int]) -> dict:
    """Assemble the full calibrated-policy document from the store."""
    rows = _ok(store)
    measured_backends = sorted({r["backend"] for r in rows})
    doc = {
        "source": str(STORE),
        "n_cells": len(store),
        "measured_backends": measured_backends,
        "unmeasured_backends": [
            b for b in ("scann", "qdrant", "pgvector") if b not in measured_backends
        ],
        "thresholds": {
            "EXACT_MAX_N": {d: exact_max_n(rows, d) for d in dims},
            "FAISS_MIN_N": {d: faiss_min_n(rows, d, 0.95) for d in dims},
            "HIGH_RECALL": {d: high_recall(rows, d) for d in dims},
        },
        "memory_compression_raw_over_index": compression_factors(rows),
        "notes": "ScaNN unmeasured: no Apple-Silicon wheel — heuristic retained "
        "(SCANN_MIN_N=1e6, SCANN_MIN_RECALL=0.98).",
    }
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 384, 768])
    ap.add_argument(
        "--apply",
        action="store_true",
        help="(guarded) patch ann_router/policy.yaml from the measured values",
    )
    args = ap.parse_args()

    if not STORE.exists():
        raise SystemExit(f"no measurements yet at {STORE} — run bench.harness coarse first")
    store = yaml.safe_load(STORE.read_text()) or {}
    doc = calibrate(store, args.dims)
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    print(f"wrote {OUT}\n")
    print(yaml.safe_dump(doc["thresholds"], sort_keys=False, allow_unicode=True))
    if args.apply:
        print(
            "--apply is intentionally a no-op stub: review calibrated_policy.yaml, "
            "then bump ann_router/policy.yaml + POLICY_VERSION deliberately."
        )


if __name__ == "__main__":
    main()
