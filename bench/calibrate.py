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
  exact/HNSW.
* memory compression factors — ``index_bytes / raw_bytes`` per backend, the
  evidence behind the tight-memory (Annoy / turbovec) branch.

ScaNN is not a registered backend (no Apple-Silicon wheel; dropped entirely —
see CHANGELOG.md), so it never appears here.

Consumes: ``results/measurements.yaml``, PyYAML.
Produces: ``results/calibrated_policy.yaml`` and a stdout summary. With
``--apply`` it would patch ``ann_router/policy.yaml`` (off by default).

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

RESULTS = Path(__file__).resolve().parent / "results"
STORE = RESULTS / "measurements.yaml"
OUT = RESULTS / "calibrated_policy.yaml"
TREE_OUT = RESULTS / "decision_tree.md"

INTERACTIVE_BUDGET_MS = 10.0  # the latency budget EXACT_MAX_N is defined against


def _ok(store: dict) -> list[dict]:
    """Return only the successfully measured rows of ``store``.

    Parameters
    ----------
    store : dict
        The full ``measurements.yaml`` mapping.

    Returns
    -------
    list of dict
        Rows whose ``status`` is ``"ok"``.
    """
    return [v for v in store.values() if v.get("status") == "ok"]


def _by(rows: list[dict], **eq: object) -> list[dict]:
    """Filter ``rows`` to those matching every ``key=value`` in ``eq``.

    Parameters
    ----------
    rows : list of dict
        Measured rows to filter.
    **eq
        Field/value pairs every returned row must match.

    Returns
    -------
    list of dict
        The matching subset, in the original order.
    """
    return [r for r in rows if all(r.get(kk) == vv for kk, vv in eq.items())]


def _evidence(r: dict) -> dict:
    """A compact, quotable subset of a measured row for the justification trail.

    Parameters
    ----------
    r : dict
        One measured row.

    Returns
    -------
    dict
        The row, narrowed to the fields worth quoting as evidence.
    """
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


def faiss_min_n(rows: list[dict], dim: int, target: float) -> dict:
    """Smallest n where FAISS meets the target with lower p50 than HNSW.

    Parameters
    ----------
    rows : list of dict
        All ``ok`` measurement rows.
    dim : int
        Dimensionality to calibrate for.
    target : float
        Recall@k target both backends must meet.

    Returns
    -------
    dict
        ``{value, method, evidence}`` — ``value`` is ``None`` if FAISS never won.
    """
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


def high_recall(rows: list[dict], dim: int) -> dict:
    """Recall target at/above which quantised backends stop meeting it.

    Parameters
    ----------
    rows : list of dict
        All ``ok`` measurement rows.
    dim : int
        Dimensionality to calibrate for.

    Returns
    -------
    dict
        ``{value, method, evidence}`` — ``value`` is ``None`` if no miss was observed.
    """
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


def compression_factors(rows: list[dict]) -> dict:
    """Median ``raw_bytes / index_bytes`` per backend — the memory evidence.

    Parameters
    ----------
    rows : list of dict
        All ``ok`` measurement rows.

    Returns
    -------
    dict
        ``{backend: ratio}`` for every backend with at least one measured ratio.
    """
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
    """Assemble the full calibrated-policy document from the store.

    Parameters
    ----------
    store : dict
        The full ``measurements.yaml`` mapping.
    dims : list of int
        Dimensionalities to derive thresholds for.

    Returns
    -------
    dict
        The calibrated-policy document (source, counts, thresholds, evidence).
    """
    rows = _ok(store)
    measured_backends = sorted({r["backend"] for r in rows})
    doc = {
        # Relative to the repo root, not absolute — this document is committed,
        # so an absolute path would leak the machine it was generated on and
        # churn on every re-run from a different clone location.
        "source": str(STORE.relative_to(RESULTS.parent.parent)),
        "n_cells": len(store),
        "measured_backends": measured_backends,
        "unmeasured_backends": [b for b in ("qdrant", "pgvector") if b not in measured_backends],
        "thresholds": {
            "EXACT_MAX_N": {d: exact_max_n(rows, d) for d in dims},
            "FAISS_MIN_N": {d: faiss_min_n(rows, d, 0.95) for d in dims},
            "HIGH_RECALL": {d: high_recall(rows, d) for d in dims},
        },
        "memory_compression_raw_over_index": compression_factors(rows),
    }
    return doc


# House palette (https://harchaoui.org/warith/colors/) — one colour per backend,
# reused verbatim in README.md/LISEZMOI.md's hand-authored tree so the generated
# and documented diagrams always match.
_PALETTE = {
    "exact": ("#808080", "#fff"),
    "turbovec": ("#AF52DE", "#fff"),
    "faiss": ("#FF9500", "#fff"),
    "pgvector": ("#28CD41", "#fff"),
    "qdrant": ("#79DBDC", "#003333"),
    "annoy": ("#FFCC00", "#3d2e00"),
    "hnsw": ("#007AFF", "#fff"),
    "decision": ("#F8F8F8", "#000000"),
}


def _fmt_threshold(per_dim: dict, key: str) -> str:
    """Render a per-dim calibrated value as ``d128=…, d384=…, d768=…`` for a label.

    Parameters
    ----------
    per_dim : dict
        ``{dim: {value, method, evidence}}`` (one of ``doc["thresholds"][...]``).
    key : str
        Unused — kept for a self-describing call site; the threshold name is
        implied by which ``per_dim`` the caller passes.

    Returns
    -------
    str
        A comma-joined ``d<dim>=<value>`` label.
    """
    parts = []
    for dim, entry in per_dim.items():
        v = entry.get("value") if isinstance(entry, dict) else entry
        parts.append(f"d{dim}={v if v is not None else '?'}")
    return ", ".join(parts)


def mermaid_decision_tree(doc: dict) -> str:
    """Render the calibrated decision tree as a coloured Mermaid flowchart.

    Mirrors the hand-authored diagram in README.md/LISEZMOI.md exactly (same
    node ids, same house palette) but with the diamond labels filled in from
    this run's calibrated thresholds instead of the generic threshold names —
    so re-running the sweep regenerates a diagram that reflects the actual
    measured numbers, per dim, rather than staying a static illustration.

    Parameters
    ----------
    doc : dict
        The document returned by :func:`calibrate` (has ``thresholds``).

    Returns
    -------
    str
        A ``## Calibrated decision tree`` Markdown section with a fenced
        ```mermaid``` flowchart, ready to write to :data:`TREE_OUT`.
    """
    t = doc["thresholds"]
    exact_max_n = _fmt_threshold(t["EXACT_MAX_N"], "EXACT_MAX_N")
    faiss_min_n = _fmt_threshold(t["FAISS_MIN_N"], "FAISS_MIN_N")

    class_defs = "\n".join(
        f"    classDef {name} fill:{fill},color:{color},stroke:{fill}"
        for name, (fill, color) in _PALETTE.items()
    )
    lines = [
        "## Calibrated decision tree",
        "",
        f"Generated by `bench.calibrate` from `{doc['source']}` ({doc['n_cells']} measured cells). "
        "Diamond labels carry this run's measured thresholds, one value per dim.",
        "",
        "```mermaid",
        "flowchart TD",
        '    Q[["n_vectors, dim, target_recall,<br/>dynamic, persistence, hardware..."]]',
        f"    Q --> D1{{n < EXACT_MAX_N?<br/><small>{exact_max_n}</small>}}",
        "    D1 -->|yes| EXACT([exact])",
        "    D1 -->|no| D2{frequent updates?}",
        "    D2 -->|yes| TURBOVEC([turbovec])",
        "    D2 -->|no| "
        f"D3{{n >= FAISS_MIN_N<br/>and GPU/batch?<br/><small>{faiss_min_n}</small>}}",
        "    D3 -->|yes| FAISS([faiss])",
        "    D3 -->|no| D4{persistence or<br/>metadata filtering?}",
        "    D4 -->|yes, DB in place| PGVECTOR([pgvector])",
        "    D4 -->|yes, no DB| QDRANT([qdrant])",
        "    D4 -->|no| D5{tight memory<br/>budget?}",
        "    D5 -->|yes| ANNOY([annoy])",
        "    D5 -->|no| HNSW([hnsw · default])",
        "",
        class_defs,
        "",
        "    class EXACT exact",
        "    class TURBOVEC turbovec",
        "    class FAISS faiss",
        "    class PGVECTOR pgvector",
        "    class QDRANT qdrant",
        "    class ANNOY annoy",
        "    class HNSW hnsw",
        "    class D1,D2,D3,D4,D5,Q decision",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    """Entry point for ``python -m bench.calibrate``: derive and write the thresholds."""
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
    TREE_OUT.write_text(mermaid_decision_tree(doc))
    print(f"wrote {OUT}")
    print(f"wrote {TREE_OUT}\n")
    print(yaml.safe_dump(doc["thresholds"], sort_keys=False, allow_unicode=True))
    if args.apply:
        print(
            "--apply is intentionally a no-op stub: review calibrated_policy.yaml, "
            "then bump ann_router/policy.yaml + POLICY_VERSION deliberately."
        )


if __name__ == "__main__":
    main()
