"""The selection policy — pure, tunable decision math (no side effects).

This is the ``score.py`` analogue from ``best-engine-ai-helper``: it holds the
named, versioned thresholds and the branch predicates that turn a
:class:`~ann_router.spec.Criteria` into an *ordered, justified* shortlist of
backends. It imports nothing heavy and touches no engine, so it is fully unit
-testable on its own — the router layer (``router.py``) is what consults
availability and instantiates the winner.

The decision tree reproduced here is the suite's own documented policy (and the
one already shipped, in two-branch form, inside the ``roitelet`` prototype):

1. ``n < EXACT_MAX_N`` (scaled by ``latency_budget_ms``, see
   :func:`effective_exact_max_n`)                -> **exact** (approximation is pointless)
2. else frequent updates                        -> **turbovec** (O(1) add/remove)
3. else very large volume + GPU/batch           -> **faiss** (IVF+PQ, scales)
4. else persistence + metadata filters          -> **qdrant / pgvector**
5. else read-only + tight memory                -> **annoy** (frozen, mmap, lean)
6. else stable in-memory corpus (the default)   -> **hnsw** (best recall/latency)

ScaNN was dropped: no Apple-Silicon wheel exists, and the project has
definitively abandoned it as a supported backend (see CHANGELOG.md).

Consumes: ``ann_router.spec``.
Produces: :func:`rank_backends`, plus the ``THRESHOLDS`` constants.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .spec import Criteria

# --- Versioned policy thresholds (rule-16-style: named, sourced, test-pinned) ---
# Bumping any of these is a policy change and should move POLICY_VERSION + a
# CHANGELOG entry, exactly like a model/threshold bump in the sibling repos.
POLICY_VERSION = "1.0.0"

# Below this corpus size a vectorised brute-force scan is already sub-millisecond
# and, being exact, has recall 1.0 — so an approximate index only adds cost.
# Grounded in the roitelet crossover study (interactive 10 ms budget, D=768),
# and this is exactly the reference budget LATENCY_REFERENCE_MS names below —
# bench.calibrate.exact_max_n() derives EXACT_MAX_N against that same budget.
EXACT_MAX_N: int = 10_000

# The per-query latency budget EXACT_MAX_N was calibrated against (see
# bench.calibrate.INTERACTIVE_BUDGET_MS, which must match this). A caller's
# own Criteria.latency_budget_ms scales the exact/ANN crossover relative to
# this reference: brute force is one O(n*dim) matmul, so its cost is ~linear
# in n for fixed dim, and a budget k times looser affords ~k times the n.
LATENCY_REFERENCE_MS: float = 10.0

# "Very large volume" — the regime where FAISS's IVF+PQ quantisation and GPU
# batch throughput start to dominate a graph index.
FAISS_MIN_N: int = 1_000_000

# At/above this target we treat the workload as "high precision", steering away
# from aggressive quantisation toward exact/HNSW.
HIGH_RECALL: float = 0.95

# Bundled so the whole set can be overridden atomically by a caller / policy.yaml.
THRESHOLDS: dict[str, float] = {
    "EXACT_MAX_N": EXACT_MAX_N,
    "LATENCY_REFERENCE_MS": LATENCY_REFERENCE_MS,
    "FAISS_MIN_N": FAISS_MIN_N,
    "HIGH_RECALL": HIGH_RECALL,
}

# --- TEMPORARY: provisional routing while bench/ calibration is in flight -------
# EXACT_MAX_N is trustworthy (pure brute-force math), but every threshold above it
# (FAISS_MIN_N, HIGH_RECALL, ...) is still *guessed*, not measured —
# see bench/README.md and .private/next.md. Until the calibration sweep lands and
# ann_router/policy.yaml is bumped from bench/results/calibrated_policy.yaml,
# ann_router.router.route() overrides every non-exact pick to turbovec (a safe,
# always-correct default) rather than trusting a guessed crossover. This constant
# does NOT touch rank_backends() itself — the decision tree above stays the
# documented, tested target policy; only the live router applies the override.
# Flip this to False (and delete the override block in router.route()) once
# calibration is applied.
PROVISIONAL_ROUTING: bool = True


def raw_memory_gb(c: Criteria) -> float:
    """Estimate the RAM a *full-precision* (float32) flat index would need.

    Parameters
    ----------
    c : Criteria
        The problem description.

    Returns
    -------
    float
        Approximate gibibytes for ``n_vectors * dim`` float32 values.

    Examples
    --------
    >>> round(raw_memory_gb(Criteria(n_vectors=1_000_000, dim=768)), 2)
    2.86
    """
    # 4 bytes per float32 element; divide by 1024**3 for gibibytes. This is the
    # yardstick the "tight memory" test compares the user's budget against.
    return c.n_vectors * c.dim * 4 / (1024**3)


def _tight_memory(c: Criteria) -> bool:
    """Return ``True`` when a full-precision index would blow the memory budget.

    Parameters
    ----------
    c : Criteria
        The problem description.

    Returns
    -------
    bool
        ``True`` when a budget is declared and the raw float index exceeds it.
    """
    # No budget declared == not memory-constrained. A budget smaller than the
    # raw float index means we need a lean (mmap/quantised) backend.
    return c.memory_budget_gb is not None and raw_memory_gb(c) > c.memory_budget_gb


def _db_in_place(c: Criteria) -> bool:
    """Return ``True`` when the caller signalled an existing SQL database.

    Parameters
    ----------
    c : Criteria
        The problem description.

    Returns
    -------
    bool
        ``True`` when ``c.extra`` carries a ``pg_dsn`` or ``db_in_place`` flag.
    """
    # A DSN in `extra` (or an explicit flag) means "reuse the DB you already run"
    # — the tie-breaker that prefers pgvector over standing up Qdrant.
    return bool(c.extra.get("pg_dsn") or c.extra.get("db_in_place"))


def effective_exact_max_n(c: Criteria, t: dict) -> float:
    """Scale ``EXACT_MAX_N`` by how loose/tight the caller's latency budget is.

    ``EXACT_MAX_N`` is calibrated at a reference budget
    (:data:`LATENCY_REFERENCE_MS`); a brute-force scan is one ``O(n * dim)``
    matmul, so its cost is ~linear in ``n`` for fixed ``dim`` and a budget
    ``k`` times looser (or tighter) than the reference affords ``k`` times
    the corpus size. This is the one place ``Criteria.latency_budget_ms``
    is load-bearing in the decision tree — every rule below compares against
    this scaled value, never the raw threshold, so the tree stays gapless.

    Parameters
    ----------
    c : Criteria
        The problem description (reads ``latency_budget_ms``).
    t : dict
        The merged thresholds (reads ``EXACT_MAX_N``, ``LATENCY_REFERENCE_MS``).

    Returns
    -------
    float
        The latency-adjusted exact/ANN crossover.

    Examples
    --------
    >>> t = {"EXACT_MAX_N": 10_000, "LATENCY_REFERENCE_MS": 10.0}
    >>> effective_exact_max_n(Criteria(n_vectors=1, dim=8, latency_budget_ms=10.0), t)
    10000.0
    >>> effective_exact_max_n(Criteria(n_vectors=1, dim=8, latency_budget_ms=1.0), t)
    1000.0
    """
    return t["EXACT_MAX_N"] * (c.latency_budget_ms / t["LATENCY_REFERENCE_MS"])


@dataclass
class Rule:
    """One branch of the decision tree: a backend, a guard, and its rationale.

    Parameters
    ----------
    backend : str
        The backend this rule selects when eligible.
    eligible : Callable[[Criteria, dict], bool]
        Predicate deciding whether the rule fires for given criteria/thresholds.
    reason : Callable[[Criteria, dict], str]
        Produces the human-readable justification when the rule fires.
    """

    backend: str
    eligible: Callable[[Criteria, dict], bool]
    reason: Callable[[Criteria, dict], str]


# The ordered decision tree. Priority is top-to-bottom; the first *eligible* rule
# whose backend is also installed wins (availability is applied in router.py).
# Predicates are written to be individually true/false so each is unit-testable.
_RULES: list[Rule] = [
    Rule(
        "exact",
        lambda c, t: c.n_vectors < effective_exact_max_n(c, t),
        lambda c, t: (
            f"n={c.n_vectors:,} < {int(effective_exact_max_n(c, t)):,} "
            f"(EXACT_MAX_N={int(t['EXACT_MAX_N']):,} scaled to a {c.latency_budget_ms:g} ms "
            "budget): a brute-force scan is already instant and exact (recall 1.0), so "
            "approximation would only add build cost."
        ),
    ),
    Rule(
        "turbovec",
        lambda c, t: c.n_vectors >= effective_exact_max_n(c, t) and c.dynamic,
        lambda c, t: (
            "corpus receives frequent updates: turbovec offers O(1) add_with_ids/remove(id) "
            "with no index rebuild, plus TurboQuant 2-4 bit (~16x) compression — graph "
            "indexes degrade under deletes, so they are avoided here."
        ),
    ),
    Rule(
        "faiss",
        lambda c, t: (
            c.n_vectors >= effective_exact_max_n(c, t)
            and not c.dynamic
            and c.n_vectors >= t["FAISS_MIN_N"]
            and (c.hardware == "gpu" or c.batch_queries)
        ),
        lambda c, t: (
            f"very large static corpus (n={c.n_vectors:,} >= {int(t['FAISS_MIN_N']):,}) with "
            f"{'a GPU' if c.hardware == 'gpu' else 'batch queries'}: FAISS IVF+PQ scales with "
            "quantisation and (on GPU) batch throughput."
        ),
    ),
    Rule(
        "pgvector",
        lambda c, t: (
            c.n_vectors >= effective_exact_max_n(c, t)
            and not c.dynamic
            and (c.persistence or c.metadata_filtering)
            and _db_in_place(c)
        ),
        lambda c, t: (
            "persistence and/or metadata filtering with a PostgreSQL already in place: "
            "pgvector keeps vectors next to relational data and filters with plain SQL WHERE, "
            "so no second datastore is needed."
        ),
    ),
    Rule(
        "qdrant",
        lambda c, t: (
            c.n_vectors >= effective_exact_max_n(c, t)
            and not c.dynamic
            and (c.persistence or c.metadata_filtering)
        ),
        lambda c, t: (
            "needs persistence and/or structured metadata filtering: Qdrant pairs an on-disk "
            "HNSW index with per-point JSON payloads and filters at query time — the in-memory "
            "engines cannot do payload filtering."
        ),
    ),
    Rule(
        "annoy",
        lambda c, t: (
            c.n_vectors >= effective_exact_max_n(c, t) and not c.dynamic and _tight_memory(c)
        ),
        lambda c, t: (
            f"read-only corpus under a tight memory budget "
            f"(~{raw_memory_gb(c):.1f} GB raw > {c.memory_budget_gb} GB budget): Annoy freezes "
            "the index and memory-maps it from disk, so many processes share it at near-zero RAM."
        ),
    ),
    Rule(
        "hnsw",
        # The catch-all in-memory default for any non-trivial static corpus, and
        # the fallback for a dynamic corpus when turbovec is unavailable.
        lambda c, t: c.n_vectors >= effective_exact_max_n(c, t),
        lambda c, t: (
            "stable in-memory corpus, high precision wanted: hnswlib gives the best "
            "recall/latency of the in-memory engines when the index rarely changes "
            "(note: it deletes only via tombstones, hence 'stable')."
            if not c.dynamic
            else "dynamic corpus but turbovec unavailable: HNSW is the working fallback, "
            "though its tombstone deletes degrade the graph over time."
        ),
    ),
]


def rank_backends(c: Criteria, thresholds: dict[str, float] | None = None) -> list[dict]:
    """Return the ordered, justified backend shortlist for the criteria.

    This is the pure heart of the router: it applies every rule in priority
    order and returns one row per *eligible* rule, each carrying the backend
    name and its rationale. Availability and the final pick are decided in
    :func:`ann_router.router.route`, keeping this function side-effect-free.

    Parameters
    ----------
    c : Criteria
        The measured problem description.
    thresholds : dict, optional
        Overrides for :data:`THRESHOLDS` (tunable policy). Missing keys fall
        back to the module defaults.

    Returns
    -------
    list of dict
        ``[{"backend": str, "reason": str}, ...]`` in priority order — the first
        element is the policy's preferred choice before availability is applied.

    Examples
    --------
    >>> rank_backends(Criteria(n_vectors=5_000, dim=128))[0]["backend"]
    'exact'
    >>> rank_backends(Criteria(n_vectors=500_000, dim=768, dynamic=True))[0]["backend"]
    'turbovec'
    >>> rank_backends(Criteria(n_vectors=200_000, dim=768,
    ...                        metadata_filtering=True))[0]["backend"]
    'qdrant'
    """
    # Merge caller overrides onto the defaults so a partial dict still works.
    t = {**THRESHOLDS, **(thresholds or {})}
    shortlist: list[dict] = []
    for rule in _RULES:
        if rule.eligible(c, t):
            shortlist.append({"backend": rule.backend, "reason": rule.reason(c, t)})
    return shortlist
