"""The router — measure criteria, select an available backend, justify it.

This is the ``recommend.py`` analogue: it takes a :class:`~ann_router.spec.Criteria`,
consults the pure policy (``policy.rank_backends``), applies *availability* (a
policy pick whose dependency is missing is skipped, with the fallback
explained), attaches recommended build parameters, and returns a fully
discussable :class:`~ann_router.spec.BackendChoice`. A convenience
:func:`auto_index` closes the loop: route -> instantiate -> build.

The contract mirrors the sibling router's promise — the answer is never just a
name, it is a name *plus the criteria that drove it plus the alternatives that
were considered*, so a human can audit or override it.

Consumes: ``ann_router.policy``, ``ann_router.registry``, ``ann_router.spec``,
``os_helper`` (logging).
Produces: :func:`route`, :func:`auto_index`, :func:`to_markdown`.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import numpy as np
import os_helper as osh

from .base import ANNIndex
from .policy import POLICY_VERSION, PROVISIONAL_ROUTING, rank_backends
from .registry import BACKENDS, get_backend
from .spec import BackendChoice, Criteria


def _recommended_config(backend: str, c: Criteria) -> dict:
    """Return sensible, criteria-scaled build parameters for ``backend``.

    Parameters
    ----------
    backend : str
        The chosen backend name.
    c : Criteria
        The problem description (used to scale index knobs to corpus size).

    Returns
    -------
    dict
        Backend-specific kwargs to pass to the adapter constructor.

    Examples
    --------
    >>> cfg = _recommended_config("hnsw", Criteria(n_vectors=100_000, dim=768))
    >>> cfg["M"], cfg["metric"]
    (16, 'cosine')
    """
    # Every backend at least needs the metric; the rest are tuned per engine to
    # the recall target and corpus scale so `auto_index` builds something good.
    cfg: dict = {"metric": c.metric}
    if backend == "hnsw":
        # Higher M/ef buy recall at the cost of memory/latency; nudge up when the
        # target is demanding.
        cfg.update(M=16 if c.target_recall < 0.98 else 32, ef_construction=200, ef=64)
    elif backend == "faiss":
        # PQ only pays off at very large N; below that keep a flat IVF for recall.
        cfg.update(nprobe=16, use_pq="auto")
    elif backend == "annoy":
        # More trees == better recall, larger file; 50 is Annoy's usual default.
        cfg.update(n_trees=50 if c.target_recall < 0.98 else 100)
    elif backend == "turbovec":
        # 4-bit is the recall/size sweet spot measured in the roitelet study.
        cfg.update(bit_width=4)
    elif backend == "scann":
        cfg.update(num_leaves_to_search=100, reordering=100)
    elif backend in ("qdrant", "pgvector") and c.extra.get("pg_dsn"):
        # Carry a DSN through if the caller supplied one for the DB path.
        cfg["dsn"] = c.extra["pg_dsn"]
    return cfg


def route(c: Criteria, thresholds: dict[str, float] | None = None) -> BackendChoice:
    """Select an available backend for the criteria and justify the choice.

    Parameters
    ----------
    c : Criteria
        The measured problem description.
    thresholds : dict, optional
        Overrides for the policy thresholds (tunable). See
        :data:`ann_router.policy.THRESHOLDS`.

    Returns
    -------
    BackendChoice
        The chosen backend, its rationale, recommended config, and the full
        considered shortlist (each entry flagged eligible/available/chosen).

    Examples
    --------
    >>> route(Criteria(n_vectors=1_000, dim=64)).backend
    'exact'
    >>> choice = route(Criteria(n_vectors=500_000, dim=768, metadata_filtering=True))
    >>> choice.backend in {"qdrant", "pgvector", "hnsw", "exact", "turbovec"}
    True
    """
    c.validate()
    # When the caller gives no explicit thresholds, load the shipped policy.yaml
    # (merged over the code defaults, plus any ANN_ROUTER_POLICY override) so the
    # tunable YAML is genuinely load-bearing rather than decorative.
    if thresholds is None:
        from .config import policy_thresholds

        thresholds = policy_thresholds()
    shortlist = rank_backends(c, thresholds)

    # Walk the priority-ordered shortlist and take the first backend that is
    # actually installed on this machine; record every entry for the audit trail.
    considered: list[dict] = []
    chosen: dict | None = None
    for entry in shortlist:
        cls = get_backend(entry["backend"])
        available = cls.is_available()
        row = {
            "backend": entry["backend"],
            "reason": entry["reason"],
            "available": available,
            "chosen": False,
        }
        if available and chosen is None:
            row["chosen"] = True
            chosen = row
        considered.append(row)

    # The policy always ends with hnsw/exact, and exact is always available, so
    # `chosen` is guaranteed non-None — but guard explicitly and fall to exact.
    if chosen is None:  # pragma: no cover - exact is always available
        chosen = {
            "backend": "exact",
            "reason": "no preferred backend installed; falling back to the always-available "
            "exact brute-force reference.",
            "available": True,
            "chosen": True,
        }
        considered.append(chosen)

    backend = chosen["backend"]
    rationale = chosen["reason"]
    # If the policy's #1 pick was skipped for being uninstalled, say so up front —
    # that is the "discussable fallback" the house style requires.
    top = shortlist[0]["backend"] if shortlist else backend
    if backend != top:
        rationale = (
            f"Preferred backend '{top}' is not installed here, so the router fell back. "
            + rationale
        )
        osh.info(f"ann-router: preferred '{top}' unavailable, using '{backend}'")

    # --- TEMPORARY provisional override — see policy.PROVISIONAL_ROUTING docstring.
    # exact stays exact (n < EXACT_MAX_N is measured math, not a guess); every other
    # pick is redirected to turbovec until the calibration sweep lands.
    if PROVISIONAL_ROUTING and backend not in ("exact", "turbovec") and get_backend(
        "turbovec"
    ).is_available():
        rationale = (
            "PROVISIONAL routing (bench/ calibration not yet applied — see bench/README.md): "
            f"every threshold above EXACT_MAX_N is still unmeasured, so turbovec is used "
            f"instead of the policy's current pick. {rationale}"
        )
        for row in considered:
            row["chosen"] = False
        turbovec_row = next((row for row in considered if row["backend"] == "turbovec"), None)
        if turbovec_row is None:
            turbovec_row = {"backend": "turbovec", "reason": rationale, "available": True}
            considered.append(turbovec_row)
        turbovec_row["reason"] = rationale
        turbovec_row["chosen"] = True
        turbovec_row["available"] = True
        backend = "turbovec"

    config = _recommended_config(backend, c)
    return BackendChoice(
        backend=backend,
        rationale=rationale,
        config=config,
        considered=considered,
        criteria={**c.to_dict(), "policy_version": POLICY_VERSION},
    )


def auto_index(
    vectors: np.ndarray,
    criteria: Criteria,
    ids: np.ndarray | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[ANNIndex, BackendChoice]:
    """Route the criteria, instantiate the winning backend, and build the index.

    Parameters
    ----------
    vectors : numpy.ndarray
        Corpus of shape ``(n, dim)``.
    criteria : Criteria
        The problem description. If its ``n_vectors``/``dim`` disagree with
        ``vectors`` the array wins (the criteria are advisory for routing).
    ids : numpy.ndarray, optional
        Explicit ids of shape ``(n,)``; defaults to ``range(n)``.
    thresholds : dict, optional
        Policy threshold overrides.

    Returns
    -------
    index : ANNIndex
        A built, queryable index of the chosen backend.
    choice : BackendChoice
        The routing decision (so the caller can inspect/log the rationale).

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> vecs = rng.standard_normal((2_000, 32)).astype(np.float32)
    >>> idx, choice = auto_index(vecs, Criteria(n_vectors=2_000, dim=32))
    >>> choice.backend
    'exact'
    >>> ids, dists = idx.search(vecs[:1], k=5)
    >>> ids.shape
    (1, 5)
    """
    choice = route(criteria, thresholds)
    cls = BACKENDS[choice.backend]
    # Instantiate with the routed config, then build; dim comes from the array so
    # a mismatched Criteria.dim never corrupts the index.
    index = cls(dim=vectors.shape[1], **choice.config)
    index.build(vectors, ids=ids)
    return index, choice


def to_markdown(choice: BackendChoice) -> str:
    """Render a routing decision as a human-readable Markdown report.

    Parameters
    ----------
    choice : BackendChoice
        A decision produced by :func:`route`.

    Returns
    -------
    str
        Markdown with the pick, the rationale, and the considered table.

    Examples
    --------
    >>> md = to_markdown(route(Criteria(n_vectors=1_000, dim=64)))
    >>> md.splitlines()[0]
    '# ann-router decision: `exact`'
    """
    lines = [
        f"# ann-router decision: `{choice.backend}`",
        "",
        f"**Rationale.** {choice.rationale}",
        "",
        f"**Recommended config.** `{choice.config}`",
        "",
        "## Considered (priority order)",
        "",
        "| backend | available | chosen | reason |",
        "| --- | --- | --- | --- |",
    ]
    for row in choice.considered:
        # A compact one-line-per-candidate table so the fallback logic is legible.
        mark = "yes" if row["available"] else "no"
        star = "**<-**" if row["chosen"] else ""
        lines.append(f"| `{row['backend']}` | {mark} | {star} | {row['reason']} |")
    lines += ["", f"_policy version {POLICY_VERSION}_"]
    return "\n".join(lines)
