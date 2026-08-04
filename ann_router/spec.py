"""Typed problem criteria that drive ANN-backend selection.

This module holds the *input* side of the router: a single, explicit
``Criteria`` dataclass describing the vector-search problem to solve
(corpus size, dimensionality, recall target, latency/memory budgets,
update pattern, metadata-filtering need, hardware, persistence), plus the
``BackendChoice`` result the router returns.

The design goal — mirrored from the ``best-engine-ai-helper`` sibling — is
that the choice is *measured and discussable*: every field here is a lever
that can flip the decision, and the router explains which levers mattered.

Consumes: nothing (pure data).
Produces: ``Criteria`` and ``BackendChoice`` values consumed by
``ann_router.policy`` and ``ann_router.router``.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# The eight backend identifiers the router can select from. Keeping them as a
# closed literal (rather than free strings) makes typos a type error and lets
# the policy table stay exhaustive.
BackendName = Literal[
    "exact",
    "turbovec",
    "hnsw",
    "faiss",
    "annoy",
    "scann",
    "qdrant",
    "pgvector",
]

# Hardware profiles that change the decision. ``apple_silicon`` is called out
# separately from ``cpu`` because turbovec (Rust + NEON) and Metal-backed
# engines behave differently there; ``gpu`` unlocks FAISS-GPU / ScaNN batch.
HardwareName = Literal["cpu", "gpu", "apple_silicon"]

# The distance metric the corpus is embedded for. Cosine is the default for
# text/vision embeddings; L2 (euclidean) and inner product cover the rest.
MetricName = Literal["cosine", "l2", "ip"]


@dataclass
class Criteria:
    """Measured description of an approximate-nearest-neighbour problem.

    Every attribute is a decision lever. The two mandatory ones — ``n_vectors``
    and ``dim`` — set the scale; the rest refine the choice and each carries a
    house default that matches the most common single-machine RAG workload.

    Parameters
    ----------
    n_vectors : int
        Number of vectors in (or expected in) the corpus. Below
        ``policy.EXACT_MAX_N`` an exact brute-force scan is already instant and
        perfectly accurate, so approximation is pointless.
    dim : int
        Embedding dimensionality (e.g. 384, 768, 1536, 2048).
    target_recall : float, optional
        Desired recall@k against the exact ground truth, in ``(0, 1]``.
        Defaults to ``0.95``. High values push toward exact/HNSW/ScaNN and away
        from aggressively quantised indexes.
    latency_budget_ms : float, optional
        Per-query latency budget in milliseconds. Defaults to ``10.0`` (an
        interactive budget). Drives the exact→ANN crossover.
    memory_budget_gb : float or None, optional
        Soft cap on index RAM in gibibytes. ``None`` (default) means "not a
        constraint". A tight budget favours quantised (turbovec/FAISS-PQ) or
        memory-mapped (Annoy) backends.
    dynamic : bool, optional
        ``True`` when the corpus receives frequent adds/removes and must not be
        rebuilt. Defaults to ``False`` (static corpus). Frequent updates favour
        turbovec (O(1) add/remove); graph indexes like HNSW handle deletes
        poorly (tombstones only).
    metadata_filtering : bool, optional
        ``True`` when queries must be filtered by structured metadata
        (payload/where clauses). Defaults to ``False``. Favours Qdrant/pgvector.
    hardware : {"cpu", "gpu", "apple_silicon"}, optional
        The accelerator available. Defaults to ``"cpu"``. Use
        :func:`ann_router.detect.detect_hardware` to fill this in automatically.
    persistence : bool, optional
        ``True`` when the index must survive process restarts / live in a
        database. Defaults to ``False``. Favours Qdrant/pgvector, or a
        ``save``/``load`` round-trip for the in-memory engines.
    batch_queries : bool, optional
        ``True`` when queries arrive in large batches (throughput regime rather
        than interactive). Defaults to ``False``. Combined with ``gpu`` this
        favours FAISS.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric the vectors are embedded for. Defaults to ``"cosine"``.
    extra : dict, optional
        Free-form escape hatch for backend-specific hints (e.g. an existing DB
        DSN). Never required by the core policy.

    Examples
    --------
    >>> c = Criteria(n_vectors=5_000, dim=768)
    >>> c.dim
    768
    >>> c.target_recall            # house default
    0.95
    >>> Criteria(n_vectors=2_000_000, dim=1536, dynamic=True).dynamic
    True

    Notes
    -----
    The dataclass is intentionally plain (no validation on construction beyond
    :meth:`validate`, which the router calls) so it round-trips cleanly through
    JSON for the CLI / API / MCP surfaces via :meth:`to_dict` / :meth:`from_dict`.
    """

    n_vectors: int
    dim: int
    target_recall: float = 0.95
    latency_budget_ms: float = 10.0
    memory_budget_gb: float | None = None
    dynamic: bool = False
    metadata_filtering: bool = False
    hardware: HardwareName = "cpu"
    persistence: bool = False
    batch_queries: bool = False
    metric: MetricName = "cosine"
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Assert the criteria are internally consistent.

        Raises
        ------
        ValueError
            If a numeric field is out of its valid range.

        Examples
        --------
        >>> Criteria(n_vectors=100, dim=8).validate() is None
        True
        >>> try:
        ...     Criteria(n_vectors=-1, dim=8).validate()
        ... except ValueError as exc:
        ...     print("rejected")
        rejected
        """
        # Scale must be physical: you cannot index a negative number of vectors
        # nor a zero-dimensional embedding.
        if self.n_vectors < 0:
            raise ValueError(f"n_vectors must be >= 0, got {self.n_vectors}")
        if self.dim <= 0:
            raise ValueError(f"dim must be >= 1, got {self.dim}")
        # Recall is a fraction; a budget of <= 0 ms/GB is meaningless.
        if not 0.0 < self.target_recall <= 1.0:
            raise ValueError(f"target_recall must be in (0, 1], got {self.target_recall}")
        if self.latency_budget_ms <= 0:
            raise ValueError(f"latency_budget_ms must be > 0, got {self.latency_budget_ms}")
        if self.memory_budget_gb is not None and self.memory_budget_gb <= 0:
            raise ValueError(f"memory_budget_gb must be > 0 or None, got {self.memory_budget_gb}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the criteria.

        Returns
        -------
        dict
            One key per attribute; safe to ``json.dumps``.

        Examples
        --------
        >>> Criteria(n_vectors=100, dim=8).to_dict()["n_vectors"]
        100
        """
        # A dataclass is not JSON-ready because of the nested ``extra`` dict and
        # the Literal fields; a plain shallow copy of the public attributes is.
        return {
            "n_vectors": self.n_vectors,
            "dim": self.dim,
            "target_recall": self.target_recall,
            "latency_budget_ms": self.latency_budget_ms,
            "memory_budget_gb": self.memory_budget_gb,
            "dynamic": self.dynamic,
            "metadata_filtering": self.metadata_filtering,
            "hardware": self.hardware,
            "persistence": self.persistence,
            "batch_queries": self.batch_queries,
            "metric": self.metric,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Criteria:
        """Rebuild a ``Criteria`` from a (possibly partial) mapping.

        Parameters
        ----------
        data : dict
            Keys matching the dataclass fields. ``n_vectors`` and ``dim`` are
            required; unknown keys are ignored so the CLI/API can pass through
            loosely.

        Returns
        -------
        Criteria
            The reconstructed criteria.

        Examples
        --------
        >>> Criteria.from_dict({"n_vectors": 100, "dim": 8, "dynamic": True}).dynamic
        True
        """
        # Only forward keys the dataclass actually declares; this keeps the
        # public JSON surface forgiving (extra keys from an older client are
        # dropped rather than raising).
        known = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class BackendChoice:
    """The router's decision: which backend, why, and how to configure it.

    Parameters
    ----------
    backend : str
        The selected backend name (one of :data:`BackendName`).
    rationale : str
        Human-readable justification naming the criteria that drove the pick —
        the "discussable" part of the router's contract.
    config : dict
        Backend-specific build parameters the router recommends (e.g. HNSW
        ``M``/``ef_construction``, FAISS ``nlist``/``m``).
    considered : list of dict, optional
        The full ranked shortlist (each entry: name, eligible, reason,
        available) so callers can audit or override the decision.
    criteria : dict, optional
        The echoed input criteria for provenance.

    Examples
    --------
    >>> choice = BackendChoice(backend="exact", rationale="tiny corpus", config={})
    >>> choice.backend
    'exact'
    """

    backend: BackendName
    rationale: str
    config: dict[str, Any] = field(default_factory=dict)
    considered: list[dict[str, Any]] = field(default_factory=list)
    criteria: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the choice.

        Returns
        -------
        dict
            Ready to ``json.dumps`` for the CLI/API/MCP surfaces.

        Examples
        --------
        >>> BackendChoice("exact", "tiny", {}).to_dict()["backend"]
        'exact'
        """
        return {
            "backend": self.backend,
            "rationale": self.rationale,
            "config": dict(self.config),
            "considered": list(self.considered),
            "criteria": dict(self.criteria),
        }
