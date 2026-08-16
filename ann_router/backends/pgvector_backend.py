"""pgvector backend: vector search inside an existing PostgreSQL.

pgvector's niche is *persistence + metadata filters when a Postgres is already
in place*: you keep vectors next to your relational data and filter with plain
SQL ``WHERE`` clauses, backed by an HNSW index on the vector column. The router
picks it (over Qdrant) when the criteria say "a database is already there",
because reusing it beats standing up a second datastore. Unlike every other
backend this one needs a live server, so it is unavailable unless a DSN is
supplied: absence is a clean skip, never a crash.

Consumes: ``pgvector`` + ``psycopg`` (optional, ``pip install 'ann-router[pgvector]'``)
and a reachable PostgreSQL with the ``vector`` extension.
Produces: :class:`PgVectorIndex`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import os

import numpy as np

from ..base import ANNIndex, BackendUnavailable, Capabilities, NotSupported
from ..spec import MetricName

# pgvector operators: <=> cosine distance, <-> L2 distance, <#> negative IP.
_OP = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}


def _require():
    """Import psycopg + pgvector or raise an actionable :class:`BackendUnavailable`."""
    try:
        import psycopg  # local import: never needed just to import ann_router
        from pgvector.psycopg import register_vector  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when absent
        raise BackendUnavailable(
            "pgvector/psycopg not installed. Run: pip install 'ann-router[pgvector]'"
        ) from exc
    return psycopg


class PgVectorIndex(ANNIndex):
    """A single vectors table in PostgreSQL, indexed with pgvector HNSW.

    Requires a DSN (``dsn=`` kwarg or the ``ANN_ROUTER_PG_DSN`` env var). Points
    live in a ``(id bigint, embedding vector(dim), payload jsonb)`` table so the
    SQL ``WHERE`` path can filter on ``payload``.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : {"cosine", "l2", "ip"}, optional
        Distance metric. Defaults to ``"cosine"``.
    dsn : str, optional
        PostgreSQL connection string. Falls back to ``ANN_ROUTER_PG_DSN``.
    table : str, optional
        Table name. Defaults to ``"ann_router"``.

    Examples
    --------
    >>> PgVectorIndex.capabilities().persistent
    True
    """

    def __init__(self, dim: int, metric: MetricName = "cosine", **kwargs: object) -> None:
        super().__init__(dim, metric, **kwargs)
        self._dsn = kwargs.get("dsn") or os.environ.get("ANN_ROUTER_PG_DSN")
        self._table = str(kwargs.get("table", "ann_router"))

    @classmethod
    def capabilities(cls) -> Capabilities:
        """Return the pgvector capability descriptor (persistent + filterable)."""
        return Capabilities(
            name="pgvector",
            supports_add=True,
            supports_remove=True,
            supports_filter=True,
            persistent=True,
            needs_gpu=False,
            approximate=True,
            metrics=("cosine", "l2", "ip"),
            pip_extra="pgvector",
        )

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` if psycopg + pgvector import (a live DSN is still needed).

        Examples
        --------
        >>> isinstance(PgVectorIndex.is_available(), bool)
        True
        """
        try:
            import psycopg  # noqa: F401
            from pgvector.psycopg import register_vector  # noqa: F401
        except ImportError:
            return False
        return True

    def _connect(self):
        """Open a connection, registering the pgvector type adapters."""
        psycopg = _require()
        from pgvector.psycopg import register_vector

        if not self._dsn:
            raise NotSupported(
                "pgvector: no DSN — pass dsn=... or set ANN_ROUTER_PG_DSN to a live Postgres"
            )
        conn = psycopg.connect(self._dsn, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)  # teaches psycopg how to send/receive numpy vectors
        return conn

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray | None = None,
        payloads: list[dict] | None = None,
    ) -> PgVectorIndex:
        """(Re)create the table, insert the corpus, and build the HNSW index.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(n, dim)``.
        ids : numpy.ndarray, optional
            Shape ``(n,)``; defaults to ``range(n)``.
        payloads : list of dict, optional
            One JSON-serialisable payload per row, aligned with ``vectors``.

        Returns
        -------
        PgVectorIndex
            ``self``.
        """
        import json

        arr = self._as_f32(vectors)
        conn = self._connect()
        self._index = conn
        # Fresh table each build so schema/dim stay consistent with this index.
        conn.execute(f"DROP TABLE IF EXISTS {self._table}")
        conn.execute(
            f"CREATE TABLE {self._table} "
            f"(id bigint PRIMARY KEY, embedding vector({self.dim}), payload jsonb)"
        )
        labels = np.arange(arr.shape[0]) if ids is None else np.asarray(ids)
        with conn.cursor() as cur:
            for row, pid in enumerate(labels.tolist()):
                payload = json.dumps(payloads[row]) if payloads else "{}"
                cur.execute(
                    f"INSERT INTO {self._table} (id, embedding, payload) VALUES (%s, %s, %s)",
                    (int(pid), arr[row], payload),
                )
        # HNSW on the right operator class so the ORDER BY uses the index.
        opclass = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}
        conn.execute(f"CREATE INDEX ON {self._table} USING hnsw (embedding {opclass[self.metric]})")
        return self

    def add(self, vectors: np.ndarray) -> None:
        """Append vectors with ids continuing past the current max.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        """
        cur = self._index.execute(f"SELECT COALESCE(MAX(id), -1) FROM {self._table}")  # type: ignore[union-attr]
        start = cur.fetchone()[0] + 1
        self.add_with_ids(vectors, np.arange(start, start + len(vectors)))

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Append vectors with explicit ids.

        Parameters
        ----------
        vectors : numpy.ndarray
            Shape ``(m, dim)``.
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids.
        """
        arr = self._as_f32(vectors)
        with self._index.cursor() as cur:  # type: ignore[union-attr]
            for row, pid in enumerate(np.asarray(ids).tolist()):
                cur.execute(
                    f"INSERT INTO {self._table} (id, embedding, payload) VALUES (%s, %s, '{{}}')",
                    (int(pid), arr[row]),
                )

    def remove(self, ids: np.ndarray) -> None:
        """Delete rows by id.

        Parameters
        ----------
        ids : numpy.ndarray
            Shape ``(m,)`` integer ids to drop.
        """
        id_list = [int(i) for i in np.asarray(ids)]
        self._index.execute(  # type: ignore[union-attr]
            f"DELETE FROM {self._table} WHERE id = ANY(%s)", (id_list,)
        )

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate top-``k`` neighbours per query (no filter).

        Parameters
        ----------
        queries : numpy.ndarray
            Shape ``(q, dim)``.
        k : int
            Neighbours per query.

        Returns
        -------
        ids : numpy.ndarray
            Shape ``(q, k)`` neighbour ids.
        distances : numpy.ndarray
            Shape ``(q, k)`` distances under the metric operator.
        """
        return self.search_filter(queries, k, where=None)

    def search_filter(
        self, queries: np.ndarray, k: int, where: dict | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-``k`` neighbours, optionally filtered by a ``payload`` clause.

        Parameters
        ----------
        queries : numpy.ndarray
            Shape ``(q, dim)``.
        k : int
            Neighbours per query.
        where : dict, optional
            ``{field: value}`` equality constraints matched against the JSONB
            ``payload`` column. ``None`` searches the whole table.

        Returns
        -------
        ids : numpy.ndarray
            Shape ``(q, k)`` (``-1`` pads short rows).
        distances : numpy.ndarray
            Shape ``(q, k)`` distances under the metric operator.
        """
        arr = self._as_f32(queries)
        op = _OP[self.metric]
        # Build the optional JSONB equality predicate; values are parameterised.
        clause, params_tail = "", []
        if where:
            preds = " AND ".join("payload->>%s = %s" for _ in where)
            clause = f"WHERE {preds} "
            for key, val in where.items():
                params_tail += [key, str(val)]
        ids_out, dist_out = [], []
        with self._index.cursor() as cur:  # type: ignore[union-attr]
            for q in arr:
                cur.execute(
                    f"SELECT id, embedding {op} %s AS dist FROM {self._table} "
                    f"{clause}ORDER BY dist LIMIT %s",
                    (q, *params_tail, k),
                )
                rows = cur.fetchall()
                row_ids = [int(r[0]) for r in rows] + [-1] * (k - len(rows))
                row_d = [float(r[1]) for r in rows] + [float("inf")] * (k - len(rows))
                ids_out.append(row_ids)
                dist_out.append(row_d)
        return np.array(ids_out, dtype=np.int64), np.array(dist_out, dtype=np.float32)

    def save(self, path: str) -> None:
        """No-op — the table lives in PostgreSQL and is already durable.

        Parameters
        ----------
        path : str
            Unused — accepted only to satisfy the shared interface.

        Notes
        -----
        Persistence is the database's job; there is nothing to serialise. Use
        the same DSN + table to :meth:`load` the index in another process.
        """
        return None

    def load(self, path: str) -> PgVectorIndex:
        """Reconnect to the existing table (``path`` may override the DSN).

        Parameters
        ----------
        path : str
            A DSN to reconnect with, or falsy to reuse the constructor's DSN.

        Returns
        -------
        PgVectorIndex
            ``self``, reconnected.
        """
        if path:
            self._dsn = path
        self._index = self._connect()
        return self
