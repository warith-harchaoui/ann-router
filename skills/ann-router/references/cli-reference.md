# ann-router CLI reference

One CLI, `ann-router`, built with click and shipped with the core package. It
dispatches to the shared `_core_cli`, which is also what the HTTP API and the
MCP server call, so every door answers the same way.

## `route` — which backend, and why

```bash
ann-router route --n-vectors N --dim D \
                 [--dynamic] [--metadata-filtering] [--persistence] \
                 [--batch-queries] [--target-recall 0.95] \
                 [--latency-budget-ms 10] [--memory-budget-gb 2] \
                 [--hardware cpu|gpu|apple_silicon] [--metric cosine|l2|ip] \
                 [--markdown]
```

Prints JSON by default: the chosen backend, the rationale (which criteria
drove it), and the alternatives considered — including any
preferred-but-uninstalled engine it fell back from. `--markdown` renders a
human-readable report instead. Without `--hardware`, the machine is probed.
The group option `-v` / `-vv` (before the subcommand) raises the log level;
logs go to stderr.

**The rationale is the deliverable.** Report *why* a backend won, not just its
name: that is what makes the choice discussable instead of arbitrary.

## `build` — route, then construct the index

```bash
ann-router build --vectors vectors.npy --index my.index \
                 --n-vectors N --dim D [route flags…] [--backend hnsw]
```

Takes the same criteria options as `route` (`--n-vectors` and `--dim` are
required); `--backend` forces an engine and skips routing. Writes the index
plus a `<index>.meta.json` sidecar recording the backend, the dimension and
the metric, which `search` reads to reopen it with the right adapter.

## `search` — query a built index

```bash
ann-router search --index index/ --queries q.npy -k 10
```

## `bench` — measure recall against exact truth

```bash
ann-router bench --n 5000 --dim 128 -k 10
```

Compares every installed backend against brute-force ground truth. This is how
the thresholds in `policy.yaml` get calibrated — from measurements, not
guesses. See `bench/README.md` in the repo.

## `capabilities` — what is installed, and what each engine can do

```bash
ann-router capabilities
```

The matrix of backends against the properties that matter: dynamic updates,
metadata filtering, persistence, memory profile, GPU. Run it before `route`
when a recommendation looks surprising — the usual explanation is that the
preferred engine is not installed.

## The decision, in one glance

| criteria | → backend |
| --- | --- |
| `n < EXACT_MAX_N` | exact (brute force) |
| frequent updates | turbovec |
| very large + GPU/batch | FAISS (IVF/PQ) |
| persistence + metadata filters | Qdrant / pgvector |
| read-only + tight memory | Annoy |
| stable in-memory (default) | HNSW |

Thresholds (`EXACT_MAX_N`, `FAISS_MIN_N`, …) live in `ann_router/policy.yaml`
and are being calibrated from measured data in `bench/`. Override the whole
policy with the `ANN_ROUTER_POLICY` environment variable.

## Installing the engines

```bash
pip install ann-router                 # core: library + click CLI + exact backend
pip install 'ann-router[hnsw]'         # HNSW
pip install 'ann-router[faiss]'        # FAISS
pip install 'ann-router[annoy]'        # Annoy
pip install 'ann-router[turbovec]'     # turbovec
pip install 'ann-router[qdrant]'       # Qdrant client
pip install 'ann-router[pgvector]'     # pgvector client
pip install 'ann-router[api,mcp]'      # FastAPI, MCP tools (click is core)
pip install 'ann-router[all]'          # every pip-installable engine
```

Platform caveats (Apple-Silicon annoy, a pgvector server) are in the repo's
`INSTALL.md`.

## Output contract

- Every command prints JSON on stdout; `route --markdown` prints a report for
  a person instead.
- `build` prints the backend, the rationale, `index_path`, `n`, `dim` and
  `build_seconds`.
- `bench` prints recall and timing per installed backend.
- A failure prints `Error: …` on stderr and exits non-zero; logs go to stderr.
