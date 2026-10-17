# ann-router CLI reference

Two CLIs over the same core. `ann-router` (argparse) ships with the core
package; `ann-router-click` (needs `[cli]`) is the click twin. Both dispatch
to the shared `_core_cli`, so flags match exactly.

## `route` — which backend, and why

```bash
ann-router route --n-vectors N --dim D \
                 [--dynamic] [--metadata-filtering] [--persistence] \
                 [--target-recall 0.95] [--memory-budget-gb 2] \
                 [--markdown] [--json]
```

Prints the chosen backend, the rationale (which criteria drove it), and the
alternatives considered — including any preferred-but-uninstalled engine it
fell back from. `--markdown` renders a human-readable report.

**The rationale is the deliverable.** Report *why* a backend won, not just its
name: that is what makes the choice discussable instead of arbitrary.

## `build` — route, then construct the index

```bash
ann-router build --vectors vectors.npy --out index/ [route flags…]
```

Writes the index plus a `*.meta.json` sidecar recording the criteria and the
backend, so a later reader knows how the index was chosen.

## `search` — query a built index

```bash
ann-router search --index index/ --queries q.npy -k 10
```

## `bench` — measure recall against exact truth

```bash
ann-router bench --n 5000 --dim 128 -k 10 [--backends hnsw,faiss,annoy]
```

Compares each available backend against brute-force ground truth. This is how
the thresholds in `policy.yaml` get calibrated — from measurements, not
guesses. See `bench/README.md` in the repo.

## `capabilities` — what is installed, and what each engine can do

```bash
ann-router capabilities [--json]
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
pip install ann-router                 # core: library + argparse CLI + exact backend
pip install 'ann-router[hnsw]'         # HNSW
pip install 'ann-router[faiss]'        # FAISS
pip install 'ann-router[annoy]'        # Annoy
pip install 'ann-router[turbovec]'     # turbovec
pip install 'ann-router[qdrant]'       # Qdrant client
pip install 'ann-router[pgvector]'     # pgvector client
pip install 'ann-router[cli,api,mcp]'  # click CLI, FastAPI, MCP tools
pip install 'ann-router[all]'          # every pip-installable engine
```

Platform caveats (Apple-Silicon annoy, a pgvector server) are in the repo's
`INSTALL.md`.

## Output contract

- `route` prints the backend, the rationale and the alternatives; `--json`
  for parsing, `--markdown` for a person.
- `build` prints the index path and the sidecar path.
- `bench` prints a recall table.
- Failures are logged through `os_helper` and exit non-zero.
