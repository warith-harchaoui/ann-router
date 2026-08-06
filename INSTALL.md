# Installing ann-router

`ann-router` uses **conda (miniconda) + pip** as its base tooling (the suite
standard is conda + poetry; poetry is not required for this pure-setuptools
package). The core is deliberately tiny — `numpy` + `os-helper` + `pyyaml` — so
the library and the always-on argparse CLI install in seconds. Every ANN engine
is an optional extra you add only if you route to it.

There are three supported paths — pick one:

- **§1-6 below (conda + pip, step by step)** — for development or when you
  want to see/control every step.
- **`environment.yaml`** — the same conda + pip install, one command:
  `conda env create -f environment.yaml && conda activate ann-router`. It
  pins Python + pip and delegates every actual dependency to
  `requirements.txt`, so it never drifts out of sync with pip's own view of
  the project.
- **`Dockerfile`** — a server image with every pip-installable backend + the
  HTTP API door baked in: `docker build -t ann-router . && docker run --rm
  -p 8018:8018 ann-router`. Does not need conda, Python, or any of the steps
  below on the host — only Docker.

The exact commands below were run and verified on this machine
(**Apple M2 Max, arm64, macOS**) on 2026-08-04.

## 1. Create the environment

```bash
conda create -n ann-router python=3.11 -y
conda activate ann-router
```

## 2. Install os-helper and ann-router (editable)

```bash
cd ~/ann-router
pip install -e .                    # core only (library + argparse CLI)
```

At this point `import ann_router`, `ann-router route ...`, and the exact backend
already work with nothing else installed.

## 3. Install the ANN backends

Most engines install cleanly with pip:

```bash
pip install hnswlib faiss-cpu qdrant-client "pgvector" "psycopg[binary]" turbovec
```

**Apple Silicon caveat — annoy.** The PyPI `annoy` wheel is built from source
and is **miscompiled on macOS/arm64** (it returns a single neighbour regardless
of `k`; the same defect made `roitelet`'s benchmark report annoy recall 0.0).
Install the working binary from conda-forge instead:

```bash
pip uninstall -y annoy               # if a broken pip build is present
conda install -c conda-forge python-annoy -y
```

**numpy < 2.** `annoy` (any build) has no numpy-2 ABI, so pin numpy below 2 in
any environment that uses annoy:

```bash
pip install "numpy<2"                # 1.26.4 verified with every other backend
```

The verified dev environment therefore runs **numpy 1.26.4** with faiss-cpu
1.15, hnswlib 0.8, turbovec 0.8, qdrant-client 1.19, pgvector 0.5, psycopg 3.3.

## 4. Install the extra "doors" (optional)

```bash
pip install click                    # ann-router-click CLI       ([cli])
pip install "fastapi<1" "uvicorn<1" httpx   # HTTP API             ([api])
pip install "fastapi-mcp<1" "mcp<2,>=1.20"  # MCP server (mounts on FastAPI)  ([mcp])
```

Or, using the package extras:

```bash
pip install -e '.[all]'              # every pip-installable backend + cli + api
pip install -e '.[dev]'             # the above + pytest for the test suite
```

## 5. Backends that need extra setup here

| Backend  | Status on this Mac | Why / how to get it |
| -------- | ------------------ | ------------------- |
| **pgvector** | installs; needs a server | The Python side (`pgvector` + `psycopg`) installs, but the backend needs a **live PostgreSQL** with the `vector` extension. Point `ANN_ROUTER_PG_DSN` at one to enable it, otherwise its tests skip cleanly. No Docker needed — see the disk-frugal conda-forge recipe below (or `docker` via `run_bench.sh pg-up`). |

This does not break anything: importing `ann_router` never requires an engine,
and the router routes around uninstalled/unreachable backends with an
explained fallback. (ScaNN is not a registered backend at all — no
Apple-Silicon wheel; the project has dropped it entirely, see CHANGELOG.md.)

### 5a. Enabling pgvector without Docker (conda-forge, disk-frugal)

You do **not** need Docker to exercise the pgvector backend. A local PostgreSQL
server plus the `vector` extension both come from conda-forge (~200 MB), which is
much lighter than pulling a Docker image and works on Apple Silicon:

```bash
# 1. Add a real Postgres server + the pgvector extension to the env
conda install -n ann-router -c conda-forge postgresql pgvector -y

# 2. initdb a throwaway data dir (kept inside the repo, gitignored) and start it.
#    Use trust auth on loopback only — this is a local test server, not production.
export PGDATA="$PWD/.pgdata"            # .pgdata/ and .pgdata.log are in .gitignore
export PGPORT=5432                      # fall back to 5433 if 5432 is taken
initdb -D "$PGDATA" -U "$USER" --auth=trust -E UTF8
pg_ctl -D "$PGDATA" -l "$PWD/.pgdata.log" \
       -o "-p $PGPORT -k /tmp -c listen_addresses=127.0.0.1" start

# 3. Create the database and the extension
createdb -h 127.0.0.1 -p "$PGPORT" -U "$USER" annrouter
psql -h 127.0.0.1 -p "$PGPORT" -U "$USER" -d annrouter -c "CREATE EXTENSION vector;"

# 4. Point the backend at it and run the pgvector tests
export ANN_ROUTER_PG_DSN="postgresql://$USER@127.0.0.1:$PGPORT/annrouter"
pytest -q -k pgvector          # 4 passed (recall@10 = 1.00, save/load skips by design)

# 5. When done: stop the server and delete the throwaway data dir
pg_ctl -D "$PGDATA" stop -m fast
rm -rf "$PGDATA" "$PWD/.pgdata.log"
```

With the DSN exported, more of the pgvector lifecycle test runs (the same
`test_backend_lifecycle` case every other backend gets); it still skips its
save/load round trip *by design* — pgvector persists via its DSN + table, not
`save()`/`load()`, the same design as qdrant.

## 6. Verify

```bash
python -c "import ann_router as ar; print(ar.available_backends())"
ann-router bench --n 3000 --dim 64        # recall@k per installed backend
pytest -q                                 # full suite (skips uninstallable)
```

Expected on this Mac: `['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant',
'pgvector']` available; `pytest` reports around **95 passed, 2 skipped**
without a Postgres server (the pgvector-needs-a-server path), fewer skips
once `ANN_ROUTER_PG_DSN` points at a live pgvector server (see § 5a).
