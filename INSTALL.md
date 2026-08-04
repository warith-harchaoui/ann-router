# Installing ann-router

`ann-router` uses **conda (miniconda) + pip** as its base tooling (the suite
standard is conda + poetry; poetry is not required for this pure-setuptools
package). The core is deliberately tiny — `numpy` + `os-helper` + `pyyaml` — so
the library and the always-on argparse CLI install in seconds. Every ANN engine
is an optional extra you add only if you route to it.

The exact commands below were run and verified on this machine
(**Apple M2 Max, arm64, macOS**) on 2026-08-04.

## 1. Create the environment

```bash
conda create -n ann-router python=3.11 -y
conda activate ann-router
```

## 2. Install os-helper (editable) and ann-router (editable)

```bash
pip install -e ~/os-helper          # the suite foundation brique
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
pip install mcp                      # MCP server                  ([mcp])
```

Or, using the package extras:

```bash
pip install -e '.[all]'              # every pip-installable backend + cli + api
pip install -e '.[dev]'             # the above + pytest for the test suite
```

## 5. Backends that do NOT install here

| Backend  | Status on this Mac | Why / how to get it |
| -------- | ------------------ | ------------------- |
| **scann** | not installable | Google ScaNN ships wheels for **Linux/x86** only. The adapter is present with a lazy import; its tests skip. Install on Linux with `pip install scann`. |
| **pgvector** | installs, tests skip | The Python side (`pgvector` + `psycopg`) installs, but the backend needs a **live PostgreSQL** with the `vector` extension. Point `ANN_ROUTER_PG_DSN` at one to enable it (e.g. the `pgvector/pgvector` Docker image), otherwise its tests skip cleanly. |

Neither absence breaks anything: importing `ann_router` never requires an engine,
and the router routes around uninstalled backends with an explained fallback.

## 6. Verify

```bash
python -c "import ann_router as ar; print(ar.available_backends())"
ann-router bench --n 3000 --dim 64        # recall@k per installed backend
pytest -q                                 # full suite (skips uninstallable)
```

Expected on this Mac: `['exact', 'turbovec', 'hnsw', 'faiss', 'annoy', 'qdrant',
'pgvector']` available; `pytest` reports **103 passed, 10 skipped** (scann +
pgvector-server paths).
