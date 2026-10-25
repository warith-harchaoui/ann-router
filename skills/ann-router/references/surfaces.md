# ann-router non-CLI surfaces

## 1. Library

```python
import ann_router as ar

choice = ar.route(ar.Criteria(n_vectors=2_000_000, dim=768, dynamic=True))
choice.backend  # 'turbovec'
choice.rationale  # the justification, in words
choice.alternatives  # what else was considered, and why it lost
```

Route, build and search in one step:

```python
index = ar.auto_index(vectors, ar.Criteria(n_vectors=len(vectors), dim=768))
hits = index.search(queries, k=10)
```

`ar.all_capabilities()` returns the backend × property matrix — the same data
the `capabilities` command prints.

**Keep the rationale.** `choice.backend` alone throws away the part that makes
the decision reviewable later.

Logging and file management route through `os_helper` — the suite convention.

## 2. CLI — click (core)

`ann-router`, over the shared `_core_cli` the API and MCP doors also call. See
`cli-reference.md`.

## 3. HTTP API — FastAPI (`[api]`)

```bash
pip install 'ann-router[api]'
python -m ann_router.api            # 127.0.0.1:8000, OpenAPI docs at /docs
```

Routes cover the same verbs as the CLI: routing a set of criteria, building
and searching an index, benchmarking, and reporting capabilities.

## 4. MCP — `fastapi-mcp` over the *same* app (`[mcp]`)

```bash
pip install 'ann-router[mcp]'
ann-router-mcp                      # same app, MCP mounted at /mcp
```

`ann_router.mcp_server` takes the FastAPI app that `api.py` serves and mounts
`FastApiMCP` on it, so **one entry point covers both doors** and the two
surfaces cannot drift.

Register it with Claude Code:

```bash
claude mcp add ann-router --transport http http://127.0.0.1:8000/mcp
```

## 5. Configuration

| File / variable | Purpose |
|---|---|
| `ann_router/policy.yaml` | the thresholds driving the decision |
| `ann_router/backends.yaml` | the backend capability matrix |
| `ann_router/hardware.yaml` | machine-shape hints |
| `ANN_ROUTER_POLICY` | path to a policy file that overrides the bundled one |

Tuning the policy is the supported way to adapt the router to a domain. Say
which threshold moved and why when you change one.

## Where this sits in the suite

ann-router is the **vector-search sibling of `best-engine-ai-helper`**: both
answer a "which engine" question from measured criteria and hand back a
justified choice rather than a bare name. `elbow-helper` is the third member
of that family, and the one allowed to answer "there is no defensible choice
here".
