# Changelog

All notable changes to `ann-router` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.4] - 2026-08-13

### Fixed

- **`rank_backends()` never actually consulted `HIGH_RECALL`**, despite the
  threshold being calibrated specifically for this (`bench/results/
  calibrated_policy.yaml`: turbovec's measured recall consistently undershoots
  0.9 at every calibrated dim/n) and `Criteria.target_recall`'s own docstring
  promising "high values push toward exact/HNSW and away from aggressively
  quantised indexes." The `turbovec` rule fired for any `dynamic=True` corpus
  regardless of `target_recall`, so a caller asking for the house default
  (`target_recall=0.95`) on a dynamic corpus was silently routed to a backend
  proven not to meet that recall. The `turbovec` rule now also requires
  `target_recall < HIGH_RECALL`; the `hnsw` fallback rule's rationale now
  distinguishes this case (recall too high for turbovec) from the pre-existing
  one (turbovec policy-eligible but not installed at runtime). `POLICY_VERSION`
  bumped `1.1.0` -> `1.2.0` (a decision-tree logic change, not just a threshold
  value). README/LISEZMOI/EXAMPLES/EXEMPLES updated: the Quick Start demo used
  `target_recall=0.95` with `dynamic=True` and claimed `'turbovec'`, which is
  no longer true and never should have been claimed as the default behavior.

## [0.1.3] - 2026-08-13

### Fixed

- **`HNSWIndex`/`AnnoyIndex`/`TurboVecIndex.search()` did not pad short
  results to `(q, k)`**: `ANNIndex.search()`'s documented contract is a
  rectangular `(q, k)` result with `-1`-padded ids when the corpus has fewer
  than `k` points (already honoured by `ExactIndex`, `FaissIndex`,
  `QdrantIndex` and `PgVectorIndex`), but hnswlib, Annoy and turbovec all
  return only as many columns as the corpus actually has, so those three
  adapters silently returned a narrower array instead. Added a shared
  `ANNIndex._pad()` helper (used by `ExactIndex` too, replacing its private
  duplicate) and applied it to the three affected backends. Covered by a new
  regression assertion in `tests/test_backend_lifecycle` (build a 3-vector
  corpus, search for k=10, assert the padded shape and fill value).

### Changed (bundled, previously unreleased on `main`)

- Converted `LANDSCAPE.md`/`PAYSAGE.md` to the suite's mechanical
  CSV -> standpoint-figure pipeline.
- Full documentation rewrite pass for `WRITING.md`/`ECRITURE.md` compliance
  and freshness; added `EXEMPLES.md` (French sibling of `EXAMPLES.md`) and a
  README "Documentation" section (`Install` renamed to `Installation`).
- `requirements-dev.txt` is now derived from `pyproject.toml`'s `[dev]`
  extra rather than hand-maintained.
- Updated `blob`/`raw` GitHub links from `master` to `main` (the default
  branch was renamed).

## [0.1.2] - 2026-08-09

### Changed

- **CI had no lint job at all** (only `test` and `package`). Added a `lint`
  job running `ruff check .` + `ruff format --check .`, matching the rest
  of the suite; the tree was already clean.
- **`assets/logo.png` recolored to the suite reference palette**
  (OT color-transfer, applied in an earlier session but never committed).

## [0.1.1] - 2026-08-08

- **Applied the bench calibration to `policy.py`**: thresholds bumped from
  the provisional values to the measured crossovers in `bench/results/`;
  provisional routing retired now that real numbers back every threshold.
- **Fixed `_fit_predict_crossover` ignoring `target`/`nq`/`seed`** when
  gathering points for the dichotomic bisection search.
- **Suite-consistency fixes for full `ai-helpers` membership**: added the
  missing `ann-router-mcp` console-script entry (`ann_router.mcp_server:main`)
  — every other surface (argparse CLI, click CLI, FastAPI API, FastAPI-MCP)
  already existed in code but had no installed entry point for the MCP/API
  door, unlike every other suite member's `<name>-mcp` script. Fixed a
  `__version__`/`pyproject.toml` drift (`0.1.0` vs the just-bumped `0.1.1`)
  caught by this repo's own `test_version_is_consistent` guard.

## [0.1.0] - 2026-08-06

Initial release. First cut of the ANN vector-search router for the *AI
Helpers* suite. (The suite's `CHANGELOG` convention keeps every change ever
made under the version that first ships it — nothing below this line reached
PyPI before today, so it is all one initial release, not a diff against a
prior published version.)

### Changed
- **MCP door rebuilt on `fastapi-mcp`**: `ann_router/mcp_server.py` no longer
  hand-writes three `@server.tool()` wrappers duplicating `api.py`'s
  route/capabilities/bench parameter lists and docstrings. It now mounts
  `fastapi-mcp` on a fresh copy of the same FastAPI app (each route tagged
  with an explicit `operation_id`), so the three MCP tools are generated
  from — and always match — the REST API's schema. This is an architectural
  change: MCP is now served over **Streamable HTTP** (`/mcp` on a running
  app, default port 8019), not stdio; `pip install 'ann-router[mcp]'` now
  pulls in `fastapi`/`uvicorn` too. Verified end to end (initialize handshake,
  `tools/list`, `tools/call` for all three tools) against a live server, not
  just import-time. `mcp<2` is pinned — `fastapi-mcp==0.4.0` breaks against
  the `mcp==2.0.0` `Server()` signature change.

### Added
- **`bench/` calibration sweep run to completion**: 491 measured cells across
  `exact, turbovec, hnsw, faiss, annoy, qdrant, pgvector` (n up to 1,000,000;
  dim in {128, 384, 768}), including dichotomic bisection of the
  `EXACT_MAX_N`/`FAISS_MIN_N` crossovers. See `bench/results/`.
- **`bench.calibrate` now also emits a coloured Mermaid decision tree**
  (`bench/results/decision_tree.md`) with the measured thresholds filled in
  per dimension, regenerated on every calibration run.
- **`bench.harness dry-run`**: a fast (~seconds), exhaustive-over-backends
  smoke test on a tiny grid, never touching `measurements.yaml`. Catches a
  broken backend before it pollutes a real measurement (see Fixed, below).
- **`run_bench.sh`** (repo root): the sweep launcher, with an `all` target
  chaining `dry -> day1 -> day2 -> day3 -> bisect -> calibrate`, resumable at
  every step. Pins the `ann-router` conda env's interpreter explicitly.
- **`environment.yaml`**: minimal conda environment for local use, delegating
  every actual dependency to `requirements.txt`.
- **`Dockerfile`**: a server image with every pip-installable backend + the
  `[api]` extra, serving the HTTP door on port 8018.
- **CI**: ultra-light GitHub Actions (`.github/workflows/tests.yml`) and
  GitLab CI (`.gitlab-ci.yml`) — a `test` job runs the suite on the
  always-available core install, and a `package`/`package` stage only runs
  (never fails) when `test` succeeds, via a `needs:` dependency.
- **`Criteria.latency_budget_ms` is now load-bearing**: it was collected but
  never consulted. `ann_router.policy.effective_exact_max_n` scales
  `EXACT_MAX_N` by `latency_budget_ms / LATENCY_REFERENCE_MS` (a new
  threshold, default 10 ms — the same reference `bench.calibrate.exact_max_n`
  was already calibrated against) before every rule's n-vectors comparison,
  since a brute-force scan's cost is ~linear in n for fixed dim: a tighter
  budget shrinks the exact/ANN crossover, a looser one extends it. Every
  other threshold (`FAISS_MIN_N`, `HIGH_RECALL`) is unaffected.

### Changed
- **Calibration applied to `ann_router/policy.yaml`** (`POLICY_VERSION`
  `1.0.0` -> `1.1.0`): `bench.calibrate` measured per-dimension thresholds
  from the 491-cell sweep (`bench/results/calibrated_policy.yaml`) —
  `EXACT_MAX_N` 384->1000/768->5000/128->never lost in range, `FAISS_MIN_N`
  128->20000/384->5000/768->1037, `HIGH_RECALL` 0.9 consistently across all
  three dims. Since the shipped policy uses one scalar per threshold (not
  per-dim), each was reduced conservatively across the measured dims —
  `EXACT_MAX_N` takes the minimum non-null value (`10000` -> `1000`),
  `FAISS_MIN_N` the maximum (`1000000` -> `20000`), `HIGH_RECALL` was
  unanimous (`0.95` -> `0.9`) — so no threshold is trusted beyond what was
  actually measured at every calibrated dim. The now-obsolete
  `ann_router.policy.PROVISIONAL_ROUTING` router-layer override (which had
  redirected every non-exact pick to `turbovec` while the calibration above
  was still unreviewed) is removed; `ann_router.router.route` now trusts
  `rank_backends` directly.
- **Test suite consolidated**: `tests/test_backends.py`'s five parametrized
  checks per backend (recall, save/load, add, remove, capabilities) merged
  into one `test_backend_lifecycle` per backend plus the standalone
  capability-descriptor check, cutting collected test items while keeping
  the same behavioural coverage on one continuously-built index. Coverage of
  `ann_router/` rose to ~86%.
- Fixed the `turbovec` `bit_width` sweep ladder (`bench/harness.py`,
  `bench/plan.yaml`): `8` is not a valid `bit_width` (only 2/3/4 are) and was
  crashing every turbovec calibration cell.
- **`.gitignore` reviewed end to end**, not just appended to. The comparison
  CSVs behind `LANDSCAPE.md`/`PAYSAGE.md` (`references/landscape-{en,fr}.csv`,
  moved from `.private/`) and their generator (`scripts/sync_docs_from_csv.py`)
  are now tracked: both are the reproducible source of already-committed docs
  and contain nothing private (public 1-5 competitive ratings, no secrets),
  so a fresh clone could not previously regenerate those tables/SVGs at all.
  `.private/` stays ignored for genuine personal session notes. The local
  index-artifact patterns (`*.ann`, `*.tv`, `*.faiss`) were replaced with
  what the code and docs actually produce (`*.idx`, `*.hnsw`, `*.meta.json`,
  `*.ids.npy`) — the old ones matched no real backend output. The dead
  `*config.json` rule (no JSON config anywhere in this project — it's all
  YAML) was dropped. `bench.calibrate`'s `source:` field in
  `calibrated_policy.yaml`/`decision_tree.md` now records a repo-relative
  path instead of the generating machine's absolute path.

### Removed
- **ScaNN dropped entirely** as a supported backend (adapter, registry entry,
  policy rule, `SCANN_MIN_N`/`SCANN_MIN_RECALL` thresholds, `backends.yaml`
  catalog entry, `scann` extra in `pyproject.toml`) — no Apple-Silicon wheel
  exists, and the project has definitively abandoned it rather than keep
  carrying an always-unmeasured heuristic. `ann-router` now routes among six
  backends: exact, turbovec, hnsw, faiss, annoy, qdrant, pgvector (plus the
  pgvector/qdrant persistence layer — seven registered adapters total).

### Fixed
- **annoy measurements recorded under the wrong interpreter**: the base conda
  env's `annoy` PyPI build is silently broken on this machine (near-empty
  neighbour lists, no error) while the project's own `ann-router` conda env
  works correctly. 23 polluted `measurements.yaml` rows were purged and
  `run_bench.sh` now pins the interpreter explicitly so this cannot recur
  silently — `bench.harness dry-run` also now catches this class of bug in
  seconds instead of a multi-hour sweep.
- **`bench.harness._fit_predict_crossover` ignored `target`/`nq`/`seed`**:
  the point filter that gathers each backend's measured `(n, p50)` pairs
  only matched on `backend`/`dim`/`k`/`metric`, so its log-log crossover fit
  could silently mix latency points measured at different recall targets or
  query configurations into one curve. Now filters on `target_recall`/`nq`/
  `seed` too, so the fit only ever compares apples to apples — one operating
  point. `cmd_bisect`'s use of the prediction was already safe either way
  (it only narrows the bisection bracket when the prediction falls strictly
  inside `[lo, hi]`, else falls back to the full range), so this changes the
  quality of the bracket seed, not the correctness of the final bisected
  crossover.

### Added
- **Router** (`ann_router.route`, `ann_router.auto_index`): selects a backend
  from measured `Criteria` and returns a justified, discussable
  `BackendChoice` (chosen backend + rationale + considered shortlist + config).
- **Pure policy** (`ann_router.policy`) with versioned thresholds
  (`POLICY_VERSION = 1.1.0`) encoding the suite's decision tree:
  exact → turbovec → FAISS → Qdrant/pgvector → Annoy → HNSW.
- **Common `ANNIndex` interface** (`build`/`add`/`add_with_ids`/`remove`/
  `search`/`save`/`load`) plus a `Capabilities` descriptor and the
  `NotSupported` / `BackendUnavailable` errors.
- **Seven backend adapters** with lazy/optional imports: `exact` (always
  available, numpy reference + ground truth), `turbovec`, `hnsw`, `faiss`,
  `annoy`, `qdrant`, `pgvector`. Qdrant and pgvector expose a
  metadata-filtering `search_filter`.
- **Five surfaces over one core**: library, argparse CLI (`ann-router`), click
  CLI (`ann-router-click`), FastAPI app (`ann_router.api`), MCP server
  (`ann_router.mcp_server`), and a `skills/ann-router` agent skill.
- **Tunable YAML config** (`policy.yaml`, `backends.yaml`, `hardware.yaml`) read
  by `ann_router.config`, overridable via `ANN_ROUTER_POLICY`.
- **Hardware probing** (`ann_router.detect`) classifying cpu / apple_silicon / gpu.
- **Test + eval suite**: decision-tree assertions, per-backend recall@k against
  the exact ground truth above versioned floors, save/load round-trips, and
  add/remove capability checks, with clean skips for uninstallable backends.

### Notes
- Built on the shipped brute-force → turbovec routing prototype from `roitelet`.
