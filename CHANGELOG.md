# Changelog

All notable changes to `ann-router` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- **`ann_router.policy.PROVISIONAL_ROUTING`** (new, default `True`): until the
  calibration above is reviewed and deliberately applied to
  `ann_router/policy.yaml`, the live router (`ann_router.router.route`)
  redirects every non-exact pick to `turbovec` instead of trusting an
  unmeasured threshold. `exact` stays exact below `EXACT_MAX_N` (pure
  brute-force math, not a guess). The pure decision tree in
  `ann_router.policy.rank_backends` is untouched — this is a router-layer
  override, easy to flip off once the policy is calibrated.
- **Test suite consolidated**: `tests/test_backends.py`'s five parametrized
  checks per backend (recall, save/load, add, remove, capabilities) merged
  into one `test_backend_lifecycle` per backend plus the standalone
  capability-descriptor check, cutting collected test items while keeping
  the same behavioural coverage on one continuously-built index. Coverage of
  `ann_router/` rose to ~86%.
- Fixed the `turbovec` `bit_width` sweep ladder (`bench/harness.py`,
  `bench/plan.yaml`): `8` is not a valid `bit_width` (only 2/3/4 are) and was
  crashing every turbovec calibration cell.

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

## [0.1.0] — 2026-08-04

Initial release. First cut of the ANN vector-search router for the *AI
Helpers* suite.

### Added
- **Router** (`ann_router.route`, `ann_router.auto_index`): selects a backend
  from measured `Criteria` and returns a justified, discussable
  `BackendChoice` (chosen backend + rationale + considered shortlist + config).
- **Pure policy** (`ann_router.policy`) with versioned thresholds
  (`POLICY_VERSION = 1.0.0`) encoding the suite's decision tree:
  exact → turbovec → FAISS → Qdrant/pgvector → ScaNN → Annoy → HNSW.
- **Common `ANNIndex` interface** (`build`/`add`/`add_with_ids`/`remove`/
  `search`/`save`/`load`) plus a `Capabilities` descriptor and the
  `NotSupported` / `BackendUnavailable` errors.
- **Eight backend adapters** with lazy/optional imports: `exact` (always
  available, numpy reference + ground truth), `turbovec`, `hnsw`, `faiss`,
  `annoy`, `scann`, `qdrant`, `pgvector`. Qdrant and pgvector expose a
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
- ScaNN ships no macOS/arm64 wheel; its adapter is present and its tests skip
  where the dependency is absent.
