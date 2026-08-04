# Changelog

All notable changes to `ann-router` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-04

Initial release. First cut of the ANN vector-search router for the *sev7n AI
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
