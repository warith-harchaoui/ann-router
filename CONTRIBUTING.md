# Contributing to ann-router

Thanks for helping improve the ANN vector-search router. This repo holds itself
to [CODING.md](CODING.md); please skim it before opening a PR.

## Development setup

```bash
conda create -n ann-router python=3.11 -y
conda activate ann-router
pip install -e ~/os-helper
pip install -e '.[dev]'
# Apple Silicon only: use the working annoy binary
conda install -c conda-forge python-annoy -y
pip install "numpy<2"
```

See [INSTALL.md](INSTALL.md) for the platform caveats (annoy, ScaNN, pgvector).

## Before you push

```bash
ruff check ann_router tests      # lint (must pass)
ruff format ann_router tests     # format
pytest -q                        # full suite; uninstallable backends skip cleanly
```

## Adding a new backend

1. Create `ann_router/backends/<name>_backend.py` subclassing `ANNIndex`.
   - **Lazy-import** the engine inside a `_require()` helper — importing
     `ann_router` must never require the engine.
   - Implement `capabilities()` and `is_available()` as classmethods (readable
     without the dependency).
   - Raise `NotSupported` for operations the engine genuinely cannot do, and
     `BackendUnavailable` (with the `pip install` line) when the dep is missing.
2. Register it in `ann_router/registry.py` (`BACKENDS`) in preference order.
3. Add a branch/reason to `ann_router/policy.py` if the router should ever pick
   it, plus a decision-tree row in `tests/test_policy.py`.
4. Document it in `ann_router/backends.yaml` and add a `RECALL_THRESHOLDS` floor
   in `tests/conftest.py` (measured, with margin).
5. Add the extra to `pyproject.toml` (`[project.optional-dependencies]`).

The parametrised backend tests (`tests/test_backends.py`) will automatically
cover recall, save/load, and add/remove for your new backend.

## Policy / threshold changes

Thresholds are versioned (`POLICY_VERSION`, `RECALL_THRESHOLDS`). Changing one is
a deliberate act: bump the version, update `CHANGELOG.md`, and make sure the
decision-tree and recall tests still pass.

## Attribution

Commits are authored by humans only. Do **not** add any AI-assistant
`Co-Authored-By` trailer or set an AI as author/committer (CODING.md rule 13).
