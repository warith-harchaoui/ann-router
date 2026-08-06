#!/usr/bin/env bash
# ============================================================================
# ann-router calibration — THE SCRIPT YOU LAUNCH (not Claude).
#
# Covers every backend installable/runnable on this Mac:
#   exact · turbovec · hnsw · faiss · annoy · qdrant (embedded) · pgvector (Docker)
#   (ScaNN was dropped from the project entirely — no Apple-Silicon build exists.)
#
# Usage:
#   bash run_bench.sh dry         # ~10s smoke test, every backend (run this FIRST)
#   bash run_bench.sh day1        # small/medium (~10-20 min): pins EXACT_MAX_N
#   bash run_bench.sh status
#   bash run_bench.sh day2        # up to 1M
#   bash run_bench.sh day3        # up to 10M (low dim; big cells skipped+logged)
#   bash run_bench.sh bisect      # dichotomic crossovers (EXACT_MAX_N, FAISS_MIN_N)
#   bash run_bench.sh calibrate   # derive justified thresholds -> calibrated_policy.yaml
#   bash run_bench.sh all         # dry -> day1 -> day2 -> day3 -> bisect -> calibrate, in order
#   bash run_bench.sh pg-up       # just bring pgvector up
#   bash run_bench.sh pg-down     # tear the pgvector container down
#
# Everything is resumable: re-running only measures cells not already in
# bench/results/measurements.yaml. Stop/resume freely, day by day (Ctrl-C
# between stages is always safe — `all` just chains the same idempotent steps).
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"   # repo root (this script now lives at the top level)

# --- pin the interpreter: THIS is the bug that produced garbage annoy numbers
# once already — the base conda env's `annoy` wheel is silently broken (returns
# near-empty neighbour lists) on this machine's Python 3.13/arm64, while the
# project's own `ann-router` conda env (Python 3.11) works correctly. Never let
# `python` resolve to whatever's active in the calling shell.
ANN_ROUTER_ENV=ann-router
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [ -n "$CONDA_BASE" ] && [ -x "$CONDA_BASE/envs/$ANN_ROUTER_ENV/bin/python" ]; then
  PYTHON="$CONDA_BASE/envs/$ANN_ROUTER_ENV/bin/python"
else
  echo "[env] conda env '$ANN_ROUTER_ENV' not found." >&2
  echo "[env] create it first: conda create -n $ANN_ROUTER_ENV python=3.11 && conda activate $ANN_ROUTER_ENV && pip install -e '.[all]'" >&2
  exit 1
fi
echo "[env] using $PYTHON ($($PYTHON -V))"

# macOS duplicate-OpenMP workaround (also baked into bench/__init__.py; harmless twice).
export KMP_DUPLICATE_LIB_OK=TRUE

PG_CONTAINER=ann-router-pg
PG_PORT=5433
PG_DSN="postgresql://postgres:postgres@localhost:${PG_PORT}/postgres"

# --- pgvector via Docker (optional; the sweep just skips pgvector if it can't start) ---
pg_up() {
  if ! docker info >/dev/null 2>&1; then
    if command -v colima >/dev/null 2>&1; then echo "[pg] starting colima…"; colima start || true; fi
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "[pg] Docker daemon unavailable — pgvector will be SKIPPED (other backends still run)."
    unset ANN_ROUTER_PG_DSN 2>/dev/null || true
    return 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 || true
    echo "[pg] launching $PG_CONTAINER (pgvector/pgvector:pg16) on :${PG_PORT}..."
    docker run -d --name "$PG_CONTAINER" -e POSTGRES_PASSWORD=postgres \
      -p ${PG_PORT}:5432 pgvector/pgvector:pg16 >/dev/null || {
        echo "[pg] container failed to start — pgvector SKIPPED."; unset ANN_ROUTER_PG_DSN 2>/dev/null || true; return 0; }
  fi
  echo -n "[pg] waiting for postgres"
  for _ in $(seq 1 60); do
    docker exec "$PG_CONTAINER" pg_isready -U postgres >/dev/null 2>&1 && break
    echo -n "."; sleep 1
  done; echo
  if docker exec "$PG_CONTAINER" psql -U postgres -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null 2>&1; then
    export ANN_ROUTER_PG_DSN="$PG_DSN"
    echo "[pg] ready → ANN_ROUTER_PG_DSN=$ANN_ROUTER_PG_DSN"
  else
    echo "[pg] postgres not ready in time — pgvector SKIPPED."; unset ANN_ROUTER_PG_DSN 2>/dev/null || true
  fi
}
pg_down() { docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 && echo "[pg] container removed" || echo "[pg] nothing to remove"; }

case "${1:-help}" in
  dry)       "$PYTHON" -m bench.harness dry-run ;;
  pg-up)     pg_up ;;
  pg-down)   pg_down ;;
  status)    "$PYTHON" -m bench.harness status ;;
  day1)      pg_up; "$PYTHON" -m bench.harness coarse --max-n 100000   --time-budget 1200; "$PYTHON" -m bench.harness status ;;
  day2)      pg_up; "$PYTHON" -m bench.harness coarse --max-n 1000000  --time-budget 3600 ;;
  day3)      pg_up; "$PYTHON" -m bench.harness coarse --max-n 10000000 --dims 128 --time-budget 3600 ;;
  bisect)    "$PYTHON" -m bench.harness bisect --a exact --b hnsw  --dim 768 --target 0.95   # -> EXACT_MAX_N
             "$PYTHON" -m bench.harness bisect --a hnsw  --b faiss --dim 768 --target 0.95 ;;# -> FAISS_MIN_N
  calibrate) "$PYTHON" -m bench.calibrate --dims 128 384 768; echo "----"; cat bench/results/calibrated_policy.yaml ;;
  all)
    "$PYTHON" -m bench.harness dry-run
    pg_up
    "$PYTHON" -m bench.harness coarse --max-n 100000   --time-budget 1200
    "$PYTHON" -m bench.harness status
    "$PYTHON" -m bench.harness coarse --max-n 1000000  --time-budget 3600
    "$PYTHON" -m bench.harness coarse --max-n 10000000 --dims 128 --time-budget 3600
    "$PYTHON" -m bench.harness bisect --a exact --b hnsw  --dim 768 --target 0.95
    "$PYTHON" -m bench.harness bisect --a hnsw  --b faiss --dim 768 --target 0.95
    "$PYTHON" -m bench.calibrate --dims 128 384 768
    echo "----"; cat bench/results/calibrated_policy.yaml
    ;;
  *) sed -n '2,32p' "$0" ;;   # print the usage header
esac
