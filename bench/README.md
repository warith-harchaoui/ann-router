# bench — measured calibration of the routing thresholds

Turns ann-router's policy from *guessed* to *measured*. The harness sweeps every
installable backend across corpus sizes, tuning each to **hit a recall target**
before timing it, and records the result into a **resumable YAML store**. The
calibration step then derives each threshold (`EXACT_MAX_N`, `FAISS_MIN_N`,
`HIGH_RECALL`, memory factors) **with the rows that justify it**.

Strategy — **coarse → interpolate → bisect** — so a 10M-scale study stays cheap:
1. a logarithmic `n` grid brackets where a crossover lives,
2. a log–log fit of the coarse latency curves *predicts* the crossover,
3. bisection on `n` pins it down (~8 steps), each probed cell cached.

## Layout

| file | role |
| --- | --- |
| `datagen.py` | deterministic clustered corpora + cached exact ground truth |
| `harness.py` | measure one cell; `coarse` / `bisect` / `status` CLI; YAML store |
| `calibrate.py` | derive justified thresholds → `results/calibrated_policy.yaml` |
| `plan.yaml` | the sweep grid (documentation; defaults also in `harness.py`) |
| `results/` | `measurements.yaml` (the store) + `gt/` (cached ground truth) |

## Launch (piece by piece, over days)

Everything is idempotent: re-running only measures cells not already present, so
you can stop and resume freely. Bound each run with `--max-n`, `--max-cells`, or
`--time-budget`.

```bash
# Day 1 — small/medium regime, ~10-20 min: pins EXACT_MAX_N precisely.
python -m bench.harness coarse --max-n 100000 --time-budget 1200

# See progress at any time.
python -m bench.harness status

# Day 2 — up to 1M (heavier cells).
python -m bench.harness coarse --max-n 1000000 --time-budget 3600

# Day 3+ — push to 10M (low dim only; big cells are skipped + logged, never silently).
python -m bench.harness coarse --max-n 10000000 --dims 128 --time-budget 3600

# Pin a specific crossover by dichotomy (seeded by interpolation of the coarse data):
python -m bench.harness bisect --a exact --b hnsw  --dim 768 --target 0.95   # -> EXACT_MAX_N
python -m bench.harness bisect --a hnsw  --b faiss --dim 768 --target 0.95   # -> FAISS_MIN_N

# Derive the justified thresholds once enough cells exist.
python -m bench.calibrate --dims 128 384 768
cat bench/results/calibrated_policy.yaml
```

Tuning knobs: `--dims`, `--targets`, `--k`, `--nq`, `--seed`, `--backends`, and
`--metric` on `coarse`; the RAM ceiling is `ANN_BENCH_MEM_GB` (default 60).

## Coverage on this machine (Apple Silicon)

Directly measurable: **exact, turbovec, hnsw, faiss, annoy**. Gaps are explicit,
never silently filled:
- **scann** — no Apple-Silicon wheel (impossible here); heuristic retained.
- **qdrant / pgvector** — need a running server; measured only when reachable.

## Applying results

`calibrate.py` never edits the shipped policy. Review
`results/calibrated_policy.yaml`, then update `ann_router/policy.yaml` +
`POLICY_VERSION` deliberately (a threshold bump is a policy change with a
CHANGELOG entry).
