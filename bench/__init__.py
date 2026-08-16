"""Calibration harness for ann-router's routing thresholds.

Turns the shipped policy from *guessed* to *measured*: :mod:`bench.harness`
sweeps backends across corpus sizes and records recall/latency into a resumable
YAML store, and :mod:`bench.calibrate` reads that store to derive each threshold
with the evidence that justifies it.

No LLM is involved anywhere in this package: it is pure numeric measurement
(numpy / faiss / hnswlib / annoy / turbovec). Everything runs locally.
"""

import os

# macOS + conda commonly link two OpenMP runtimes (faiss brings one, hnswlib /
# numpy another). When the second `libomp` initialises at first use, the process
# aborts with "OMP: Error #15 ... already initialized". Allowing the duplicate is
# the standard, widely-used workaround; set it here, before any backend loads, so
# `python -m bench.harness ...` runs without needing an env-var prefix. Override
# by exporting KMP_DUPLICATE_LIB_OK yourself if your environment has a single OMP.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
