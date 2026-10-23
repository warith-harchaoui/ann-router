"""Hardware probing: fill in the ``hardware`` criterion automatically.

The router's decision changes with the accelerator: a GPU unlocks FAISS batch
search, Apple Silicon makes turbovec's Rust/NEON path especially attractive, and
a plain CPU box narrows the field. Rather than make the caller hand-classify
their machine, this module probes it the same way the ``best-engine-ai-helper``
sibling's ``detect.py`` does: cheap subprocess/library calls with graceful
fallbacks, never raising, always returning one of the three ``HardwareName``
values.

Consumes: ``os_helper`` (worker count), optional ``psutil``/``torch`` if present.
Produces: :func:`detect_hardware`, :func:`hardware_report`.

Author: Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import functools
import os
import platform

import os_helper as osh

from .spec import HardwareName


@functools.lru_cache(maxsize=1)
def _has_cuda_gpu() -> bool:
    """Return ``True`` if an NVIDIA CUDA GPU looks usable on this machine.

    Cached: the answer cannot change inside one process, and the probes below
    are not free.
    """
    # Prefer a real driver probe: nvidia-smi exits 0 only when a GPU is present.
    try:
        res = osh.system("nvidia-smi -L", check_exitcode=False)
        if res.get("out", "").strip().upper().startswith("GPU"):
            return True
    except Exception:  # noqa: BLE001 - probing must never raise
        pass

    # Fall back to torch's view where CUDA can actually exist — which is not
    # macOS. Two reasons to gate this, and the second is the serious one:
    #
    #   * Apple never shipped CUDA, so the answer there is always False and the
    #     import buys nothing.
    #   * `import torch` is not free of consequence. On macOS it loads its own
    #     `libomp.dylib`, and FAISS loads another; the second threaded FAISS
    #     call in that process then dies with `OMP: Error #15` — an abort, not
    #     an exception, so nothing can catch it. This package's own test suite
    #     hit exactly that: asking for capabilities pulled torch in, and the
    #     next `bench` killed the interpreter. A capability probe has no
    #     business loading a deep-learning framework as a side effect.
    #
    # The documented escape, `KMP_DUPLICATE_LIB_OK=TRUE`, is described upstream
    # as unsafe and liable to "silently produce incorrect results" — not a
    # trade worth making underneath a recall benchmark.
    if platform.system() == "Darwin":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def _is_apple_silicon() -> bool:
    """Return ``True`` on an arm64 macOS host (M-series)."""
    # macOS + arm64 uniquely identifies Apple Silicon; Rosetta reports x86 so
    # this stays honest about the native architecture.
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def detect_hardware() -> HardwareName:
    """Classify the local accelerator into the router's three-way taxonomy.

    Returns
    -------
    {"gpu", "apple_silicon", "cpu"}
        ``"gpu"`` when a CUDA GPU is found (it dominates the batch regime),
        else ``"apple_silicon"`` on M-series Macs, else ``"cpu"``.

    Examples
    --------
    >>> detect_hardware() in {"gpu", "apple_silicon", "cpu"}
    True
    """
    # GPU wins first: its effect on the policy (FAISS batch) outranks the CPU
    # microarchitecture distinction below it.
    if _has_cuda_gpu():
        return "gpu"
    if _is_apple_silicon():
        return "apple_silicon"
    return "cpu"


def hardware_report() -> dict[str, object]:
    """Return a small JSON-ready dict describing the host for the CLI/API.

    Returns
    -------
    dict
        ``os``, ``machine``, ``cpu_count``, ``workers`` and the classified
        ``hardware`` label.

    Examples
    --------
    >>> report = hardware_report()
    >>> report["hardware"] in {"gpu", "apple_silicon", "cpu"}
    True
    """
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "workers": osh.get_nb_workers(),  # suite-standard worker resolution
        "hardware": detect_hardware(),
    }
