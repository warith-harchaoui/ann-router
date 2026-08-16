"""Load the shipped YAML config (policy thresholds, backend + hardware catalogs).

The suite keeps its *data* in commented YAML next to the code (best-engine's
``models.yaml`` / ``hardware.yaml``); ann-router does the same with
``policy.yaml`` (the tunable thresholds), ``backends.yaml`` (the prose backend
catalog) and ``hardware.yaml`` (the accelerator profiles). This module is the
single reader for them: the Python constants in :mod:`ann_router.policy` stay the
canonical defaults, and these functions let an operator *override* the
thresholds from a YAML file (or the ``ANN_ROUTER_POLICY`` env var) without
touching code: the "example config, profusely commented" house rule.

Consumes: ``pyyaml``; the packaged ``*.yaml`` files (via importlib.resources).
Produces: :func:`policy_thresholds`, :func:`backend_catalog`,
:func:`hardware_profiles`.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import os
from importlib import resources
from typing import Any

import yaml

from .policy import THRESHOLDS


def _load_packaged(name: str) -> dict[str, Any]:
    """Read a YAML file shipped inside the ``ann_router`` package.

    Parameters
    ----------
    name : str
        File name (e.g. ``"policy.yaml"``).

    Returns
    -------
    dict
        Parsed YAML content.

    Examples
    --------
    >>> _load_packaged("policy.yaml")["version"]
    '1.2.0'
    """
    # importlib.resources reads the file whether the package is installed as a
    # wheel or in editable mode, so this works in every install layout.
    text = resources.files("ann_router").joinpath(name).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def policy_thresholds(path: str | None = None) -> dict[str, float]:
    """Return the policy thresholds, optionally overridden from a YAML file.

    Resolution order: the in-code :data:`ann_router.policy.THRESHOLDS` defaults,
    overlaid with the packaged ``policy.yaml``, overlaid with an external file
    (``path`` argument or the ``ANN_ROUTER_POLICY`` env var) if present.

    Parameters
    ----------
    path : str, optional
        Path to an external ``policy.yaml`` to overlay. Falls back to the
        ``ANN_ROUTER_POLICY`` environment variable, then to no override.

    Returns
    -------
    dict
        The merged ``{THRESHOLD_NAME: value}`` mapping, ready to pass to
        :func:`ann_router.policy.rank_backends`.

    Examples
    --------
    >>> t = policy_thresholds()
    >>> t["EXACT_MAX_N"]
    1000
    """
    # Start from the code defaults so a partial YAML never drops a key.
    merged: dict[str, float] = dict(THRESHOLDS)
    merged.update(_load_packaged("policy.yaml").get("thresholds", {}))
    external = path or os.environ.get("ANN_ROUTER_POLICY")
    if external and os.path.exists(external):
        with open(external, encoding="utf-8") as fh:
            merged.update((yaml.safe_load(fh) or {}).get("thresholds", {}))
    return merged


def backend_catalog() -> list[dict[str, Any]]:
    """Return the prose backend catalog from ``backends.yaml``.

    Returns
    -------
    list of dict
        One entry per backend: ``name``, ``summary``, ``when``, ``pip_extra``.

    Examples
    --------
    >>> {b["name"] for b in backend_catalog()} >= {"exact", "hnsw", "faiss"}
    True
    """
    return _load_packaged("backends.yaml")["backends"]


def hardware_profiles() -> list[dict[str, Any]]:
    """Return the accelerator profiles from ``hardware.yaml``.

    Returns
    -------
    list of dict
        One entry per hardware class: ``hardware``, ``summary``, ``unlocks``.

    Examples
    --------
    >>> sorted(p["hardware"] for p in hardware_profiles())
    ['apple_silicon', 'cpu', 'gpu']
    """
    return _load_packaged("hardware.yaml")["profiles"]
