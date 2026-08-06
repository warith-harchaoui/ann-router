"""Docs-consistency guard — keep the README and package honest with each other.

A lightweight version of the sibling repos' README-pin test: it verifies the
README documents every backend the registry actually ships and that the package
version string is consistent, so a backend added in code without a doc mention
(or a stale version) fails CI rather than shipping.

Author: Warith Harchaoui <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from pathlib import Path

import ann_router as ar
from ann_router.registry import BACKENDS

ROOT = Path(__file__).resolve().parent.parent


def test_readme_mentions_every_backend() -> None:
    """README.md names every backend the registry actually ships."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    for name in BACKENDS:
        # turbovec/hnsw/faiss/... must each appear so the docs never lag the code.
        assert name in readme, f"README does not mention backend '{name}'"


def test_version_is_consistent() -> None:
    """pyproject.toml's version matches ann_router.__version__."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'version = "{ar.__version__}"' in pyproject


def test_changelog_has_current_version() -> None:
    """CHANGELOG.md mentions the current ann_router.__version__."""
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert ar.__version__ in changelog
