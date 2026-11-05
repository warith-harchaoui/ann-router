"""
Every console script answers ``--help`` and returns, instead of serving.

``ann-router-mcp`` called ``uvicorn.run`` straight away and ignored its
arguments, so ``ann-router-mcp --help`` started a server on 8019 and never
returned. Each script named in ``[project.scripts]`` now runs with ``--help``
under a timeout and must print a click usage line.

Author
------
Warith HARCHAOUI <warithmetics@deraison.ai>
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# A script whose module needs an extra: without it the import fails with an
# install hint, which is the right answer and not the defect under test. The
# light CI installs the bare package, so the MCP door is checked by the
# pre-push hook, where the [mcp] extra is present.
_NEEDS = {"ann_router.mcp_server": ("fastapi_mcp", "[mcp]")}


def _scripts() -> list[tuple[str, str]]:
    # Read rather than parsed: ``tomllib`` is 3.11+.
    manifest = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = manifest.split("[project.scripts]", 1)[1].split("\n[", 1)[0]
    return re.findall(r'^([\w-]+)\s*=\s*"([\w.]+):(\w+)"', block, re.M)


@pytest.mark.parametrize(("name", "module", "func"), _scripts())
def test_the_command_answers_help(name: str, module: str, func: str) -> None:
    if module in _NEEDS and importlib.util.find_spec(_NEEDS[module][0]) is None:
        pytest.skip(f"{name} needs the {_NEEDS[module][1]} extra, not installed here")
    code = f"import sys; from {module} import {func}; sys.argv[0] = {name!r}; {func}()"
    proc = subprocess.run(
        [sys.executable, "-c", code, "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "Usage:" in proc.stdout, proc.stdout[:400]
