"""Tests for scripts/sync_docs_from_csv.py — the LANDSCAPE.md/PAYSAGE.md generator.

``render_svg`` shells out to the external ``standpoint`` tool (itself calling
Ollama) and is intentionally not exercised here — that is an optional,
heavy, network-adjacent tool, out of scope for this ultra-light suite. Every
test below drives the CSV -> Markdown-table path only (``--no-svg``), which
is also the only path CI can run.

Author: Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# scripts/ is not an installable package (gitignored from packaging, see
# pyproject.toml), so it is loaded by path rather than imported normally.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sync_docs_from_csv.py"
_spec = importlib.util.spec_from_file_location("sync_docs_from_csv", _SCRIPT_PATH)
sync_docs = importlib.util.module_from_spec(_spec)
sys.modules["sync_docs_from_csv"] = sync_docs
_spec.loader.exec_module(sync_docs)


def test_decorate_bolds_the_label() -> None:
    assert sync_docs.decorate("FAISS", "en") == "**FAISS**"
    assert sync_docs.decorate("FAISS", "fr") == "**FAISS**"  # lang is a uniform-call-site no-op


def test_build_table_renders_stars_from_ratings(tmp_path) -> None:
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("Tool,Speed,Filtering\nFAISS,5,1\nQdrant,3,5\n", encoding="utf-8")
    lines = sync_docs.build_table(csv_path, "en")
    assert lines[0] == "| Tool | Speed | Filtering |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| **FAISS** | ⭐⭐⭐⭐⭐ | ⭐ |"
    assert lines[3] == "| **Qdrant** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |"


def test_replace_table_swaps_only_the_pipe_block(tmp_path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text(
        "# Title\n\nSome prose.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nMore prose.\n",
        encoding="utf-8",
    )
    changed = sync_docs.replace_table(md_path, ["| X | Y |", "| --- | --- |", "| 9 | 9 |"])
    assert changed is True
    text = md_path.read_text(encoding="utf-8")
    assert "| X | Y |" in text
    assert "| A | B |" not in text
    assert "Some prose." in text and "More prose." in text  # surrounding content untouched


def test_replace_table_is_a_noop_when_unchanged(tmp_path) -> None:
    md_path = tmp_path / "doc.md"
    table = ["| A | B |", "| --- | --- |", "| 1 | 2 |"]
    md_path.write_text("# Title\n\n" + "\n".join(table) + "\n", encoding="utf-8")
    assert sync_docs.replace_table(md_path, table) is False


def test_replace_table_raises_when_no_table_present(tmp_path) -> None:
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Title\n\nNo table here.\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        sync_docs.replace_table(md_path, ["| A |", "| --- |"])


def test_main_updates_the_configured_docs_with_no_svg(tmp_path, monkeypatch, capsys) -> None:
    csv_path = tmp_path / "ratings.csv"
    csv_path.write_text("Tool,Speed\nFAISS,5\n", encoding="utf-8")
    md_path = tmp_path / "doc.md"
    md_path.write_text("# Title\n\n| old | table |\n| --- | --- |\n| 0 | 0 |\n", encoding="utf-8")

    monkeypatch.setattr(
        sync_docs, "TARGETS", [{"csv": csv_path, "md": md_path, "lang": "en", "stem": "test"}]
    )
    monkeypatch.setattr(sys, "argv", ["sync_docs_from_csv.py", "--no-svg"])

    sync_docs.main()

    out = capsys.readouterr().out
    assert "doc.md: table updated from ratings.csv" in out
    assert "**FAISS**" in md_path.read_text(encoding="utf-8")
