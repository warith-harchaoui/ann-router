#!/usr/bin/env python3
"""Make the landscape CSVs the single source of truth for the comparison docs.

Each ``references/landscape-<lang>.csv`` (options as rows, criteria as
columns, integer 1-5 ratings in the cells) drives two rendered artefacts:

* the **star table** inside the matching Markdown file (``LANDSCAPE.md`` /
  ``PAYSAGE.md``): integers become ``⭐`` runs, and the row labels regain their
  editorial voice ("Just use **FAISS**" in English, "**FAISS** seul" in French);
* the **positioning map** SVG under ``assets/``: produced by the external
  ``standpoint`` tool, which reads the same CSV.

Run it after any edit to a CSV so the table and the map never drift from the
numbers::

    python scripts/sync_docs_from_csv.py            # tables + SVGs
    python scripts/sync_docs_from_csv.py --no-svg   # tables only (skip standpoint)

Author: Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent

# One entry per language: which CSV feeds which doc, how to decorate its labels,
# the standpoint output stem, and the reference row pinned to the top-right.
TARGETS = [
    {
        "csv": ROOT / "references/landscape-en.csv",
        "md": ROOT / "LANDSCAPE.md",
        "lang": "en",
        "stem": "landscape",
    },
    {
        "csv": ROOT / "references/landscape-fr.csv",
        "md": ROOT / "PAYSAGE.md",
        "lang": "fr",
        "stem": "paysage",
    },
]


def decorate(label: str, lang: str) -> str:
    """Return the Markdown-styled row label for ``label``.

    Every option is simply bolded — no "Just use" / "seul" framing.

    Parameters
    ----------
    label : str
        The plain option name from the CSV's first column.
    lang : {"en", "fr"}
        Target language (accepted for a uniform call site; unused here).

    Returns
    -------
    str
        ``**X**``.
    """
    return f"**{label}**"


def build_table(csv_path: Path, lang: str) -> list[str]:
    """Render the CSV as GitHub-flavoured Markdown table lines (with stars).

    Parameters
    ----------
    csv_path : pathlib.Path
        Source ratings CSV (header row, then one option per row).
    lang : {"en", "fr"}
        Language passed through to :func:`decorate`.

    Returns
    -------
    list of str
        The table lines (header, separator, one per option) without newlines.
    """
    rows = list(csv.reader(csv_path.open(encoding="utf-8")))
    header, body = rows[0], [r for r in rows[1:] if r]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        cells = [decorate(r[0], lang)] + ["⭐" * int(v) for v in r[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def replace_table(md_path: Path, table: list[str]) -> bool:
    """Swap the single Markdown pipe-table in ``md_path`` for ``table``.

    Locates the one contiguous block of lines beginning with ``|`` (each doc has
    exactly one such table — the at-a-glance grid) and replaces it in place.

    Parameters
    ----------
    md_path : pathlib.Path
        The Markdown file to edit.
    table : list of str
        Replacement table lines from :func:`build_table`.

    Returns
    -------
    bool
        ``True`` if the file changed on disk.
    """
    lines = md_path.read_text(encoding="utf-8").splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("|")), None)
    if start is None:
        raise SystemExit(f"{md_path.name}: no Markdown table found to update")
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    new = lines[:start] + table + lines[end:]
    if new == lines:
        return False
    md_path.write_text("\n".join(new) + "\n", encoding="utf-8")
    return True


def render_svg(csv_path: Path, stem: str, model: str) -> None:
    """Regenerate the positioning-map SVG for ``csv_path`` via standpoint.

    Parameters
    ----------
    csv_path : pathlib.Path
        The ratings CSV (also the table source).
    stem : str
        Output basename under ``assets/`` (e.g. ``landscape``).
    model : str
        Ollama model standpoint uses to name the axes.
    """
    subprocess.run(
        [
            sys.executable,
            "-m",
            "standpoint",
            str(csv_path),
            "-r",
            "ann-router",
            "-o",
            str(ROOT / "assets"),
            "--stem",
            stem,
            "--model",
            model,
        ],
        check=True,
        cwd=ROOT,
    )


# ``-h`` beside ``--help``. argparse supplied the short form for free and
# click does not, so the conversion has to ask for it back.
_CONTEXT: dict[str, object] = {"help_option_names": ["-h", "--help"]}


@click.command("sync_docs_from_csv", context_settings=_CONTEXT)
@click.option("--no-svg", is_flag=True, help="update tables only, skip standpoint")
@click.option("--model", default="gemma3:12b", help="Ollama model for axis naming")
def cli(no_svg: bool, model: str) -> None:
    """Sync every configured CSV into its Markdown table and SVG.

    \f

    Everything past the form feed is for readers of the source: click
    truncates --help there. The argparse version handed it the whole module
    docstring as the description, which printed that RST to users verbatim.

    Parameters
    ----------
    no_svg : bool
        Update the Markdown tables only, skipping the standpoint render.
    model : str
        Ollama model standpoint uses to name the axes.
    """
    for t in TARGETS:
        changed = replace_table(t["md"], build_table(t["csv"], t["lang"]))
        print(f"{t['md'].name}: table {'updated' if changed else 'unchanged'} from {t['csv'].name}")
        if not no_svg:
            render_svg(t["csv"], t["stem"], model)
            print(f"{t['stem']}.white.svg: regenerated")


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return its exit code instead of exiting.

    ``standalone_mode=False`` is what keeps this an ordinary function:
    ``tests/test_scripts.py`` calls ``main()`` and asserts on what it printed,
    which a click command line that called :func:`sys.exit` itself would turn
    into a ``SystemExit`` the test has to catch. The two ``bench/`` command
    lines do run in standalone mode, like :mod:`ann_router.cli` — they are
    ``python -m`` entry points, not modules a test loads by path.

    The observable behaviour is argparse's: 0 on success, 2 on a usage error.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on success, or click's own code (2) for a usage error.
    """
    try:
        return int(cli.main(args=argv, prog_name="sync_docs_from_csv", standalone_mode=False) or 0)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
