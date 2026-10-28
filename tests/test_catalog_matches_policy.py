"""Docs-consistency guard — the prose catalogue must quote the real thresholds.

``ann_router/backends.yaml`` is the human-facing half of the router: it is what
``ann-router capabilities`` prints and what the docs render. The numbers in it
are copies of constants that live in :mod:`ann_router.policy`, and a copy drifts.
This one did: the ``exact`` entry promised ``n < 10k`` for a year while the
router crossed over at :data:`~ann_router.policy.EXACT_MAX_N` = 1000, so every
reader of the CLI was told a figure ten times too large.

Prose cannot be generated from the constant without making the catalogue
unreadable, so it is checked instead: whatever corpus size the entry quotes has
to be the one the decision tree actually uses.

Author: Warith Harchaoui <warithmetics@deraison.ai>
"""

from __future__ import annotations

import re

from ann_router.config import backend_catalog
from ann_router.policy import EXACT_MAX_N, LATENCY_REFERENCE_MS


def _entry(name: str) -> dict:
    """Return the catalogue entry for ``name``.

    ``backend_catalog()`` hands back the YAML list verbatim, so the lookup is by
    scan rather than by key.
    """
    for entry in backend_catalog():
        if entry.get("name") == name:
            return entry
    raise AssertionError(f"no '{name}' entry in backends.yaml")


def test_exact_entry_quotes_the_real_crossover() -> None:
    """The `exact` entry's corpus size is EXACT_MAX_N, not a rounded stand-in."""
    when = _entry("exact")["when"]

    # Any run of digits, with or without separators: 1000, 1_000, 1,000, 10k.
    quoted = re.findall(r"\b(\d[\d_,]*)\s*(k\b)?", when, flags=re.IGNORECASE)
    sizes = {int(n.replace("_", "").replace(",", "")) * (1000 if k else 1) for n, k in quoted}

    assert EXACT_MAX_N in sizes, (
        f"backends.yaml says {sorted(sizes)} for exact's corpus ceiling, "
        f"but policy.EXACT_MAX_N is {EXACT_MAX_N}. Update the prose, not this test."
    )


def test_exact_entry_quotes_the_reference_budget() -> None:
    """The crossover is only meaningful next to the budget it was measured at."""
    when = _entry("exact")["when"]
    budget = f"{LATENCY_REFERENCE_MS:g} ms"

    assert budget in when, (
        f"backends.yaml does not name the reference budget ({budget}); without it "
        "the quoted corpus size reads as absolute when it scales with latency."
    )
