# Some Coding Standards

A practical coding standard for the *AI Helpers* suite, adapted for
`ann-router`. Code here should be readable, maintainable, testable, and easy for
downstream users to adopt — every project should read like a **teaching
artefact**, not a private side project. The normative source is
[`os-helper/CODING.md`](https://github.com/warith-harchaoui/os-helper); this file
mirrors it and adds the ANN-router-specific notes.

## 0. Scope: all functions, all languages, no exceptions
The standards apply to every function/method/class — public, private (`_helper`),
dunder, nested, lambdas — and to every language in the repo. No exemption based
on naming convention or visibility.

## 1. Use Numpy-style docstrings for every function and class
Sections in order: short summary, optional extended summary, `Parameters`,
`Returns`/`Yields`, `Raises`, `Examples`, `Notes`. Private helpers follow the
same rule (Examples may be shortened if trivial).

## 2. Add a module-level docstring to every `.py` file
Explain what the module does, why it exists, what it consumes and produces, and
end with an `Author:` line. For ann-router, say **which part of the ANN problem**
the module owns (a backend adapter, the policy, the router, a surface).

## 3. Use full typing
Annotate every signature, class attribute and module constant. Prefer
`from __future__ import annotations`, PEP-604 unions, and `dataclasses`/`Protocol`
/`Literal` for structured data (see `spec.py`'s `Criteria`).

## 4. Comment generously — everywhere, in every function
Target ≈ 1 comment line per 3-4 lines of code (≈25-30% density), docstrings
excluded; floor 1 per 6 lines (≈15%). Measure with `cloc --by-file`. Comments
must document **the problem** (ANN engine selection, quantisation trade-offs, the
id-mapping a backend needs), never parrot trivial code.

## 5. Include an `EXAMPLES.md` at the repository root
A self-contained, runnable cookbook in English, linked from the README.

## 6. Avoid bare `print(...)` in library and script code
Use `os_helper` logging (`osh.info` / `osh.error`) or `logging.getLogger`.
Exception: docs, README, EXAMPLES, docstrings. The CLIs log to **stderr** and
emit data (JSON) to **stdout** so output stays pipeable.

## 7. Document expected output after `print(...)` in examples
e.g. `print(choice.backend)  # => turbovec`.

## 8. Provide example config files when configuration is involved
Ship commented YAML (`policy.yaml`, `backends.yaml`, `hardware.yaml`) explaining
every field and its rationale. Prefer YAML over JSON because YAML supports
comments.

## 9. Gitignore real config files, but keep examples/packaged config tracked
The packaged `ann_router/*.yaml` stays tracked; user-local overrides pointed at
by `ANN_ROUTER_POLICY` do not.

## 10. Add a Homebrew hint after every `brew install` mention
Follow it with the brew.sh install hint. (No `brew` is required for ann-router
today.)

## 11. Keep acknowledgements optional and project-specific
Neutral EN/FR forms; avoid hard-coding personal names in public templates.

## 12. Provide cross-platform install instructions
Cover macOS / Ubuntu / Windows; never silently omit a platform. Call out the
**Apple Silicon annoy** caveat and the **Linux-only ScaNN** reality explicitly
(see `INSTALL.md`).

## 13. Keep AI-assistant attribution policy explicit
Do **not** list any AI assistant as author or co-author; no `Co-Authored-By`
trailers. Attribute humans only.

## 14. Use `pytest` and require CI to pass
`tests/` + pytest, CI on every push/PR. Cover every function/class at least once
via functional/scenario tests. Deterministic: **seed randomness** (the recall
fixtures do), mock network/servers, skip uninstallable backends with a reason
rather than failing. Rationalise the suite at the ~100-test mark toward
end-to-end tests.

## 15. Add AI evaluation when the project uses AI / measured quality
ann-router's "AI eval" is its **recall contract**: `tests/conftest.py` pins
per-backend recall@k **floors as versioned constants** (`RECALL_THRESHOLDS`),
measured against the exact brute-force ground truth on a fixed synthetic corpus.
The policy thresholds are likewise versioned (`POLICY_VERSION`). Bumping a floor
or a threshold is a deliberate, reviewed change with a CHANGELOG entry — not a
"vibe check". Backends whose dependency is absent are skipped, never faked.

## How to apply these standards

**When editing an existing `.py` file** — keep docstrings/typing/comments in sync
with the change; re-run `ruff check` and `pytest`.

**When creating a new `.py` file** — start with `from __future__ import
annotations`, write the module docstring first, comment from the first draft
(never "I'll add comments later"), add tests.

**When making a docs-only or style-only release** — patch bump + `CHANGELOG.md`.

## Core principle
A good repository should be understandable by a new reader, testable by a
contributor, reproducible in CI, and usable without private context.
Documentation, examples, typing, tests, and evaluation are not extras.
