# Repository Guidelines

## Project Structure & Module Organization
- Keep code in `src/` and tests/fixtures in `tests/`. Flux captures: `tests/ACMS80217` and `tests/ACMS80221` (each has PNG references and a `*-HS32.scp`).
- Prefer small, documented fixtures; if you must add large captures, explain origin/size in a `README` inside the fixture folder.
- One-off tooling belongs in `scripts/` and should be named for intent (e.g., `inspect_flux.py`).

## Build, Test, and Development Commands
- Use Python 3.11+ with a local venv:  
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -e .[dev]
  ```
- Run checks locally before pushing: `pytest`, `ruff check .`, and `black .`.
- When experimenting with new decoders, add a thin CLI in `scripts/` and run it against `tests/ACMS80217/ACMS80217-HS32.scp` to keep behavior reproducible.

## Coding Style & Naming Conventions
- Format with Black (88 cols) and lint with Ruff; fix or consciously silence warnings.
- Use type hints and docstrings on public functions; keep modules and files in `snake_case`, constants in `UPPER_SNAKE_CASE`, classes in `PascalCase`, and functions/vars in `lower_snake_case`.
- Prefer pure functions for parsers; isolate I/O at the edges so logic can be unit-tested.

## Testing Guidelines
- Write `pytest` tests under `tests/` with files named `test_*.py`; group fixtures in `conftest.py` when shared.
- Cover both bit-level parsing and high-level interpretation; include at least one assertion that exercises the sample flux files.
- Target meaningful coverage (≈80% for new modules) and document any intentionally untested paths (e.g., Greaseweazle hardware hooks).
- Use markers to separate slow/fixture-heavy tests (e.g., `@pytest.mark.slow`) and keep default runs fast.

## Commit & Pull Request Guidelines
- Commit messages should be imperative and scoped (e.g., `Add track decoder`, `Harden sector timing parser`); keep subject lines under 72 chars.
- In PRs, include: a short summary, linked issues (if any), test commands/output, and before/after notes when parser output changes.
- Call out new fixtures or large binaries explicitly; avoid committing assets over ~10–20 MB without prior discussion (disk images here sit below GitHub’s 50 MB ceiling).

## Data & Safety Notes
- Verify any shared flux images are safe to redistribute; strip personal metadata from captures and prefer checksums for integrity notes.
- When unsure about format details, document assumptions inline and in PRs so future contributors can validate against additional disks.

## Workflow Tips
- `scripts/inspect_flux.py` is the main entry point. Useful combos:
  - Quick header + hard-sector summary: `python scripts/inspect_flux.py <scp> --track 0 --hard-sector-summary --flux-deltas`.
  - Mark sweeps across holes: add `--scan-bit-patterns --bruteforce-marks --merge-hole-pairs --fixed-spacing-scan`.
  - Whole-rotation experiments: `--stitch-rotation --stitch-gap-comp` plus `--score-grid` to rank FM/MFM hypotheses.
