# Repository Guidelines

## Project Structure & Module Organization
- Keep code in a `src/` package (add it if missing) and reserve `tests/` for automated checks; sample flux artifacts currently live in `tests/ACMS80217` (two reference PNGs and `ACMS80217-HS32.scp`).
- Prefer small, documented fixtures; if you must add large captures, explain the origin and size in a `README` inside the fixture folder.
- Place one-off scripts in `scripts/` and name them after their intent (e.g., `scripts/inspect_flux.py`).

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
- Cover both bit-level parsing and high-level interpretation; include at least one assertion that exercises the sample flux file.
- Target meaningful coverage (≈80% for new modules) and document any intentionally untested paths (e.g., Greaseweazle hardware hooks).
- Use markers to separate slow/fixture-heavy tests (e.g., `@pytest.mark.slow`) and keep default runs fast.

## Commit & Pull Request Guidelines
- Commit messages should be imperative and scoped (e.g., `Add track decoder`, `Harden sector timing parser`); keep subject lines under 72 chars.
- In PRs, include: a short summary, linked issues (if any), test commands/output, and before/after notes when parser output changes.
- Call out new fixtures or large binaries explicitly; avoid committing assets over ~10–20 MB without prior discussion.

## Data & Safety Notes
- Verify any shared flux images are safe to redistribute; strip personal metadata from captures and prefer checksums for integrity notes.
- When unsure about format details, document assumptions inline and in PRs so future contributors can validate against additional disks.
