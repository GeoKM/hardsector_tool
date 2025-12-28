from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def repo_src_path() -> Path:
    """Return the repository's ``src`` directory."""

    return Path(__file__).resolve().parents[1] / "src"


def _ensure_repo_on_path() -> None:
    if importlib.util.find_spec("hardsector_tool") is None:
        sys.path.insert(0, str(repo_src_path()))


_ensure_repo_on_path()


def require_fixture(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"fixture not available: {path}", allow_module_level=True)
    return path
