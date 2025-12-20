"""CLI entrypoint for hardsector_tool."""

from __future__ import annotations

import sys

from .diskdump import main


if __name__ == "__main__":
    sys.exit(main())
