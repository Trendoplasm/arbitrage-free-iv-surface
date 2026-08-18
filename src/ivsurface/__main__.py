"""Entry point for ``python -m ivsurface``."""

from __future__ import annotations

import sys

from ivsurface.cli import main

if __name__ == "__main__":
    sys.exit(main())
