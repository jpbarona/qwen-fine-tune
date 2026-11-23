"""Allow `python -m rlvr`."""

from __future__ import annotations

import sys

from rlvr.app.cli import main

if __name__ == "__main__":
    sys.exit(main())
