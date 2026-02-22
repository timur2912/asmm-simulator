#!/usr/bin/env python
"""Run a single experiment from a config file."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mm_sim.__main__ import main

if __name__ == "__main__":
    main(["run"] + sys.argv[1:])
