#!/usr/bin/env python
"""Generate plots from saved experiment results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mm_sim.__main__ import main

if __name__ == "__main__":
    main(["reproduce", "--record-trajectories"] + sys.argv[1:])
