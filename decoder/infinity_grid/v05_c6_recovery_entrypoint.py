from __future__ import annotations

"""Zero-payload C6 recovery supervisor entrypoint."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
_SOURCE = Path(__file__).resolve().parent.parent
# Import only source-bound bootstrap code before admitting runtime packages.
sys.path.insert(0, str(_SOURCE))
from infinity_grid.isolated_runtime import configure_isolated_paths
configure_isolated_paths(_SOURCE)

from infinity_grid.v05_c6_recovery import supervisor_main

if __name__ == "__main__":
    raise SystemExit(supervisor_main())
