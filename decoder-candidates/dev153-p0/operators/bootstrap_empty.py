#!/usr/bin/env python3
"""P0 runtime restore compatibility entry; no master run starts here."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1] / 'source/tools/bootstrap_p0_runtime.py'), run_name='__main__')
