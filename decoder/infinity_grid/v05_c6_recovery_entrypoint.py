from __future__ import annotations

"""Zero-payload C6 recovery supervisor entrypoint."""

import sys
import sysconfig
from pathlib import Path

sys.dont_write_bytecode = True
_SOURCE = Path(__file__).resolve().parent.parent
FIXED_LIBS = [
    "/opt/pyvenv-overrides",
    "/opt/pyvenv-libs",
    "/opt/pyvenv/lib/python3.13/site-packages",
]
# Derive standard-library paths from this isolated interpreter, never from
# PYTHONPATH or a user site. Keep -I -S and source/attestation checks intact.
_stdlib = Path(sysconfig.get_path("stdlib")).resolve()
_platstdlib = Path(sysconfig.get_path("platstdlib")).resolve()
_version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
_venv_site = Path(sys.executable).absolute().parent.parent / "lib" / _version_dir / "site-packages"
sys.path[:] = [str(_SOURCE), str(_stdlib), str(_platstdlib),
               str(_stdlib / "lib-dynload"), str(_platstdlib / "lib-dynload"),
               str(_venv_site)] + FIXED_LIBS

from infinity_grid.v05_c6_recovery import supervisor_main

if __name__ == "__main__":
    raise SystemExit(supervisor_main())
