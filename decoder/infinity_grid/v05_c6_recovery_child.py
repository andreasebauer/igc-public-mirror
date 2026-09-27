from __future__ import annotations

"""Fixed C6 controller-child entrypoint. Direct invocation without supervisor attestation fails."""

import os
import sys
import sysconfig
from pathlib import Path

sys.dont_write_bytecode = True


def _read_secret() -> str:
    raw_fd = os.environ.get("IG_C6_SECRET_FD")
    if raw_fd is None or not raw_fd.isdigit():
        raise RuntimeError("REJECT_EXTERNAL_EXECUTION_ORIGIN:C6_SUPERVISOR_ATTESTATION")
    fd = int(raw_fd)
    try:
        raw = os.read(fd, 4096)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    secret = raw.decode("ascii", "strict").strip()
    if len(secret) != 64 or any(c not in "0123456789abcdef" for c in secret):
        raise RuntimeError("REJECT_EXTERNAL_EXECUTION_ORIGIN:C6_SUPERVISOR_ATTESTATION")
    return secret


def main() -> int:
    if len(sys.argv) != 1:
        raise RuntimeError("REJECT_RECOVERY_PAYLOAD")
    secret = _read_secret()
    source = Path(os.environ.get("IG_C6_SOURCE_ROOT", "")).resolve(strict=True)
    runtime = Path(os.environ.get("IG_C6_RUNTIME_ROOT", "")).resolve()
    # The fixed child path must itself belong to the source it is about to execute.
    if Path(__file__).resolve().parent.parent != source:
        raise RuntimeError("REJECT_EXTERNAL_EXECUTION_ORIGIN:C6_SOURCE_BINDING")
    fixed_libs = [
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
    sys.path[:] = [str(source), str(_stdlib), str(_platstdlib),
                   str(_stdlib / "lib-dynload"), str(_platstdlib / "lib-dynload"),
                   str(_venv_site)] + fixed_libs
    from infinity_grid.v05_controller_event_loop import controller_child_main
    return int(controller_child_main(runtime, source, supervisor_pid=os.getppid(), supervisor_secret=secret))


if __name__ == "__main__":
    raise SystemExit(main())
