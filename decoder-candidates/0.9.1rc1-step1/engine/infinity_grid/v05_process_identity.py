from __future__ import annotations

"""Fail-closed Linux process identity across nested PID/proc namespaces.

``os.getpid()`` and ``os.getppid()`` report IDs in the caller's PID namespace.
The mounted procfs may belong to an ancestor namespace.  This module binds both
views without accepting a caller-selected proc root or identity override.
"""

import os
from pathlib import Path
from typing import Any


class ProcessIdentityError(RuntimeError):
    pass


def _strict_int(raw: str, label: str, *, allow_zero: bool = False) -> int:
    try:
        value = int(raw, 10)
    except (TypeError, ValueError) as exc:
        raise ProcessIdentityError("PROC_IDENTITY_" + label) from exc
    if value < 0 or (value == 0 and not allow_zero):
        raise ProcessIdentityError("PROC_IDENTITY_" + label)
    return value


def _parse_status(text: str) -> dict[str, Any]:
    wanted = {"Pid", "PPid", "NSpid"}
    found: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key in wanted:
            if key in found:
                raise ProcessIdentityError("PROC_IDENTITY_STATUS_DUPLICATE:" + key)
            found[key] = value.strip()
    if set(found) != wanted:
        raise ProcessIdentityError("PROC_IDENTITY_STATUS_FIELDS")
    proc_pid = _strict_int(found["Pid"], "PID")
    proc_ppid = _strict_int(found["PPid"], "PPID", allow_zero=True)
    words = found["NSpid"].split()
    if not words:
        raise ProcessIdentityError("PROC_IDENTITY_NSPID")
    nspid = tuple(_strict_int(word, "NSPID") for word in words)
    if nspid[0] != proc_pid:
        raise ProcessIdentityError("PROC_IDENTITY_NSPID_OUTER")
    return {"proc_pid": proc_pid, "proc_ppid": proc_ppid, "nspid": nspid}


def _parse_stat_start_time(text: str) -> int:
    end = text.rfind(")")
    if end < 0 or end + 2 > len(text):
        raise ProcessIdentityError("PROC_IDENTITY_STAT_SHAPE")
    tail = text[end + 2 :].split()
    # tail[0] is field 3 (state); field 22 (starttime) is therefore index 19.
    if len(tail) <= 19:
        raise ProcessIdentityError("PROC_IDENTITY_STAT_FIELDS")
    return _strict_int(tail[19], "START_TIME")


def _read_status(path: Path) -> dict[str, Any]:
    try:
        return _parse_status(path.read_text(encoding="utf-8"))
    except ProcessIdentityError:
        raise
    except Exception as exc:
        raise ProcessIdentityError("PROC_IDENTITY_STATUS_READ") from exc


def _read_start_time(path: Path) -> int:
    try:
        return _parse_stat_start_time(path.read_text(encoding="utf-8"))
    except ProcessIdentityError:
        raise
    except Exception as exc:
        raise ProcessIdentityError("PROC_IDENTITY_STAT_READ") from exc


def process_identity(proc_pid: int) -> dict[str, Any]:
    """Read one process by its ID in the mounted procfs namespace."""
    if type(proc_pid) is not int or proc_pid <= 1:
        raise ProcessIdentityError("PROC_IDENTITY_LOOKUP_PID")
    root = Path("/proc") / str(proc_pid)
    status = _read_status(root / "status")
    if status["proc_pid"] != proc_pid:
        raise ProcessIdentityError("PROC_IDENTITY_LOOKUP_MISMATCH")
    status["start_time_ticks"] = _read_start_time(root / "stat")
    return status


def current_process_identity() -> dict[str, Any]:
    """Bind the caller's namespace-local PID to its mounted-procfs identity."""
    local_pid = os.getpid()
    if type(local_pid) is not int or local_pid <= 1:
        raise ProcessIdentityError("PROC_IDENTITY_LOCAL_PID")
    status = _read_status(Path("/proc/self/status"))
    if status["nspid"][-1] != local_pid:
        raise ProcessIdentityError("PROC_IDENTITY_LOCAL_MAPPING")
    status["local_pid"] = local_pid
    status["start_time_ticks"] = _read_start_time(
        Path("/proc") / str(status["proc_pid"]) / "stat"
    )
    return status


__all__ = [
    "ProcessIdentityError",
    "current_process_identity",
    "process_identity",
]
