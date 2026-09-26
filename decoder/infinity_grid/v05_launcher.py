from __future__ import annotations

import argparse
import math
import os
import resource
import sys
from pathlib import Path


def _drop(uid: int, gid: int, cpu_seconds: float | None = None) -> None:
    if cpu_seconds:
        lim = max(1, int(math.ceil(float(cpu_seconds))))
        resource.setrlimit(resource.RLIMIT_CPU, (lim, lim))
    os.setgroups([])
    os.setgid(int(gid))
    os.setuid(int(uid))
    os.umask(0o077)


def _run_worker(ns) -> int:
    _drop(ns.uid, ns.gid, ns.cpu_seconds)
    from .v05_worker import worker_main
    try:
        result = worker_main(Path(ns.job))
        import json
        print(json.dumps({"status": "PASS", "job_sha256": result["job_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _run_engineering_worker(ns) -> int:
    _drop(ns.uid, ns.gid, ns.cpu_seconds)
    from .v05_engineering_worker import worker_main
    try:
        result = worker_main(Path(ns.job))
        import json
        print(json.dumps({"status": result["status"], "job_id": result["job_id"]}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _probe_write(ns) -> int:
    _drop(ns.uid, ns.gid, None)
    p = Path(ns.path)
    try:
        p.write_text("forbidden", encoding="utf-8")
    except (PermissionError, OSError):
        return 0
    else:
        p.unlink(missing_ok=True)
        return 9


def main(argv=None) -> int:
    from .v05_route_closure import REJECT_DIRECT_EXECUTION_ROUTE
    print(REJECT_DIRECT_EXECUTION_ROUTE, file=sys.stderr)
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
