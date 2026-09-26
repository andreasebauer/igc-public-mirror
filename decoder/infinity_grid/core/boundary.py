from __future__ import annotations

import hashlib
from typing import Any


def destination_tuple(t: Any) -> tuple:
    return t if isinstance(t, tuple) else (t,)


def canonical_boundary_record(r: Any) -> tuple:
    """Canonical v0.26 boundary-record representation used by Scout/Conveyor.

    This is a representation-preserving extraction, not a new quotient or a
    change to the finite L2 canonical interface theorem.
    """
    return (
        int(r[0]),
        tuple(sorted(tuple(map(int, p)) for p in r[1])),
        int(r[2]),
        tuple(sorted(map(int, destination_tuple(r[3])))),
        int(r[4]),
    )


def record_sha256(r: Any) -> str:
    return hashlib.sha256(repr(r).encode()).hexdigest()


def bridge_ports(pa: tuple[int, int], pb: tuple[int, int]) -> bool:
    Pa, Ma = pa
    Pb, Mb = pb
    return (Ma == 0 or (Ma & Pb) != 0) and (Mb == 0 or (Mb & Pa) != 0)


def compose_boundary_records(a: Any, sa: int, b: Any, sb: int):
    if not bridge_ports(a[1][sa], b[1][sb]):
        return None
    bd = tuple(v for j, v in enumerate(a[1]) if j != sa) + tuple(v for j, v in enumerate(b[1]) if j != sb)
    return canonical_boundary_record((a[0] | b[0], bd, min(a[2], b[2]), destination_tuple(a[3]) + destination_tuple(b[3]), int(a[4] and b[4])))
