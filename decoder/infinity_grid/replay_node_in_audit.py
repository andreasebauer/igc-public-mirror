from __future__ import annotations

"""Bounded fresh NODE_IN mature-boundary calculation and historical comparison."""

import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from .core.boundary import bridge_ports, canonical_boundary_record, compose_boundary_records, record_sha256

PORTS = ((0, 0), (1, 0), (1, 4), (1, 24), (4, 0), (8, 0), (10, 0))
INPUT_HASHES = {
    "NODE_IN_ORIGINAL_OBSERVED_S15_RECORDS.json.gz": "ab397b2d40891a6f3cec742edbfd25c5eb1b5d627a98641a9666f8a265862f5f",
    "NODE_IN_ORIGINAL_FROZEN_PRIMITIVES.json": "8abbccb9578cfe92367482f19164738165e6ca072253cf8c311102cd397142e7",
    "NODE_IN_ORIGINAL_NODE_GRADUATION_RESULT.json": "d791ed9b46d5d7e550726fc67bec9070a52597b4ce35ae7dda37f12f0b27a7d3",
}


def audit(resource_root: Path) -> dict[str, Any]:
    root = Path(resource_root)
    for name, expected in INPUT_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"NODE_IN input hash mismatch: {name}")
    raw = json.loads(gzip.decompress((root / "NODE_IN_ORIGINAL_OBSERVED_S15_RECORDS.json.gz").read_bytes()))
    primitive = json.loads((root / "NODE_IN_ORIGINAL_FROZEN_PRIMITIVES.json").read_text())
    old = json.loads((root / "NODE_IN_ORIGINAL_NODE_GRADUATION_RESULT.json").read_text())
    if raw.get("count") != 22885 or len(raw.get("records", [])) != 22885 or primitive.get("count") != 9:
        raise ValueError("NODE_IN input cardinality changed")
    def parse(row):
        r = canonical_boundary_record(ast.literal_eval(row["record"]))
        if record_sha256(r) != row["sha256"]:
            raise ValueError("NODE_IN record identity mismatch")
        return r
    prim = [parse(row) for row in primitive["primitive"]]
    records = [parse(row) for row in raw["records"]]
    port_atoms = tuple(sorted({p for r in prim for p in r[1]}))
    t_atoms = tuple(sorted({t for r in prim for t in r[3]}))
    if port_atoms != PORTS or len(t_atoms) != 9:
        raise ValueError("NODE_IN primitive alphabet changed")
    roles: dict[str, tuple[str, tuple]] = {}
    coordinates: set[tuple] = set()
    failures: list[str] = []
    for r in records:
        h = record_sha256(r)
        role = hashlib.sha256(repr((r[0], r[2], r[4], tuple(sorted(set(r[1]))))).encode()).hexdigest()[:20]
        if role not in roles or h < roles[role][0]:
            roles[role] = (h, r)
        pc = tuple(Counter(r[1])[p] for p in PORTS)
        tc = tuple(Counter(r[3])[t] for t in t_atoms)
        if (pc, tc) in coordinates or (r[0], r[2], r[4]) != (15, 1, 0) or len(r[1]) != len(r[3]) + 2 or any(p not in PORTS for p in r[1]) or any(t not in t_atoms for t in r[3]):
            failures.append(h)
        coordinates.add((pc, tc))
    if failures or len(roles) != 64 or any(len(r[1]) != 3 or len(r[3]) != 1 for r in prim):
        raise ValueError(f"NODE_IN representation failures: {failures[:5]}")
    panel = sorted(roles.values(), key=lambda pair: pair[0])
    compat = [(a, b) for a in PORTS for b in PORTS if bridge_ports(a, b)]
    cases = 0
    mismatch = []
    closure = []
    for _, a in panel:
        ca = Counter(a[1]); ta = Counter(a[3])
        for _, b in panel:
            cb = Counter(b[1]); tb = Counter(b[3])
            for pa, pb in compat:
                if not ca[pa] or not cb[pb]:
                    continue
                ia = a[1].index(pa); ib = b[1].index(pb)
                exact = compose_boundary_records(a, ia, b, ib)
                if exact is None:
                    raise ValueError("NODE_IN bridge inconsistency")
                ports = (ca + cb)
                ports[pa] -= 1; ports[pb] -= 1
                tokens = ta + tb
                projected = canonical_boundary_record((15, tuple(p for p in PORTS for _ in range(ports[p])), 1, tuple(t for t in t_atoms for _ in range(tokens[t])), 0))
                cases += 1
                if exact != projected:
                    mismatch.append([record_sha256(a), record_sha256(b), pa, pb])
                if (exact[0], exact[2], exact[4]) != (15, 1, 0) or len(exact[1]) != len(exact[3]) + 2 or any(p not in PORTS for p in exact[1]) or any(t not in t_atoms for t in exact[3]):
                    closure.append([record_sha256(a), record_sha256(b), pa, pb])
    fresh = {"observed_records": len(records), "primitive_count": len(prim), "role_count": len(roles), "panel_states": len(panel), "ordered_bridge_pairs": len(compat), "regression_exact_cases": cases, "representation_failures": failures, "counter_vs_exact_mismatches": mismatch, "closure_failures": closure}
    historical = {"observed_records": old["NG0_integrity"]["observed_exact_S15"], "primitive_count": old["NG0_integrity"]["primitive_count"], "role_count": old["NG0_integrity"]["observed_role_count"], "panel_states": old["NG3_exact_mature_macro_closure"]["regression_panel_states"], "ordered_bridge_pairs": old["NG4_finite_control"]["ordered_bridge_type_pairs"], "regression_exact_cases": old["NG3_exact_mature_macro_closure"]["regression_exact_cases"], "representation_failures": old["NG1_exact_representation"]["reconstruction_failures"], "counter_vs_exact_mismatches": old["NG3_exact_mature_macro_closure"]["counter_vs_exact_mismatches"], "closure_failures": old["NG3_exact_mature_macro_closure"]["closure_failures"]}
    return {"schema_id": "IG_NODE_IN_FRESH_AUDIT_V1", "new_library": "0.8lib/dev58", "fresh": fresh, "historical": historical, "matches_historical": fresh == historical, "historical_science_sha256": old["science_sha256"], "input_sha256": INPUT_HASHES}
