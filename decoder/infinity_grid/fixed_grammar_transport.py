from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

from . import regime_scanner as rs
from .semantic_sentinel import verify_native_semantics
from .theorem_registry import verify_theorem


STREAMING_SPEC_RESOURCE = "O_REGIME_STREAMING_DESCRIPTOR_SCANNER_SPEC_v2.json"
STREAMING_O7_SEED_RESOURCE = "O7_STREAMING_SEED_v2.json"
STREAMING_BASELINE_RESOURCE = "O_REGIME_STREAMING_RECIPE_BASELINE_v2.json"
AUTHORITY_PACK_RESOURCE = "O_REGIME_AUTHORITY_PACK_V1.json"


def _cbytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(_cbytes(obj)).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _verify_embedded_hash(obj: dict, field: str, label: str) -> str:
    expected = obj.get(field)
    observed = _sha({k: v for k, v in obj.items() if k != field})
    if expected != observed:
        raise RuntimeError(f"{label} hash mismatch: {expected} != {observed}")
    return observed


def load_streaming_regime_spec() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder", STREAMING_SPEC_RESOURCE)
    d = json.loads(p.read_text(encoding="utf-8"))
    _verify_embedded_hash(d, "spec_sha256", "streaming scanner spec")
    return d


def load_streaming_o7_seed() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder", STREAMING_O7_SEED_RESOURCE)
    d = json.loads(p.read_text(encoding="utf-8"))
    _verify_embedded_hash(d, "seed_sha256", "streaming O7 seed")
    return d


def _zip_json_by_suffix(zf: zipfile.ZipFile, suffix: str) -> dict:
    hits = [n for n in zf.namelist() if n.endswith(suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {suffix} in Phase8 seed, found {len(hits)}")
    return json.loads(zf.read(hits[0]).decode("utf-8"))


def load_authority_pack() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder", AUTHORITY_PACK_RESOURCE)
    d = json.loads(p.read_text(encoding="utf-8"))
    _verify_embedded_hash(d, "authority_pack_sha256", "O-regime authority pack")
    return d

def _verify_phase8_authority(phase8_seed: Path | None, spec: dict) -> dict:
    pack = load_authority_pack()
    exp = spec["authority"]["phase8_seed_sha256"]
    if pack["phase8_seed_sha256"] != exp or pack["normalized_grammar_sha256"] != spec["authority"]["normalized_grammar_sha256"]:
        raise RuntimeError("embedded authority pack does not match frozen streaming specification")
    if pack["theorem_accelerated_calibration_science_sha256"] != spec["authority"]["theorem_accelerated_calibration_science_sha256"]:
        raise RuntimeError("authority pack calibration pin mismatch")
    if phase8_seed is None:
        return {
            "phase8_seed_sha256": exp,
            "phase8_seed_bytes_present": False,
            "authority_source": "EMBEDDED_PINNED_AUTHORITY_PACK",
            "authority_pack_sha256": pack["authority_pack_sha256"],
            "O8_science_sha256": pack["O8"]["science_sha256"],
            "O9_science_sha256": pack["O9"]["science_sha256"],
        }
    phase8_seed = Path(phase8_seed)
    got = _sha_file(phase8_seed)
    if got != exp:
        raise RuntimeError(f"Phase8 seed SHA mismatch: {got} != {exp}")
    with zipfile.ZipFile(phase8_seed) as zf:
        bad = zf.testzip()
        if bad: raise RuntimeError(f"Phase8 seed CRC failure at {bad}")
        o8 = _zip_json_by_suffix(zf, "authority/O8_BP0_RESULT.json")
        o9 = _zip_json_by_suffix(zf, "authority/O9_BP0_RESULT.json")
    grammar = spec["authority"]["normalized_grammar_sha256"]
    g8 = o8.get("normalized_grammar", {}).get("candidate_sha256"); g9 = o9.get("normalized_grammar", {}).get("candidate_sha256")
    if g8 != grammar or g9 != grammar: raise RuntimeError(f"Phase8 O8/O9 grammar authority mismatch: O8={g8}, O9={g9}, expected={grammar}")
    return {"phase8_seed_sha256":got,"phase8_seed_bytes_present":True,"authority_source":"FULL_PHASE8_ARCHIVE","authority_pack_sha256":pack["authority_pack_sha256"],"O8_grammar_sha256":g8,"O9_grammar_sha256":g9}


@dataclass(frozen=True)
class _RecipeState:
    lane: str
    motif_id: str
    top_pairs_t: tuple[tuple[int, int], ...]
    typed_edges_t: tuple[tuple[int, int, int, int], ...]
    owner_caps_t: tuple[tuple[int, ...], ...]
    construction_digest: str
    skin: str
    leaf_count: int = 1000
    relation_count_total: int = 100

    @property
    def owner_caps(self):
        return list(self.owner_caps_t)

    @property
    def total_caps(self):
        return tuple(sum(c[t] for c in self.owner_caps_t) for t in range(7))

    @property
    def top_pairs(self):
        return list(self.top_pairs_t)

    @property
    def typed_edges(self):
        return list(self.typed_edges_t)



def _typed_recipe_state(lane: str, motif_id: str, n: int, edges: list[tuple[int, int]], pairs: list[tuple[int, int]], schedule_seed: int, force_pair: tuple[int, int] | None = None) -> _RecipeState:
    caps = [[100] * 7 for _ in range(n)]
    typed: list[tuple[int, int, int, int]] = []
    edges = sorted(tuple(sorted(map(int, e))) for e in edges)
    for idx, (u, v) in enumerate(edges):
        a, b = force_pair if force_pair is not None else pairs[(idx + schedule_seed) % len(pairs)]
        a, b = int(a), int(b)
        caps[u][a] -= 1
        caps[v][b] -= 1
        if caps[u][a] <= 0 or caps[v][b] <= 0:
            raise RuntimeError(f"streaming recipe dummy support exhausted for {lane}|{motif_id}")
        typed.append((u, v, a, b))
    digest = _sha({"lane": lane, "motif_id": motif_id, "typed_edges": typed})
    # Deliberately unique dummy skins: resource-skin population identity is not a
    # v2 promotion observable. The authority-checked lanes below depend only on
    # top graph + endpoint-support pattern.
    skin = _sha({"dummy_resource_class": f"{lane}|{motif_id}"})
    return _RecipeState(lane, motif_id, tuple(edges), tuple(typed), tuple(tuple(x) for x in caps), digest, skin)


@lru_cache(maxsize=1)
def recompute_streaming_recipe_baseline() -> dict:
    spec = load_streaming_regime_spec()
    verify_native_semantics(raise_on_change=True)
    verify_theorem("O_GENERIC_FIXED_GRAMMAR_FACTORISATION_V1", raise_on_stale=True)
    seed = load_streaming_o7_seed()
    pairs = [tuple(map(int, x)) for x in seed["bridge_pairs"]]
    motifs = rs.load_motif_library()
    pool: list[_RecipeState] = []
    recipe_rows: list[dict] = []
    for mi, m in enumerate(motifs):
        n = int(m["n"])
        edges = [tuple(map(int, e)) for e in m["edges"]]
        definitions = [("HOM", f"HOM:{n}:{mi}", mi % len(pairs))]
        if n in (4, 5) or (n == 6 and mi % 6 == 0):
            definitions.append(("HET", f"HET:{n}:{mi}", (mi * 3 + 1) % len(pairs)))
        if n == 4:
            definitions.append(("MIX", f"MIX:4:{mi}", (mi * 5 + 2) % len(pairs)))
        for lane, motif_id, schedule_seed in definitions:
            st = _typed_recipe_state(lane, motif_id, n, edges, pairs, schedule_seed)
            pool.append(st)
            recipe_rows.append({
                "recipe_id": f"{lane}|{motif_id}",
                "owners": n,
                "edges": len(edges),
                "typed_edges": [list(x) for x in st.typed_edges_t],
            })
    for label, edges0 in (("A", rs.GA), ("B", rs.GB)):
        st = _typed_recipe_state("TWIN", f"TWIN:{label}", 6, [tuple(e) for e in edges0], pairs, 0, force_pair=(0, 0))
        pool.append(st)
        recipe_rows.append({
            "recipe_id": f"TWIN|TWIN:{label}",
            "owners": 6,
            "edges": len(edges0),
            "typed_edges": [list(x) for x in st.typed_edges_t],
        })
    expected_count = int(spec["authority"]["full_pool_recipe_count"])
    if len(pool) != expected_count:
        raise RuntimeError(f"streaming recipe count mismatch: {len(pool)} != {expected_count}")
    scan, _ = rs._scan_level(pool, 11, pairs, spec["authority"]["normalized_grammar_sha256"])
    lane_hashes = {lane: _sha(scan["normalized_signature"][lane]) for lane in ("branching", "symmetry", "lineage", "topology_services")}
    expected = spec["authority"]["full_pool_lane_hashes"]
    if lane_hashes != expected:
        raise RuntimeError(f"streaming full-pool lane authority mismatch: {lane_hashes} != {expected}")
    static = {
        "schema": "IG_O_REGIME_STREAMING_STATIC_RECIPE_BASELINE_V2",
        "recipe_count": len(pool),
        "recipe_catalogue_sha256": _sha(recipe_rows),
        "lane_hashes": lane_hashes,
        "normalized_lane_payloads": {
            "branching": scan["normalized_signature"]["branching"],
            "symmetry": scan["normalized_signature"]["symmetry"],
            "lineage": scan["normalized_signature"]["lineage"],
            "topology_services": scan["normalized_signature"]["topology_services"],
        },
        "diversity": {
            "topology_classes": scan["diversity"]["topology_classes"],
            "organizational_classes": scan["diversity"]["organizational_classes"],
            "owner_count_hist": scan["diversity"]["owner_count_hist"],
            "edge_count_hist": scan["diversity"]["edge_count_hist"],
        },
        "overlap_gluing": {
            k: scan["overlap_gluing"][k]
            for k in ("shared_fiber_pairs", "strict_inclusions", "intersection_realized_fraction", "union_realized_fraction")
        },
        "obstruction_relief": scan["obstruction_relief"],
        "quotient_observer": {
            "same_resource_topology_hidden_witness": "TWIN_A_TWIN_B_INHERITED_EXACT_AUTHORITY",
            "resource_population_materialized": False,
            "note": "v2 does not use exact resource-skin population counts for promotion; the exact twin/topology separation authority is inherited separately.",
        },
        "authority_match": {
            "O12_O13_raise_audit_science_sha256": spec["authority"]["o12_o13_raise_audit_science_sha256"],
            "depth_erased_normalized_org_law_science_sha256": spec["authority"]["depth_erased_normalized_org_law_science_sha256"],
        },
    }
    static["static_signature_sha256"] = _sha(static)
    return static



@lru_cache(maxsize=1)
def streaming_recipe_baseline() -> dict:
    spec = load_streaming_regime_spec()
    p = files("infinity_grid").joinpath("resources/decoder", STREAMING_BASELINE_RESOURCE)
    d = json.loads(p.read_text(encoding="utf-8"))
    observed = _science_sha({k: v for k, v in d.items() if k != "resource_sha256"})
    if d.get("resource_sha256") != observed:
        raise RuntimeError(f"streaming recipe baseline resource hash mismatch: {d.get('resource_sha256')} != {observed}")
    if d["resource_sha256"] != spec["authority"]["streaming_recipe_baseline_resource_sha256"]:
        raise RuntimeError("streaming recipe baseline resource does not match frozen spec")
    if d["recipe_count"] != int(spec["authority"]["full_pool_recipe_count"]):
        raise RuntimeError("streaming recipe baseline count mismatch")
    if d["lane_hashes"] != spec["authority"]["full_pool_lane_hashes"]:
        raise RuntimeError("streaming recipe baseline lane hashes do not match earned authority")
    return {k: v for k, v in d.items() if k != "resource_sha256"}

def _verify_earned_laws(spec: dict) -> dict:
    reg = rs.load_earned_regime_laws()
    byid = {x.get("law_id"): x for x in reg.get("laws", []) if x.get("status") == "EARNED"}
    checks = {}
    for key_id, key_sha in (
        ("depth_erased_normalized_org_law_id", "depth_erased_normalized_org_law_science_sha256"),
        ("depth_erased_service_law_id", "depth_erased_service_law_science_sha256"),
    ):
        law_id = spec["authority"][key_id]
        law = byid.get(law_id)
        if not law:
            raise RuntimeError(f"required earned O-regime law missing: {law_id}")
        expected = spec["authority"][key_sha]
        observed = law.get("science_sha256")
        if observed != expected:
            raise RuntimeError(f"earned law science hash mismatch for {law_id}: {observed} != {expected}")
        checks[law_id] = observed
    return {"registry_sha256": reg["registry_sha256"], "laws": checks}


def _stable_jsonish(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _stable_jsonish(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_stable_jsonish(v) for v in obj]
    return obj


def _science_sha(obj: Any) -> str:
    return _sha(_stable_jsonish(obj))


def _science_payload_for_record(obj: dict) -> dict:
    schema = obj.get("schema")
    drop = {"science_sha256", "operational_record_sha256"}
    if schema == "IG_O_REGIME_STREAMING_RUN_STATE_V2": drop |= {"updated_unix"}
    if schema == "IG_O_REGIME_STREAMING_RESULT_V2": drop |= {"date", "levels_computed_this_invocation", "levels_computed_count", "elapsed_seconds"}
    return {k:v for k,v in obj.items() if k not in drop}

def _record_hash_ok(obj: dict, field: str = "science_sha256") -> bool:
    expected = obj.get(field)
    return isinstance(expected, str) and expected == _science_sha(_science_payload_for_record(obj))

def _with_science_hash(obj: dict) -> dict:
    out = dict(obj)
    out["science_sha256"] = _science_sha(_science_payload_for_record(out))
    return out

def _with_operational_hash(obj: dict) -> dict:
    out=dict(obj); out["operational_record_sha256"]=_sha({k:v for k,v in out.items() if k!="operational_record_sha256"}); return out


def _initial_frontier(spec: dict, seed: dict, static: dict, earned: dict) -> dict:
    f = {
        "schema": "IG_O_REGIME_STREAMING_FRONTIER_V2",
        "level": 7,
        "parent_frontier_sha256": None,
        "last_delta_science_sha256": None,
        "spec_sha256": spec["spec_sha256"],
        "phase8_seed_sha256": seed["phase8_seed_sha256"],
        "grammar_sha256": spec["authority"]["normalized_grammar_sha256"],
        "read_set_sha256": spec["contract"]["read_set_sha256"],
        "observer_sha256": spec["contract"]["observer_sha256"],
        "lift_schema_sha256": spec["contract"]["lift_schema_sha256"],
        "earned_law_registry_sha256": earned["registry_sha256"],
        "static_recipe_signature_sha256": static["static_signature_sha256"],
        "recipe_catalogue_sha256": static["recipe_catalogue_sha256"],
        "growth_bounds": {
            "min_total_free_by_type": list(map(int, seed["min_total_free_by_type"])),
            "max_total_free_by_type": list(map(int, seed["max_total_free_by_type"])),
            "leaf_count_min": int(seed["leaf_count_min"]),
            "leaf_count_max": int(seed["leaf_count_max"]),
            "relation_count_min": int(seed["relation_count_min"]),
            "relation_count_max": int(seed["relation_count_max"]),
        },
        "full_support_certified": True,
        "support_certificate_basis": "EXACT_O7_SELECTED_AUTHORITY_SEED",
        "rolling_delta_hashes": [],
    }
    return _with_science_hash(f)


def _advance_frontier(frontier: dict, spec: dict, static: dict, earned: dict) -> tuple[dict, dict]:
    if not _record_hash_ok(frontier):
        raise RuntimeError("parent streaming frontier science hash mismatch")
    for field, expected in (
        ("spec_sha256", spec["spec_sha256"]),
        ("grammar_sha256", spec["authority"]["normalized_grammar_sha256"]),
        ("read_set_sha256", spec["contract"]["read_set_sha256"]),
        ("observer_sha256", spec["contract"]["observer_sha256"]),
        ("lift_schema_sha256", spec["contract"]["lift_schema_sha256"]),
        ("earned_law_registry_sha256", earned["registry_sha256"]),
        ("static_recipe_signature_sha256", static["static_signature_sha256"]),
    ):
        if frontier.get(field) != expected:
            raise RuntimeError(f"streaming contract reopen: {field} changed ({frontier.get(field)} != {expected})")
    level = int(frontier["level"]) + 1
    g = frontier["growth_bounds"]
    mn = list(map(int, g["min_total_free_by_type"]))
    mx = list(map(int, g["max_total_free_by_type"]))
    max_owner = list(map(int, spec["support_induction"]["max_endpoint_consumption_per_owner_by_type"]))
    max_state = list(map(int, spec["support_induction"]["max_endpoint_consumption_per_state_by_type"]))
    owner_margin = [mn[t] - max_owner[t] for t in range(7)]
    if any(x <= 0 for x in owner_margin):
        delta = _with_science_hash({
            "schema": "IG_O_REGIME_STREAMING_LEVEL_DELTA_V2",
            "level": level,
            "parent_frontier_sha256": frontier["science_sha256"],
            "status": "REOPEN_REQUIRED",
            "classification": "ENDPOINT_FULL_SUPPORT_INDUCTION_FAILED",
            "owner_support_margin_by_type": owner_margin,
            "reopen_action": spec["reopen_action"],
        })
        return frontier, delta
    next_min = [4 * mn[t] - max_state[t] for t in range(7)]
    next_max = [6 * mx[t] for t in range(7)]
    if any(x <= 0 for x in next_min):
        raise RuntimeError("support recurrence produced nonpositive next-state lower bound despite positive owner margins")
    gb = {
        "min_total_free_by_type": next_min,
        "max_total_free_by_type": next_max,
        "leaf_count_min": 4 * int(g["leaf_count_min"]),
        "leaf_count_max": 6 * int(g["leaf_count_max"]),
        "relation_count_min": 4 * int(g["relation_count_min"]) + 3,
        "relation_count_max": 6 * int(g["relation_count_max"]) + 15,
    }
    transport_mode = "DIRECT_DESCRIPTOR_FACTOR_EVALUATION" if level < int(spec["streaming"]["theorem_transport_default_from_level"]) else "THEOREM_TRANSPORTED_BY_O_DEPTH_ERASED_NORMALIZED_ORGANIZATION_V1"
    genealogy = {
        "grammar": "PERSISTS",
        "diversity": "PERSISTS_FULL_RECIPE_CENSUS",
        "branching": "PERSISTS",
        "symmetry": "PERSISTS",
        "overlap_gluing": "PERSISTS_IMPLEMENTATION_FACTOR",
        "lineage": "PERSISTS",
        "quotient": "PERSISTS_INHERITED_TWIN_AUTHORITY",
        "topology_services": "PERSISTS",
        "obstruction_relief": "PERSISTS",
        "representation": "PERSISTS",
    }
    organization = {
        "mode": transport_mode,
        "static_signature_sha256": static["static_signature_sha256"],
        "recipe_count": static["recipe_count"],
        "lane_hashes": static["lane_hashes"],
        "diversity": static["diversity"],
        "overlap_gluing": static["overlap_gluing"],
        "obstruction_relief": static["obstruction_relief"],
        "quotient_observer": static["quotient_observer"],
        "genealogy": genealogy,
        "raw_magnitude_excluded_from_novelty": True,
    }
    frontier_payload = {
        "schema": "IG_O_REGIME_STREAMING_FRONTIER_V2",
        "level": level,
        "parent_frontier_sha256": frontier["science_sha256"],
        "spec_sha256": spec["spec_sha256"],
        "phase8_seed_sha256": frontier["phase8_seed_sha256"],
        "grammar_sha256": spec["authority"]["normalized_grammar_sha256"],
        "read_set_sha256": spec["contract"]["read_set_sha256"],
        "observer_sha256": spec["contract"]["observer_sha256"],
        "lift_schema_sha256": spec["contract"]["lift_schema_sha256"],
        "earned_law_registry_sha256": earned["registry_sha256"],
        "static_recipe_signature_sha256": static["static_signature_sha256"],
        "recipe_catalogue_sha256": static["recipe_catalogue_sha256"],
        "growth_bounds": gb,
        "full_support_certified": True,
        "support_certificate_basis": "INDUCTIVE_4_OWNER_LOWER_BOUND_MINUS_FROZEN_MAX_ENDPOINT_CONSUMPTION",
    }
    delta = _with_science_hash({
        "schema": "IG_O_REGIME_STREAMING_LEVEL_DELTA_V2",
        "level": level,
        "parent_frontier_sha256": frontier["science_sha256"],
        "status": "PASS",
        "classification": "FIXED_GRAMMAR_STREAMING_LEVEL_ADVANCED",
        "contract": {
            "grammar_sha256": spec["authority"]["normalized_grammar_sha256"],
            "read_set_sha256": spec["contract"]["read_set_sha256"],
            "observer_sha256": spec["contract"]["observer_sha256"],
            "lift_schema_sha256": spec["contract"]["lift_schema_sha256"],
            "successor_semantics": spec["contract"]["successor_semantics"],
        },
        "support_certificate": {
            "parent_min_total_free_by_type": mn,
            "max_endpoint_consumption_per_owner_by_type": max_owner,
            "owner_support_margin_by_type": owner_margin,
            "next_min_total_free_by_type": next_min,
            "full_seven_type_support_preserved": True,
        },
        "normalized_organization": organization,
        "growth_bounds": gb,
        "frontier_after_payload": frontier_payload,
        "frontier_after_payload_sha256": _science_sha(frontier_payload),
        "earned_laws_honored": sorted(earned["laws"]),
        "reopen_triggered": False,
        "stop": {
            "classification": "UNRESOLVED_WITHIN_REQUESTED_BUDGET_NOT_NEGATIVE",
            "through": level,
            "reason": "No contract/read-set/support/reopen trigger fired. Normalized organization is transported under the earned fixed-recipe depth-erasure law; raw growth alone is not novelty.",
        },
        "nonclaims": spec["forbidden_claims"],
    })
    W = int(spec["streaming"]["rolling_window"])
    persisted_frontier = dict(frontier_payload)
    persisted_frontier["last_delta_science_sha256"] = delta["science_sha256"]
    persisted_frontier["rolling_delta_hashes"] = (list(frontier.get("rolling_delta_hashes", [])) + [delta["science_sha256"]])[-W:]
    persisted_frontier = _with_science_hash(persisted_frontier)
    return persisted_frontier, delta

def _state_record(frontier: dict, spec: dict, status: str = "RUNNING", *, updated_unix: float | None = None) -> dict:
    obj = {
        "schema": "IG_O_REGIME_STREAMING_RUN_STATE_V2",
        "status": status,
        "last_completed_level": int(frontier["level"]),
        "next_level": int(frontier["level"]) + 1,
        "frontier_science_sha256": frontier["science_sha256"],
        "last_delta_science_sha256": frontier.get("last_delta_science_sha256"),
        "spec_sha256": spec["spec_sha256"],
        "updated_unix": float(time.time() if updated_unix is None else updated_unix),
    }
    return _with_operational_hash(_with_science_hash(obj))


def _load_verified_json(path: Path, label: str) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not _record_hash_ok(d):
        raise RuntimeError(f"{label} science hash mismatch: {path}")
    return d


def _delta_path(output: Path, level: int) -> Path:
    return Path(output) / "deltas" / f"O{int(level):05d}.json"


def _snapshot_path(output: Path, level: int) -> Path:
    return Path(output) / "snapshots" / f"O{int(level):05d}_FRONTIER.json"


def _initialize(output: Path, phase8_seed: Path | None, spec: dict, seed: dict, static: dict, earned: dict) -> dict:
    authority = _verify_phase8_authority(phase8_seed, spec)
    if seed["phase8_seed_sha256"] != authority["phase8_seed_sha256"]:
        raise RuntimeError("O7 streaming seed/Phase8 authority mismatch")
    frontier = _initial_frontier(spec, seed, static, earned)
    anchor_delta = _with_science_hash({
        "schema": "IG_O_REGIME_STREAMING_LEVEL_DELTA_V2",
        "level": 7,
        "parent_frontier_sha256": None,
        "status": "PASS",
        "classification": "O7_EXACT_AUTHORITY_STREAMING_ANCHOR",
        "o7_normalized_signature_sha256": seed["o7_normalized_signature_sha256"],
        "o7_selected_state_count": seed["selected_count"],
        "growth_bounds": frontier["growth_bounds"],
        "phase8_authority": authority,
        "stop": {"classification": "ANCHOR_ONLY", "through": 7},
        "nonclaims": spec["forbidden_claims"],
    })
    frontier["last_delta_science_sha256"] = anchor_delta["science_sha256"]
    frontier["rolling_delta_hashes"] = [anchor_delta["science_sha256"]]
    frontier = _with_science_hash({k: v for k, v in frontier.items() if k != "science_sha256"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "deltas").mkdir(exist_ok=True)
    (output / "snapshots").mkdir(exist_ok=True)
    _atomic_json(_delta_path(output, 7), anchor_delta)
    _atomic_json(output / "CURRENT_FRONTIER.json", frontier)
    _atomic_json(_snapshot_path(output, 7), frontier)
    _atomic_json(output / "RUN_STATE.json", _state_record(frontier, spec, "READY"))
    return frontier


def _verify_existing_state(output: Path, spec: dict, static: dict, earned: dict, phase8_seed: Path | None) -> dict:
    authority = _verify_phase8_authority(phase8_seed, spec)
    frontier = _load_verified_json(output / "CURRENT_FRONTIER.json", "streaming frontier")
    state = _load_verified_json(output / "RUN_STATE.json", "streaming run state")
    if state["frontier_science_sha256"] != frontier["science_sha256"]:
        raise RuntimeError("run state/frontier pin mismatch")
    expected_fields = {
        "spec_sha256": spec["spec_sha256"],
        "phase8_seed_sha256": authority["phase8_seed_sha256"],
        "grammar_sha256": spec["authority"]["normalized_grammar_sha256"],
        "read_set_sha256": spec["contract"]["read_set_sha256"],
        "observer_sha256": spec["contract"]["observer_sha256"],
        "lift_schema_sha256": spec["contract"]["lift_schema_sha256"],
        "earned_law_registry_sha256": earned["registry_sha256"],
        "static_recipe_signature_sha256": static["static_signature_sha256"],
    }
    for k, v in expected_fields.items():
        if frontier.get(k) != v:
            raise RuntimeError(f"existing streaming frontier requires reopen: {k} {frontier.get(k)} != {v}")
    last = frontier.get("last_delta_science_sha256")
    if last:
        dp = _delta_path(output, int(frontier["level"]))
        delta = _load_verified_json(dp, "last streaming delta")
        if delta["science_sha256"] != last:
            raise RuntimeError("frontier/last delta pin mismatch")
    return frontier


def _recover_orphan_delta(output: Path, frontier: dict, spec: dict) -> dict:
    # A crash may happen after an immutable delta is written but before the
    # frontier/state replacement. Adopt only an exact next-level delta whose
    # parent pin matches the current frontier. The delta contains a non-circular
    # frontier payload; the persisted frontier is reconstructed deterministically
    # by adding the final delta hash and rolling-window hashes.
    while True:
        next_level = int(frontier["level"]) + 1
        dp = _delta_path(output, next_level)
        if not dp.exists():
            return frontier
        delta = _load_verified_json(dp, "orphan streaming delta")
        if delta.get("parent_frontier_sha256") != frontier["science_sha256"]:
            raise RuntimeError(f"orphan delta O{next_level} parent mismatch; fail closed")
        payload = delta.get("frontier_after_payload")
        if not isinstance(payload, dict) or _science_sha(payload) != delta.get("frontier_after_payload_sha256"):
            raise RuntimeError(f"orphan delta O{next_level} frontier payload mismatch")
        W = int(spec["streaming"]["rolling_window"])
        recovered = dict(payload)
        recovered["last_delta_science_sha256"] = delta["science_sha256"]
        recovered["rolling_delta_hashes"] = (list(frontier.get("rolling_delta_hashes", [])) + [delta["science_sha256"]])[-W:]
        recovered = _with_science_hash(recovered)
        frontier = recovered
        _atomic_json(output / "CURRENT_FRONTIER.json", frontier)
        _atomic_json(output / "RUN_STATE.json", _state_record(frontier, spec, "RECOVERED"))

def _result(output: Path, frontier: dict, spec: dict, static: dict, computed: list[int], elapsed: float, status: str = "PASS", stop: dict | None = None) -> dict:
    if stop is None:
        stop = {
            "classification": "UNRESOLVED_WITHIN_REQUESTED_BUDGET_NOT_NEGATIVE",
            "through": int(frontier["level"]),
            "reason": "No preregistered contract/read-set/support/reopen trigger fired. This is not an eternal-closure claim.",
        }
    r = {
        "schema": "IG_O_REGIME_STREAMING_RESULT_V2",
        "date": "2026-08-30",
        "status": status,
        "classification": "FIXED_GRAMMAR_THEOREM_TRANSPORT",
        "spec_sha256": spec["spec_sha256"],
        "through": int(frontier["level"]),
        "levels_computed_this_invocation": computed,
        "levels_computed_count": len(computed),
        "elapsed_seconds": elapsed,
        "frontier_science_sha256": frontier["science_sha256"],
        "last_delta_science_sha256": frontier.get("last_delta_science_sha256"),
        "static_recipe_baseline": {
            "recipe_count": static["recipe_count"],
            "static_signature_sha256": static["static_signature_sha256"],
            "recipe_catalogue_sha256": static["recipe_catalogue_sha256"],
            "lane_hashes": static["lane_hashes"],
            "diversity": static["diversity"],
        },
        "growth_bounds_at_through": frontier["growth_bounds"],
        "full_support_certified": frontier["full_support_certified"],
        "stop": stop,
        "scientific_interpretation": "Within the frozen GRRL/read-set/observer scope, the complete 193-recipe normalized organizational observer is theorem-transported while exact conservative resource/growth bounds advance. No transported invariance or raw growth is promoted as a raised O-group.",
        "nonclaims": spec["forbidden_claims"],
        "next": "If a reopen trigger fires, stop and run a targeted exact-carrier audit. Otherwise the same streaming frontier may advance directly to any requested deeper budget without historical replay.",
    }
    return _with_operational_hash(_with_science_hash(r))


def run_fixed_grammar_transport(output: Path, through: int = 1000, *, phase8_seed: Path | None = None, snapshot_interval: int | None = None, reset: bool = False) -> dict:
    started = time.perf_counter()
    output = Path(output)
    through = int(through)
    if through < 7:
        raise ValueError("fixed-grammar transport requires through >= 7")
    spec = load_streaming_regime_spec()
    seed = load_streaming_o7_seed()
    if seed["seed_sha256"] != spec["authority"]["o7_streaming_seed_sha256"]:
        raise RuntimeError("streaming O7 seed does not match frozen spec")
    static = streaming_recipe_baseline()
    earned = _verify_earned_laws(spec)
    if reset and output.exists():
        shutil.rmtree(output)
    if (output / "CURRENT_FRONTIER.json").exists():
        frontier = _verify_existing_state(output, spec, static, earned, phase8_seed)
        frontier = _recover_orphan_delta(output, frontier, spec)
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("streaming output exists without CURRENT_FRONTIER.json; use --reset only if intentional")
        frontier = _initialize(output, phase8_seed, spec, seed, static, earned)
    interval = int(snapshot_interval if snapshot_interval is not None else spec["streaming"]["snapshot_interval_default"])
    computed: list[int] = []
    stop = None
    status = "PASS"
    _atomic_json(output / "RUN_STATE.json", _state_record(frontier, spec, "RUNNING"))
    while int(frontier["level"]) < through:
        next_level = int(frontier["level"]) + 1
        dp = _delta_path(output, next_level)
        if dp.exists():
            # Exactly-once: only crash-recovery may adopt an existing next delta.
            frontier = _recover_orphan_delta(output, frontier, spec)
            if int(frontier["level"]) >= next_level:
                continue
            raise RuntimeError(f"unexpected preexisting delta O{next_level}")
        next_frontier, delta = _advance_frontier(frontier, spec, static, earned)
        _atomic_json(dp, delta)
        if delta.get("status") != "PASS":
            stop = {
                "classification": delta.get("classification", "REOPEN_REQUIRED"),
                "through": int(frontier["level"]),
                "trigger_level": next_level,
                "reopen_action": spec["reopen_action"],
            }
            status = "REOPEN_REQUIRED"
            _atomic_json(output / "RUN_STATE.json", _state_record(frontier, spec, "REOPEN_REQUIRED"))
            break
        frontier = next_frontier
        _atomic_json(output / "CURRENT_FRONTIER.json", frontier)
        if interval > 0 and (int(frontier["level"]) % interval == 0 or int(frontier["level"]) == through):
            _atomic_json(_snapshot_path(output, int(frontier["level"])), frontier)
        _atomic_json(output / "RUN_STATE.json", _state_record(frontier, spec, "RUNNING"))
        computed.append(int(frontier["level"]))
    final_state = "REOPEN_REQUIRED" if status != "PASS" else ("COMPLETE" if int(frontier["level"]) >= through else "STOPPED")
    _atomic_json(output / "RUN_STATE.json", _state_record(frontier, spec, final_state))
    result = _result(output, frontier, spec, static, computed, time.perf_counter() - started, status=status, stop=stop)
    _atomic_json(output / "O_REGIME_STREAMING_RESULT.json", result)
    return result


def run_o_regime_streaming_scanner(phase8_seed: Path, output: Path, through: int = 1000, *, snapshot_interval: int | None = None, reset: bool = False) -> dict:
    """Compatibility facade for the historical name; this is theorem transport, not a novelty scanner."""
    return run_fixed_grammar_transport(Path(output), through=through, phase8_seed=Path(phase8_seed), snapshot_interval=snapshot_interval, reset=reset)

def fixed_grammar_transport_status(output: Path) -> dict:
    return o_regime_streaming_status(output)

def o_regime_streaming_status(output: Path) -> dict:
    output = Path(output)
    spec = load_streaming_regime_spec()
    frontier = _load_verified_json(output / "CURRENT_FRONTIER.json", "streaming frontier")
    state = _load_verified_json(output / "RUN_STATE.json", "streaming run state")
    if state["frontier_science_sha256"] != frontier["science_sha256"]:
        raise RuntimeError("streaming status state/frontier mismatch")
    return {
        "schema": "IG_O_REGIME_STREAMING_STATUS_V2",
        "status": state["status"],
        "last_completed_level": state["last_completed_level"],
        "next_level": state["next_level"],
        "frontier_science_sha256": frontier["science_sha256"],
        "last_delta_science_sha256": frontier.get("last_delta_science_sha256"),
        "full_support_certified": frontier["full_support_certified"],
        "spec_sha256": spec["spec_sha256"],
        "growth_bounds": frontier["growth_bounds"],
    }
