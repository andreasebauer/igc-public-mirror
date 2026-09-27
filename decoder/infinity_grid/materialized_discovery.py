from __future__ import annotations

"""Materialized bounded O-regime discovery provider (Decoder v0.28.0).

This module closes the v0.27.5 audit's main epistemic blocker: O14+ scientific Tests no
longer receive a depth-erased theorem-transport projection as their primary observation.
Instead, a deterministic 24-state exact finite scanner panel is *materialized* at each depth
from the certified O7 compact replay root and the frozen generic-lift generator.  Structural
observables are computed from those actual generated carriers.

Scope is deliberately narrow and explicit:

* O7 is the graduated exact authority anchor.
* O8+ panels are exact finite synthetic instances of the frozen ownership/resource lift.
* O7..O13 are replay-calibrated byte-for-byte at the normalized scanner level against the
  frozen v0.26 scanner oracle before O14+ evidence is trusted.
* The provider can discover topology/branching/symmetry/lineage/service reorganization and
  can detect a resource-quotient future separator inside the admitted relation-add action
  language.
* It does NOT prove that the unrestricted future algebra cannot invent an entirely new action
  language. A new action/state/read semantics remains a theorem-reopen event.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Mapping
import atexit
import gc
import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
import zipfile

from .canon import canonical_sha256
from .execution import ExecutionPolicy
from . import maturation_parallel as mp_exec
from .o7_science_compat import load_frozen_legacy_scanner_fixture
from . import regime_scanner as rs


DISCOVERY_PROVIDER_REF = "IG_O_REGIME_MATERIALIZED_DISCOVERY_CAPABILITY_PROVIDER@1.0.0"
DISCOVERY_SEED_RESOURCE = "resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip"
DISCOVERY_SEED_SHA256 = "cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b"
DISCOVERY_SPEC_RESOURCE = "resources/decoder/O_REGIME_MATERIALIZED_DISCOVERY_SPEC_v1.json"
TRUST_REPAIR_CALIBRATION_RESOURCE = "resources/decoder/O7_O13_TRUST_REPAIR_CALIBRATION_V1.json"

# The frozen v0.26 regime scanner intentionally carries three process-global O7 memo
# dictionaries.  They are safe inside one scanner run, but O7 keys are scientific keys rather
# than engine-instance keys, so reusing them across independently extracted O7 replay engines
# can bind a new session to objects created by an older engine module.  Phase 3 must preserve
# the frozen scanner source byte-for-byte, therefore discovery sessions isolate those legacy
# caches at the adapter boundary instead of modifying regime_scanner.py.
_DISCOVERY_SESSION_LOCK = threading.Lock()
_SCANNER_CACHE_NAMES = ("_O7_DATA_CACHE", "_O7_SKIN_CACHE", "_O7_OWNER_CAP_CACHE")
_O7_RUNTIME_MODULE_NAME = "ig_materialized_discovery_o7_runtime"
_PROCESS_O7_RUNTIME: dict[str, Any] | None = None
_PROCESS_O7_RUNTIME_ATEXIT_REGISTERED = False


def _install_isolated_scanner_caches() -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for name in _SCANNER_CACHE_NAMES:
        value = getattr(rs, name, None)
        if not isinstance(value, dict):
            raise MaterializedDiscoveryError(f"frozen scanner runtime cache {name} is unavailable or not a dict")
        previous[name] = value
        setattr(rs, name, {})
    return previous


def _restore_scanner_caches(previous: Mapping[str, Any] | None) -> None:
    if not previous:
        return
    for name in _SCANNER_CACHE_NAMES:
        value = previous.get(name)
        if isinstance(value, dict):
            setattr(rs, name, value)


def _close_process_o7_runtime() -> None:
    """Release the one warm O7 replay kernel retained for this Python process."""
    global _PROCESS_O7_RUNTIME
    runtime = _PROCESS_O7_RUNTIME
    _PROCESS_O7_RUNTIME = None
    if not runtime:
        return
    try:
        base_states = runtime.get("base_states")
        if isinstance(base_states, list):
            base_states.clear()
        engine = runtime.get("engine")
        if engine is not None:
            try:
                engine.O6 = None
            except Exception:
                pass
        sys.modules.pop(_O7_RUNTIME_MODULE_NAME, None)
        prereg = sys.modules.get("ig_oscout_prereg_core")
        prereg_file = getattr(prereg, "__file__", None) if prereg is not None else None
        tmp = runtime.get("tmp")
        if prereg_file and tmp is not None:
            try:
                if Path(prereg_file).resolve().is_relative_to(Path(tmp).resolve()):
                    sys.modules.pop("ig_oscout_prereg_core", None)
            except Exception:
                pass
        ctx = runtime.get("seed_context")
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
        if tmp is not None:
            shutil.rmtree(Path(tmp), ignore_errors=True)
    finally:
        gc.collect()


def _get_process_o7_runtime(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable warm O7 replay kernel for the current interpreter.

    Caller holds _DISCOVERY_SESSION_LOCK. Scientific O8+ carrier state is never shared.
    """
    global _PROCESS_O7_RUNTIME, _PROCESS_O7_RUNTIME_ATEXIT_REGISTERED
    if _PROCESS_O7_RUNTIME is not None:
        return _PROCESS_O7_RUNTIME
    tmp = Path(tempfile.mkdtemp(prefix="ig_o_materialized_discovery_runtime_"))
    seed_context = None
    old_data_root = os.environ.get("OSCOUT_DATA_ROOT")
    try:
        seed_context = as_file(files("infinity_grid").joinpath("resources", "decoder", "O7_MATERIAL_ROOT_cb6f48641eb9.zip"))
        seed_path = Path(seed_context.__enter__())
        seed_root, o7root = _extract_compact_seed(seed_path, tmp)
        os.environ["OSCOUT_DATA_ROOT"] = str(o7root.resolve())
        sys.modules.pop(_O7_RUNTIME_MODULE_NAME, None)
        engine = rs._load_module(_O7_RUNTIME_MODULE_NAME, o7root / "02_CODE" / "o7_live_engine.py")
        engine.O6 = engine.import_o6()
        parent_map = engine.load_parent_records()
        _, bpairs = engine.O6.load_rules()
        bridge_pairs = sorted(tuple(map(int, x)) for x in bpairs)
        records = json.loads(
            (seed_root / "graduation_compact" / "07_INPUT_SNAPSHOTS" / "O7_IMMUTABLE_SURVIVORS.json").read_text(encoding="utf-8")
        )["records"]
        base = []
        for r in records:
            ctx = engine._profile_row_context(r, parent_map)
            edges = tuple(tuple(x) for x in r["edges"])
            base.append(rs.O7State(engine, ctx, edges, (0, 0, 0, 0, 0, 0, 0), r["state_digest"], r["lane"]))
        selected = rs._farthest_select(base, int(spec["panel"]["beam"]))
        _PROCESS_O7_RUNTIME = {
            "tmp": tmp, "seed_context": seed_context, "seed_path": seed_path,
            "seed_root": seed_root, "o7root": o7root, "engine": engine,
            "bridge_pairs": bridge_pairs, "base_states": selected, "base_candidate_count": len(base),
        }
        if not _PROCESS_O7_RUNTIME_ATEXIT_REGISTERED:
            atexit.register(_close_process_o7_runtime)
            _PROCESS_O7_RUNTIME_ATEXIT_REGISTERED = True
        return _PROCESS_O7_RUNTIME
    except Exception:
        sys.modules.pop(_O7_RUNTIME_MODULE_NAME, None)
        if seed_context is not None:
            try: seed_context.__exit__(None, None, None)
            except Exception: pass
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        if old_data_root is None:
            os.environ.pop("OSCOUT_DATA_ROOT", None)
        else:
            os.environ["OSCOUT_DATA_ROOT"] = old_data_root


class MaterializedDiscoveryError(RuntimeError):
    pass


def _json_tree(obj: Any) -> Any:
    """Normalize scanner Python objects to the machine-readable JSON tree used by Decoder."""
    return json.loads(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _resource_path(relative: str):
    p = files("infinity_grid")
    for part in relative.split("/"):
        if part == "resources" and str(p).endswith("infinity_grid"):
            p = p.joinpath(part)
        else:
            p = p.joinpath(part)
    return p


def load_materialized_discovery_spec() -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources", "decoder", "O_REGIME_MATERIALIZED_DISCOVERY_SPEC_v1.json")
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_compact_seed(seed_path: Path, work: Path) -> tuple[Path, Path]:
    seed_path = Path(seed_path)
    got = rs._sha_file(seed_path)
    if got != DISCOVERY_SEED_SHA256:
        raise MaterializedDiscoveryError(f"O7 compact discovery seed SHA mismatch: {got}")
    seed_out = work / "seed"
    with zipfile.ZipFile(seed_path) as zf:
        bad = zf.testzip()
        if bad:
            raise MaterializedDiscoveryError(f"O7 compact discovery seed CRC failure at {bad}")
        zf.extractall(seed_out)
    roots = [p for p in seed_out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise MaterializedDiscoveryError("ambiguous O7 compact discovery seed root")
    seed_root = roots[0]
    replay_zip = seed_root / "replay" / "Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29.zip"
    if not replay_zip.is_file():
        raise MaterializedDiscoveryError("compact discovery seed is missing the certified O7 replay root")
    replay_out = work / "o7"
    with zipfile.ZipFile(replay_zip) as zf:
        bad = zf.testzip()
        if bad:
            raise MaterializedDiscoveryError(f"O7 compact replay CRC failure at {bad}")
        zf.extractall(replay_out)
    roots = [p for p in replay_out.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise MaterializedDiscoveryError("ambiguous O7 compact replay root")
    return seed_root, roots[0]


def _anon_type_pair(a: int, b: int) -> tuple[int, int]:
    """Canonicalize action orientation under anonymous selected-owner exchange."""
    x = (int(a), int(b))
    y = (int(b), int(a))
    return min(x, y)


def _resource_future_signature(state: Any, bridge_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    """Exact finite one-step resource-future signature for top relation-add actions.

    Owner IDs are intentionally absent.  The selected-side type pair is quotiented by side
    exchange.  Successors are compared by the inherited resource skin, not by topology.
    This is the higher-O analogue of the O7 topology-aware read separator: if two states have
    equal inherited resource skin but different topology and this signature differs, the
    inherited quotient has become too coarse for the admitted action future.
    """
    rows: list[tuple[tuple[int, int], int, str]] = []
    for u in range(len(state.owner_caps)):
        for v in range(u + 1, len(state.owner_caps)):
            for a, b in bridge_pairs:
                ca = int(state.owner_caps[u][a])
                cb = int(state.owner_caps[v][b])
                if ca <= 0 or cb <= 0:
                    continue
                nxt = state.add_top_relation(u, v, a, b)
                if nxt is None:
                    continue
                rows.append((_anon_type_pair(a, b), int(ca * cb), str(nxt.skin)))
    normalized = [
        {"type_pair": list(tp), "action_copies": copies, "successor_resource_skin": skin}
        for tp, copies, skin in sorted(rows)
    ]
    return {
        "action_rows": len(normalized),
        "action_copy_total": sum(int(x["action_copies"]) for x in normalized),
        "signature_sha256": canonical_sha256(normalized),
    }


def _read_probe(states: list[Any], bridge_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    by_skin: dict[str, list[Any]] = defaultdict(list)
    for state in states:
        by_skin[str(state.skin)].append(state)
    checked: list[dict[str, Any]] = []
    separator_groups = 0
    for skin, vals in sorted(by_skin.items()):
        if len(vals) < 2:
            continue
        topo = {
            rs._sha(rs._uncolored_graph_canon(len(s.owner_caps), s.top_pairs))
            for s in vals
        }
        if len(topo) < 2:
            continue
        futures = []
        for s in sorted(vals, key=lambda x: x.construction_digest):
            future = _resource_future_signature(s, bridge_pairs)
            futures.append({
                "construction_digest": s.construction_digest,
                "motif_id": getattr(s, "motif_id", ""),
                "topology_sha256": rs._sha(rs._uncolored_graph_canon(len(s.owner_caps), s.top_pairs)),
                "resource_future_signature_sha256": future["signature_sha256"],
                "resource_future_action_rows": future["action_rows"],
                "resource_future_action_copy_total": future["action_copy_total"],
            })
        distinct = sorted({x["resource_future_signature_sha256"] for x in futures})
        separated = len(distinct) > 1
        separator_groups += int(separated)
        checked.append({
            "resource_skin": skin,
            "states": futures,
            "different_topology": True,
            "resource_future_equal": not separated,
        })
    if not checked:
        classification = "INCONCLUSIVE_NO_SAME_RESOURCE_DIFFERENT_TOPOLOGY_SEPARATOR_PAIR"
    elif separator_groups:
        classification = "INHERITED_RESOURCE_QUOTIENT_BREAK_CANDIDATE"
    else:
        classification = "TOPOLOGY_OBSERVER_ONLY_UNDER_TESTED_RELATION_ADD_ACTIONS"
    semantic_signature = {
        "admitted_action_scope": "PAIRWISE_TOP_RELATION_ADD_ON_MATERIALIZED_FIXED_LIFT_CARRIERS",
        "declared_read_set": [
            "HIERARCHICAL_OWNER_OCCURRENCES",
            "OWNER_RESOURCE_COUNTERS",
            "BRIDGE_COMPATIBILITY",
            "SELECTED_SCOPE",
            "COUNTER_MULTIPLICITY",
        ],
        "hidden_topology_operational": bool(separator_groups),
        "classification": classification,
    }
    return {
        "schema_id": "IG_O_REGIME_MATERIALIZED_READ_PROBE_V1",
        "status": "CANDIDATE" if separator_groups else ("PASS" if checked else "INCONCLUSIVE"),
        "classification": classification,
        "same_resource_different_topology_groups_checked": len(checked),
        "operational_future_separator_groups": separator_groups,
        "groups": checked,
        "semantic_signature": semantic_signature,
        "semantic_signature_sha256": canonical_sha256(semantic_signature),
        "nonclaim": (
            "This finite probe tests the admitted pairwise top relation-add future only. "
            "It does not prove that an unrestricted future algebra cannot introduce a new state field, "
            "action language, orientation, rewiring rule, or hidden read."
        ),
    }


def _state_probe(state: Any, bridge_pairs: list[tuple[int, int]]) -> dict[str, Any]:
    graph = rs._graph_basic(len(state.owner_caps), state.top_pairs)
    action = rs._action_aggregate(state, bridge_pairs)
    fiber = rs._factor_fiber(state)
    return {
        "construction_digest": str(state.construction_digest),
        "resource_skin_sha256": str(state.skin),
        "lane": str(getattr(state, "lane", "")),
        "motif_id": str(getattr(state, "motif_id", "")),
        "owner_count": len(state.owner_caps),
        "top_relation_count": len(state.top_pairs),
        "typed_top_edges": [list(map(int, e)) for e in state.typed_edges],
        "uncolored_topology_sha256": rs._sha(rs._uncolored_graph_canon(len(state.owner_caps), state.top_pairs)),
        "owner_resource_caps": [list(map(int, c)) for c in state.owner_caps],
        "total_free_by_type": list(map(int, state.total_caps)),
        "leaf_count": int(state.leaf_count),
        "relation_count_total": int(state.relation_count_total),
        "graph": {
            "degree": list(map(int, graph["degree"])),
            "diameter": int(graph["diameter"]),
            "radius": int(graph["radius"]),
            "articulations": int(graph["articulations"]),
            "bridges": int(graph["bridges"]),
            "triangles": int(graph["triangles"]),
            "cycle_rank": int(graph["beta"]),
        },
        "action": {
            "legal_action_labels": int(action["legal_action_labels"]),
            "action_orbits": int(action["action_orbits"]),
            "type_pair_support": int(action["type_pair_support"]),
            "owner_pair_support": int(action["owner_pair_support"]),
            "service_classes": int(action["service_classes"]),
            "service_signature_sha256": str(action["service_signature_sha256"]),
            "automorphism_size": int(action["automorphism_size"]),
            "owner_orbits": int(action["owner_orbits"]),
        },
        "factor_fiber_size": len(fiber),
        "factor_fiber_sha256": canonical_sha256(sorted(fiber)),
    }



def _cohort_structural_seed(state: Any) -> dict[str, Any]:
    """Cheap depth-erased structural seed for the *pre-selection* candidate cohort.

    Under the frozen O-regime scanner algorithms, the normalized branching/service/symmetry
    and factor-fiber observables are functions of the top typed relation graph and the
    owner resource-support masks.  Exact resource magnitudes, construction digest, leaf count
    and total action-copy multiplicity are deliberately excluded.  This gives maturation a
    longitudinal identity that is stable under pure recursive growth but sensitive to an actual
    within-motif structural change.
    """
    graph = rs._graph_basic(len(state.owner_caps), state.top_pairs)
    return {
        "motif_id": str(getattr(state, "motif_id", "")),
        "lane": str(getattr(state, "lane", "")),
        "owner_count": len(state.owner_caps),
        "top_relation_count": len(state.top_pairs),
        "top_pairs": [list(map(int, e)) for e in state.top_pairs],
        "typed_top_edges": [list(map(int, e)) for e in state.typed_edges],
        "uncolored_topology_sha256": rs._sha(rs._uncolored_graph_canon(len(state.owner_caps), state.top_pairs)),
        # Resource support is a vertex color.  Canonicalize it jointly with
        # typed owner relations; sorting colors independently destroys
        # owner<->resource incidence and can merge non-isomorphic states.
        "colored_owner_resource_topology_sha256": canonical_sha256(
            rs._colored_typed_canon(
                len(state.owner_caps),
                [tuple(int(x > 0) for x in c) for c in state.owner_caps],
                list(state.typed_edges),
            )
        ),
        "graph": {
            "degree": list(map(int, graph["degree"])),
            "diameter": int(graph["diameter"]),
            "radius": int(graph["radius"]),
            "articulations": int(graph["articulations"]),
            "bridges": int(graph["bridges"]),
            "triangles": int(graph["triangles"]),
            "cycle_rank": int(graph["beta"]),
        },
    }


def _candidate_cohort_evidence(candidate_states: list[Any], selected_states: list[Any]) -> dict[str, Any]:
    """Bind the full pre-selection candidate population to a longitudinal maturation identity."""
    rows = []
    seen: set[str] = set()
    for state in sorted(candidate_states, key=lambda x: (str(getattr(x, "motif_id", "")), x.construction_digest)):
        motif_id = str(getattr(state, "motif_id", ""))
        if not motif_id or motif_id in seen:
            raise MaterializedDiscoveryError(f"candidate cohort requires unique non-empty motif_id; got {motif_id!r}")
        seen.add(motif_id)
        seed = _cohort_structural_seed(state)
        rows.append({"motif_id": motif_id, "structural_seed_sha256": canonical_sha256(seed)})
    selected_ids = sorted(str(getattr(s, "motif_id", "")) for s in selected_states)
    population_signature = canonical_sha256(rows)
    obj = {
        "schema_id": "IG_O_REGIME_LONGITUDINAL_CANDIDATE_COHORT_V1",
        "schema_version": "1.0.0",
        "candidate_count": len(rows),
        "motif_structural_signatures": rows,
        "motif_id_set_sha256": canonical_sha256(sorted(seen)),
        "population_structural_signature_sha256": population_signature,
        "selected_motif_ids": selected_ids,
        "selected_membership_sha256": canonical_sha256(selected_ids),
        "selection_is_scientific_observation": False,
        "scientific_rule": (
            "Regime maturation is evaluated longitudinally on the complete deterministic pre-selection "
            "candidate cohort. Selected-panel turnover is a sampling/composition diagnostic and cannot "
            "by itself establish within-entity structural evolution or a regime boundary."
        ),
        "depth_erased_fields_excluded": [
            "exact resource counter magnitudes", "leaf_count", "total relation-count growth",
            "construction_digest", "resource_skin_sha256", "total action-copy multiplicity",
        ],
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


def _candidate_cohort_evidence_from_descriptors(candidate_rows: list[Mapping[str, Any]], selected_states: list[Any]) -> dict[str, Any]:
    """Build the exact longitudinal cohort artifact from parallel candidate descriptors."""
    rows = []
    seen: set[str] = set()
    for row in sorted(candidate_rows, key=lambda x: (str(x.get("motif_id", "")), str(x.get("construction_digest", "")))):
        motif_id = str(row.get("motif_id", ""))
        if not motif_id or motif_id in seen:
            raise MaterializedDiscoveryError(f"candidate cohort requires unique non-empty motif_id; got {motif_id!r}")
        seen.add(motif_id)
        rows.append({"motif_id": motif_id, "structural_seed_sha256": str(row["structural_seed_sha256"])})
    selected_ids = sorted(str(getattr(st, "motif_id", "")) for st in selected_states)
    obj = {
        "schema_id": "IG_O_REGIME_LONGITUDINAL_CANDIDATE_COHORT_V1",
        "schema_version": "1.0.0",
        "candidate_count": len(rows),
        "motif_structural_signatures": rows,
        "motif_id_set_sha256": canonical_sha256(sorted(seen)),
        "population_structural_signature_sha256": canonical_sha256(rows),
        "selected_motif_ids": selected_ids,
        "selected_membership_sha256": canonical_sha256(selected_ids),
        "selection_is_scientific_observation": False,
        "scientific_rule": (
            "Regime maturation is evaluated longitudinally on the complete deterministic pre-selection "
            "candidate cohort. Selected-panel turnover is a sampling/composition diagnostic and cannot "
            "by itself establish within-entity structural evolution or a regime boundary."
        ),
        "depth_erased_fields_excluded": [
            "exact resource counter magnitudes", "leaf_count", "total relation-count growth",
            "construction_digest", "resource_skin_sha256", "total action-copy multiplicity",
        ],
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


def _state_probe_from_analysis(state: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    graph = row["graph"]
    action = row["action"]
    fiber = set(row["fiber"])
    return {
        "construction_digest": str(state.construction_digest),
        "resource_skin_sha256": str(state.skin),
        "lane": str(getattr(state, "lane", "")),
        "motif_id": str(getattr(state, "motif_id", "")),
        "owner_count": len(state.owner_caps),
        "top_relation_count": len(state.top_pairs),
        "typed_top_edges": [list(map(int, e)) for e in state.typed_edges],
        "uncolored_topology_sha256": rs._sha(rs._uncolored_graph_canon(len(state.owner_caps), state.top_pairs)),
        "owner_resource_caps": [list(map(int, c)) for c in state.owner_caps],
        "total_free_by_type": list(map(int, state.total_caps)),
        "leaf_count": int(state.leaf_count),
        "relation_count_total": int(state.relation_count_total),
        "graph": {
            "degree": list(map(int, graph["degree"])),
            "diameter": int(graph["diameter"]),
            "radius": int(graph["radius"]),
            "articulations": int(graph["articulations"]),
            "bridges": int(graph["bridges"]),
            "triangles": int(graph["triangles"]),
            "cycle_rank": int(graph["beta"]),
        },
        "action": {
            "legal_action_labels": int(action["legal_action_labels"]),
            "action_orbits": int(action["action_orbits"]),
            "type_pair_support": int(action["type_pair_support"]),
            "owner_pair_support": int(action["owner_pair_support"]),
            "service_classes": int(action["service_classes"]),
            "service_signature_sha256": str(action["service_signature_sha256"]),
            "automorphism_size": int(action["automorphism_size"]),
            "owner_orbits": int(action["owner_orbits"]),
        },
        "factor_fiber_size": len(fiber),
        "factor_fiber_sha256": canonical_sha256(sorted(fiber)),
    }


def _build_next_panel_with_candidate_capture(
    engine: Any,
    prev: list[Any],
    level: int,
    pairs: list[tuple[int, int]],
    motifs: list[dict],
    spec: dict,
) -> tuple[list[Any], dict[str, Any], list[Any], list[Any]]:
    """Call the frozen scanner builder while capturing its exact pre-selection candidate pool.

    The frozen regime_scanner.py remains byte-identical.  MaterializedDiscoverySession already
    serializes access through _DISCOVERY_SESSION_LOCK, so this short adapter-local hook cannot
    race another discovery session.  The original selector is always restored fail-closed.
    """
    captured: list[list[Any]] = []
    original = rs._farthest_select

    def capture(pool: list[Any], k: int, must_include: list[Any] | None = None) -> list[Any]:
        captured.append(list(pool))
        return original(pool, k, must_include=must_include)

    rs._farthest_select = capture
    try:
        selected, build_meta, backbone = rs._build_next_panel(engine, prev, level, pairs, motifs, spec)
    finally:
        rs._farthest_select = original
    if len(captured) != 1:
        raise MaterializedDiscoveryError(
            f"frozen next-panel adapter expected exactly one farthest-selection call at O{level}; got {len(captured)}"
        )
    candidates = captured[0]
    if int(build_meta.get("candidates", -1)) != len(candidates):
        raise MaterializedDiscoveryError(f"candidate cohort capture count mismatch at O{level}")
    return selected, build_meta, backbone, candidates


def _materialization_evidence(
    *,
    states: list[Any],
    level: int,
    summary: Mapping[str, Any],
    bridge_pairs: list[tuple[int, int]],
    build_meta: Mapping[str, Any],
    calibration: Mapping[str, Any],
    candidate_cohort: Mapping[str, Any] | None = None,
    state_analysis: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if state_analysis is None:
        probes = [_state_probe(s, bridge_pairs) for s in sorted(states, key=lambda x: x.construction_digest)]
    else:
        probes = [_state_probe_from_analysis(s, state_analysis[s.construction_digest]) for s in sorted(states, key=lambda x: x.construction_digest)]
    read_probe = _read_probe(states, bridge_pairs)
    panel_payload = {
        "level": int(level),
        "state_probes": probes,
        "read_probe": read_probe,
    }
    evidence = {
        "schema_id": "IG_O_REGIME_MATERIALIZED_DISCOVERY_EVIDENCE_V1",
        "schema_version": "1.0.0",
        "status": "PASS" if read_probe["status"] != "INCONCLUSIVE" else "REVIEW_REQUIRED",
        "provider_ref": DISCOVERY_PROVIDER_REF,
        "level": int(level),
        "entity_population_materialized": True,
        "materialized_state_count": len(probes),
        "panel_science_sha256": canonical_sha256(panel_payload),
        "state_probes": probes,
        "read_probe": read_probe,
        "build_meta": _json_tree(build_meta),
        "scanner_summary_sha256": canonical_sha256(_json_tree(summary)),
        "source_seed_sha256": DISCOVERY_SEED_SHA256,
        "calibration": dict(calibration),
        "candidate_cohort": None if candidate_cohort is None else _json_tree(candidate_cohort),
        "scope": "BOUNDED_EXACT_FINITE_SYNTHETIC_FIXED_LIFT_DISCOVERY_PANEL",
        "nonclaims": [
            "NOT_A_HISTORICAL_O_LEVEL_GRADUATION",
            "NOT_AN_UNBOUNDED_O_TOWER_PROOF",
            "NOT_A_PROOF_THAT_NEW_ACTION_SEMANTICS_CAN_NEVER_EMERGE",
            "NOT_GEOMETRY_OR_PHYSICS",
        ],
    }
    evidence["science_sha256"] = canonical_sha256(evidence)
    return evidence


def theorem_transport_consistency(
    summary: Mapping[str, Any],
    materialization: Mapping[str, Any],
    transport_delta: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep theorem transport as an auxiliary consistency/growth-bound lane, not discovery."""
    if transport_delta is None:
        obj = {
            "schema_id": "IG_THEOREM_TRANSPORT_CONSISTENCY_AND_GROWTH_BOUND_AUDIT_V1",
            "status": "NOT_SUPPLIED",
            "classification": "AUXILIARY_TRANSPORT_LANE_ABSENT",
            "checks": [],
            "discovery_authority": False,
        }
        obj["science_sha256"] = canonical_sha256(obj)
        return obj
    checks: list[dict[str, Any]] = []
    grammar_ok = transport_delta.get("contract", {}).get("grammar_sha256") == summary.get("grammar_sha256")
    checks.append({"check_id": "GRAMMAR_PIN", "status": "PASS" if grammar_ok else "FAIL"})
    growth = transport_delta.get("growth_bounds", {})
    probes = materialization.get("state_probes", [])
    bounds_ok = True
    failures: list[dict[str, Any]] = []
    for p in probes:
        leaf = int(p["leaf_count"])
        rel = int(p["relation_count_total"])
        if not (int(growth["leaf_count_min"]) <= leaf <= int(growth["leaf_count_max"])):
            bounds_ok = False
            failures.append({"state": p["construction_digest"], "field": "leaf_count", "value": leaf})
        if not (int(growth["relation_count_min"]) <= rel <= int(growth["relation_count_max"])):
            bounds_ok = False
            failures.append({"state": p["construction_digest"], "field": "relation_count_total", "value": rel})
        for t, value in enumerate(p["total_free_by_type"]):
            if not (int(growth["min_total_free_by_type"][t]) <= int(value) <= int(growth["max_total_free_by_type"][t])):
                bounds_ok = False
                failures.append({"state": p["construction_digest"], "field": f"free_type_{t}", "value": int(value)})
    checks.append({
        "check_id": "CONSERVATIVE_GROWTH_BOUNDS_CONTAIN_MATERIALIZED_PANEL",
        "status": "PASS" if bounds_ok else "FAIL",
        "failures": failures[:20],
        "failure_count": len(failures),
    })
    org = transport_delta.get("normalized_organization", {})
    transport_non_discovery = (
        org.get("mode") == "THEOREM_TRANSPORTED_BY_O_DEPTH_ERASED_NORMALIZED_ORGANIZATION_V1"
        and org.get("quotient_observer", {}).get("resource_population_materialized") is False
    )
    checks.append({
        "check_id": "TRANSPORT_ROLE_IS_NON_DISCOVERY",
        "status": "PASS" if transport_non_discovery else "FAIL",
    })
    ok = all(x["status"] == "PASS" for x in checks)
    obj = {
        "schema_id": "IG_THEOREM_TRANSPORT_CONSISTENCY_AND_GROWTH_BOUND_AUDIT_V1",
        "schema_version": "1.0.0",
        "status": "PASS" if ok else "FAIL",
        "classification": "THEOREM_TRANSPORT_CONSISTENCY_AND_GROWTH_BOUND_AUDIT",
        "level": int(summary["level"]),
        "transport_science_sha256": transport_delta.get("science_sha256"),
        "materialized_panel_science_sha256": materialization.get("panel_science_sha256"),
        "checks": checks,
        "discovery_authority": False,
        "scientific_rule": "The materialized discovery panel is primary evidence; theorem transport is only a consistency and conservative-bound lane.",
    }
    obj["science_sha256"] = canonical_sha256(obj)
    return obj


@dataclass
class MaterializedDiscoveryLevel:
    level: int
    summary: dict[str, Any]
    materialization: dict[str, Any]
    states: list[Any]


class MaterializedDiscoverySession:
    """Stateful O7->On generator so a long run pays replay cost only once per process."""

    def __init__(self, *, honor_earned_laws: bool = True, execution_policy: ExecutionPolicy | None = None) -> None:
        # A discovery session owns the legacy scanner memoization namespace for its full
        # lifetime.  This makes sequential sessions deterministic and prevents cache objects
        # from one dynamically loaded O7 engine leaking into the next.
        self._session_lock_acquired = False
        self._scanner_cache_snapshot: dict[str, Any] | None = None
        _DISCOVERY_SESSION_LOCK.acquire()
        self._session_lock_acquired = True
        try:
            self._scanner_cache_snapshot = _install_isolated_scanner_caches()
        except Exception:
            _DISCOVERY_SESSION_LOCK.release()
            self._session_lock_acquired = False
            raise
        self.execution_policy = execution_policy or ExecutionPolicy(backend="AUTO", requested_workers="AUTO", scheduler="COST_WEIGHTED_SHARDS", owner="o-regime-maturation")
        self.execution_metadata: dict[int, dict[str, Any]] = {}
        self._tmp = None
        self._seed_context = None
        self._engine_module_name = None
        self._old_data_root = os.environ.get("OSCOUT_DATA_ROOT")
        try:
            # Everything after cache isolation is part of session initialization.  A failure
            # in any loader must unwind the isolated cache namespace and release the session
            # lock just as a failure in the replay-kernel bootstrap would.
            self.spec = rs.load_regime_scanner_spec()
            self.motifs = rs.load_motif_library()
            earned = rs.load_earned_regime_laws() if honor_earned_laws else {"laws": []}
            self.earned_laws = {
                x.get("law_id"): x
                for x in earned.get("laws", [])
                if x.get("status") == "EARNED"
            }
            self.legacy = load_frozen_legacy_scanner_fixture()
            runtime = _get_process_o7_runtime(self.spec)
            self.seed_root = runtime["seed_root"]
            self.o7root = runtime["o7root"]
            self.engine = runtime["engine"]
            self.bridge_pairs = list(runtime["bridge_pairs"])
            selected = list(runtime["base_states"])
            base_count = int(runtime["base_candidate_count"])
            self.level_states: dict[int, list[Any]] = {7: selected}
            self.candidate_cohorts: dict[int, dict[str, Any]] = {}
            self.backbones: dict[int, list[Any]] = {}
            self.levels: dict[int, MaterializedDiscoveryLevel] = {}
            self.build_meta: dict[int, dict[str, Any]] = {
                7: {
                    "candidates": base_count,
                    "selected": len(selected),
                    "source": "IMMUTABLE_O7_AUTHORITY_FROM_COMPACT_DISCOVERY_SEED",
                    "matched_backbone_states": 0,
                }
            }
            self._build_and_store(7)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        try:
            if getattr(self, "_old_data_root", None) is None:
                os.environ.pop("OSCOUT_DATA_ROOT", None)
            else:
                os.environ["OSCOUT_DATA_ROOT"] = self._old_data_root
            ctx = getattr(self, "_seed_context", None)
            if ctx is not None:
                try:
                    ctx.__exit__(None, None, None)
                except Exception:
                    pass
                self._seed_context = None
            # The O7 engine is dynamically imported under a unique module name.  Keeping that
            # module in sys.modules retains its O6 module, extracted-path constants and the
            # complete generated carrier graph after the session closes.  Repeated Phase-3
            # sessions would therefore accumulate hundreds of MB and progressively slow down.
            # Tear down those runtime roots explicitly before deleting the extracted tree.
            levels = getattr(self, "levels", None)
            if isinstance(levels, dict):
                for level_row in list(levels.values()):
                    states = getattr(level_row, "states", None)
                    if isinstance(states, list):
                        states.clear()
                    try: level_row.states = []
                    except Exception: pass
                levels.clear()
            level_states = getattr(self, "level_states", None)
            if isinstance(level_states, dict):
                level_states.clear()
            candidate_cohorts = getattr(self, "candidate_cohorts", None)
            if isinstance(candidate_cohorts, dict):
                candidate_cohorts.clear()
            execution_metadata = getattr(self, "execution_metadata", None)
            if isinstance(execution_metadata, dict):
                execution_metadata.clear()
            backbones = getattr(self, "backbones", None)
            if isinstance(backbones, dict):
                backbones.clear()
            build_meta = getattr(self, "build_meta", None)
            if isinstance(build_meta, dict):
                build_meta.clear()
            # The hash-pinned O7/O6 authority kernel is deliberately process-warm.
            self.engine = None
            gc.collect()
        finally:
            snapshot = getattr(self, "_scanner_cache_snapshot", None)
            _restore_scanner_caches(snapshot)
            self._scanner_cache_snapshot = None
            if getattr(self, "_session_lock_acquired", False):
                self._session_lock_acquired = False
                _DISCOVERY_SESSION_LOCK.release()

    def __enter__(self) -> "MaterializedDiscoverySession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def current_level(self) -> int:
        return max(self.levels)

    def _apply_inherited_representation_event(self, level: int, scan: dict[str, Any]) -> None:
        inherited = self.earned_laws.get("O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1")
        if not inherited or level < 11:
            return
        W = int(self.spec["promotion"]["representation_transition"]["W"])
        if level < 8 + W - 1:
            return
        ids = list(range(level - W + 1, level + 1))
        if not all(i in self.levels or i == level for i in ids):
            return
        rows = [self.levels[i].summary for i in ids[:-1]] + [scan]
        bsha = [x["matched_backbone"].get("signature_sha256") for x in rows]
        same_backbone = None not in bsha and len(set(bsha)) == 1
        broad_guard = all(
            x["grammar_sha256"] == rs.GRAMMAR_EXPECTED
            and x["obstruction_relief"]["all_bridge_types_supported_everywhere"]
            and x["obstruction_relief"]["all_owner_pairs_have_some_action"]
            and x["topology_services"]["access_bottleneck_service_present"]
            for x in rows
        )
        if same_backbone and broad_guard and inherited.get("discovery_backbone_signature_sha256") == bsha[-1]:
            scan.setdefault("inherited_law_events", []).append({
                "law_id": "O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1",
                "window": ids,
                "disposition": "SUPPRESSED_AS_ALREADY_EARNED",
            })

    def _calibration_for_level(self, level: int, scan: Mapping[str, Any]) -> dict[str, Any]:
        if level <= 13:
            expected = self.legacy["level_summaries"].get(str(level))
            if expected is None:
                raise MaterializedDiscoveryError(f"legacy calibration oracle missing O{level}")
            checks = {
                "normalized_signature_sha256": scan["normalized_signature_sha256"] == expected["normalized_signature_sha256"],
                "grammar_sha256": scan["grammar_sha256"] == expected["grammar_sha256"],
                "diversity_sha256": canonical_sha256(scan["diversity"]) == canonical_sha256(expected["diversity"]),
                "branching_sha256": canonical_sha256(scan["branching"]) == canonical_sha256(expected["branching"]),
                "symmetry_sha256": canonical_sha256(scan["symmetry"]) == canonical_sha256(expected["symmetry"]),
                "lineage_sha256": canonical_sha256(scan["lineage"]) == canonical_sha256(expected["lineage"]),
                "topology_services_sha256": canonical_sha256(scan["topology_services"]) == canonical_sha256(expected["topology_services"]),
            }
            # v0.28.8 intentionally repairs H01/H02.  Historical v0.26 hashes are
            # preserved for provenance, while live trust-repair replay is calibrated
            # against a separately frozen corrected O7-O13 oracle generated after the
            # independent automorphism/resource-incidence repair tests passed.
            calib_path = files("infinity_grid").joinpath(TRUST_REPAIR_CALIBRATION_RESOURCE)
            corrected = json.loads(calib_path.read_text(encoding="utf-8"))
            expected2 = corrected.get("levels", {}).get(str(level))
            if not isinstance(expected2, Mapping):
                raise MaterializedDiscoveryError(f"trust-repair calibration oracle missing O{level}")
            corrected_checks = {
                "normalized_signature_sha256": scan["normalized_signature_sha256"] == expected2["normalized_signature_sha256"],
                "grammar_sha256": scan["grammar_sha256"] == expected2["grammar_sha256"],
                "diversity_sha256": canonical_sha256(scan["diversity"]) == expected2["diversity_sha256"],
                "branching_sha256": canonical_sha256(scan["branching"]) == expected2["branching_sha256"],
                "symmetry_sha256": canonical_sha256(scan["symmetry"]) == expected2["symmetry_sha256"],
                "lineage_sha256": canonical_sha256(scan["lineage"]) == expected2["lineage_sha256"],
                "topology_services_sha256": canonical_sha256(scan["topology_services"]) == expected2["topology_services_sha256"],
            }
            if not all(corrected_checks.values()):
                bad2 = [k for k, ok in corrected_checks.items() if not ok]
                raise MaterializedDiscoveryError(f"O{level} trust-repair replay calibration failed: {bad2}")
            return {
                "status": "PASS",
                "mode": "FROZEN_O7_O13_TRUST_REPAIR_SCANNER_ORACLE_REPLAY",
                "trust_repair_calibration_science_sha256": corrected["science_sha256"],
                "legacy_scanner_science_sha256": self.legacy["science_sha256"],
                "legacy_byte_equal_checks": checks,
                "known_intentional_corrections": [k for k, ok in checks.items() if not ok],
                "checks": corrected_checks,
            }
        return {
            "status": "PASS",
            "mode": "POST_CALIBRATION_MATERIALIZED_DISCOVERY",
            "replay_calibrated_through": 13,
            "legacy_scanner_science_sha256": self.legacy["science_sha256"],
            "source_seed_sha256": DISCOVERY_SEED_SHA256,
        }

    def _build_and_store(self, level: int) -> MaterializedDiscoveryLevel:
        states = self.level_states[level]
        state_analysis = None
        if level >= 8:
            state_analysis, analysis_meta = mp_exec.analyze_selected_states(
                states, self.bridge_pairs, policy=self.execution_policy
            )
            self.execution_metadata.setdefault(level, {})["selected_state_analysis"] = analysis_meta
            scan, _ = mp_exec.scan_level_from_analysis(
                states, level, self.bridge_pairs, rs.GRAMMAR_EXPECTED, state_analysis
            )
        else:
            scan, _ = rs._scan_level(states, level, self.bridge_pairs, rs.GRAMMAR_EXPECTED)
        scan["matched_backbone"] = (
            rs._backbone_signature(self.backbones[level], self.bridge_pairs)
            if level in self.backbones
            else {"status": "HISTORICAL_O7_ANCHOR_NOT_SYNTHETIC_BACKBONE", "signature_sha256": None}
        )
        # The historical oracle was sealed as JSON.  Normalize live Counter/histogram keys
        # before genealogy/calibration so O7..O13 are compared in exactly the same data model
        # and all downstream capability hashes are canonical-JSON encodable.
        scan = _json_tree(scan)
        inherited_service = self.earned_laws.get("O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1")
        if inherited_service and scan["matched_backbone"].get("signature_sha256") == inherited_service.get("discovery_backbone_signature_sha256"):
            scan["matched_backbone"]["inherited_earned_law"] = "O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1"
        previous_scan = self.levels[level - 1].summary if level - 1 in self.levels else None
        genealogy = rs._classify_genealogy(previous_scan, scan)
        scan["genealogy"] = genealogy
        scan["build_meta"] = _json_tree(self.build_meta[level])
        scan["structural_shock_changed_families"] = rs._changed_families(genealogy)
        if level >= 10 and len(scan["structural_shock_changed_families"]) >= int(self.spec["promotion"]["structural_shock"]["minimum_changed_families"]):
            scan.setdefault("exploratory_shock_events", []).append({
                "level": level,
                "families": list(scan["structural_shock_changed_families"]),
                "status": "MATCHED_LONGITUDINAL_AUDIT_REQUIRED",
                "promotion_blocked": True,
            })
        self._apply_inherited_representation_event(level, scan)
        calibration = self._calibration_for_level(level, scan)
        cohort_evidence = None
        if level in self.candidate_cohorts:
            cohort_evidence = dict(self.candidate_cohorts[level])
        materialization = _materialization_evidence(
            states=states,
            level=level,
            summary=scan,
            bridge_pairs=self.bridge_pairs,
            build_meta=self.build_meta[level],
            calibration=calibration,
            candidate_cohort=cohort_evidence,
            state_analysis=state_analysis,
        )
        if level >= 8 and materialization["read_probe"]["same_resource_different_topology_groups_checked"] < 1:
            raise MaterializedDiscoveryError(f"O{level} discovery packet lost the preregistered same-resource topology separator control")
        scan["materialized_discovery"] = {
            "provider_ref": DISCOVERY_PROVIDER_REF,
            "entity_population_materialized": True,
            "materialized_state_count": materialization["materialized_state_count"],
            "panel_science_sha256": materialization["panel_science_sha256"],
            "read_probe_semantic_signature_sha256": materialization["read_probe"]["semantic_signature_sha256"],
            "read_probe_classification": materialization["read_probe"]["classification"],
            "source_seed_sha256": DISCOVERY_SEED_SHA256,
            "candidate_cohort_science_sha256": (cohort_evidence or {}).get("science_sha256"),
            "candidate_population_structural_signature_sha256": (cohort_evidence or {}).get("population_structural_signature_sha256"),
        }
        row = MaterializedDiscoveryLevel(level=level, summary=scan, materialization=materialization, states=states)
        self.levels[level] = row
        return row

    def advance_to(self, level: int) -> MaterializedDiscoveryLevel:
        level = int(level)
        if level < 7:
            raise MaterializedDiscoveryError("materialized O-regime discovery starts at O7")
        while self.current_level < level:
            nxt = self.current_level + 1
            prev_states = self.level_states[nxt - 1]
            candidates, selected_desc, candidate_exec_meta, center = mp_exec.parallel_candidate_descriptors(
                engine=self.engine,
                prev=prev_states,
                level=nxt,
                pairs=self.bridge_pairs,
                motifs=self.motifs,
                spec=self.spec,
                policy=self.execution_policy,
            )
            selected = mp_exec.rebuild_selected_states(
                selected_desc, engine=self.engine, prev=prev_states, level=nxt,
                pairs=self.bridge_pairs, motifs=self.motifs, center=center,
            )
            twin_constructed = {x.get("motif_id") for x in candidates} >= {"TWIN:A", "TWIN:B"}
            backbone = rs._build_backbone(self.engine, nxt, center, self.bridge_pairs)
            bm = {
                "candidates": len(candidates),
                "build_failures": int(candidate_exec_meta.get("build_failures", 0)),
                "twin_constructed": twin_constructed,
                "selected": len(selected),
                "matched_backbone_states": len(backbone),
            }
            if not bm.get("twin_constructed"):
                raise MaterializedDiscoveryError(f"O{nxt} failed to construct the topology-twin read control")
            self.level_states[nxt] = selected
            self.candidate_cohorts[nxt] = _candidate_cohort_evidence_from_descriptors(candidates, selected)
            self.backbones[nxt] = backbone
            self.build_meta[nxt] = bm
            self.execution_metadata.setdefault(nxt, {})["candidate_construction"] = candidate_exec_meta
            self._build_and_store(nxt)
        return self.levels[level]


def materialized_level(level: int) -> MaterializedDiscoveryLevel:
    """Convenience one-shot provider entry point; long runs should reuse a Session."""
    with MaterializedDiscoverySession() as session:
        row = session.advance_to(level)
        # state objects cannot outlive the session's extracted engine tree; return science JSON only
        return MaterializedDiscoveryLevel(level=row.level, summary=dict(row.summary), materialization=dict(row.materialization), states=[])
