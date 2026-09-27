from __future__ import annotations

"""Registered G6:S7 complete depth-2 continuation after the A6 budget review.

The frozen ordinary observer is unchanged. This stage reuses the completed A6
exact parent panels and certified depth-1 classes, then refines every remaining
non-singleton class by exact ordinary depth-2 behavior. No prior census or
A6 depth-1 science is recomputed.
"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

STAGE_ID = "G6:S7"
SUBSTAGE_ID = "G6:S7_DEPTH2_COMPLETION"
PLAN_SCHEMA = "IG_G6_S7_DEPTH2_COMPLETION_PREREGISTRATION_V1"
REUSE_SCHEMA = "IG_G6_S7_A6_DEPTH1_CERTIFIED_REUSE_INPUT_V1"
RESULT_SCHEMA = "IG_G6_S7_DEPTH2_COMPLETION_RESULT_V1"
D2_COMPONENT_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator"  # retained reference oracle
OUTER_GENERATION_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator"
CHILD_PROFILE_GENERATION_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator"
PARENT_ASSEMBLY_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_depth2_parent_profile_multiset_evaluator"

# Execution-only monotone cumulative prefixes of a lossless 124-coordinate
# representative basis for the frozen 248-action observer. Factor swap with
# operator transpose proves
#   (seed,(a,b),RIGHT) == (seed,(b,a),LEFT)
# as an exact canonical relation, so every omitted RIGHT coordinate is exactly
# reconstructible. The scientific observer remains the original 248 actions.
FACTOR_SWAP_NORMALIZATION_ID = "FACTOR_SWAP_LEFT_V1"
EXECUTION_CONTEXT_COUNT = 124
INNER_PREFIX_SCHEDULE = (124,)
EXECUTION_STRATEGY_ID = "FULL_INNER_OUTER_MONOTONE_FACTOR_SWAP_NORMALIZED_V2"

# A22: compact, reusable E2 evidence. These artifacts are provenance/recovery
# outputs only; they do not enter the scientific signature or refinement rule.
E2_MEMBERSHIPS_SCHEMA = "IG_G6_S7_DEPTH2_E2_MEMBERSHIPS_V1"
E2_SPLIT_WITNESSES_SCHEMA = "IG_G6_S7_DEPTH2_E2_SPLIT_WITNESSES_V1"
E2_SURVIVORS_SCHEMA = "IG_G6_S7_DEPTH2_E2_SURVIVORS_V1"
E2_CONTEXT_NORMALIZATION_SCHEMA = "IG_G6_S7_DEPTH2_E2_CONTEXT_NORMALIZATION_V1"
E2_PROGRESS_SCHEMA = "IG_G6_S7_DEPTH2_E2_PROGRESS_V1"
E2_MANIFEST_SCHEMA = "IG_G6_S7_DEPTH2_E2_MANIFEST_V1"
E2_CHECKPOINT_OUTER_INDICES = frozenset((7, 31, 63, 123))


class G6S7Depth2Error(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_id", "stage_id", "substage_id", "date", "status", "scientific_role",
        "question", "authority", "separation_rule", "observer", "panels",
        "execution_requirements", "registered_outcomes", "stopping_rule",
        "next_on_candidate", "next_on_discrete", "nonclaims",
        "preregistration_lineage", "scientific_design_sha256", "question_sha256",
    }
    if type(plan) is not dict or set(plan) != required:
        raise G6S7Depth2Error("S7D2_PLAN_FIELDS")
    if plan["schema_id"] != PLAN_SCHEMA or plan["stage_id"] != STAGE_ID or plan["substage_id"] != SUBSTAGE_ID:
        raise G6S7Depth2Error("S7D2_PLAN_SCHEMA")
    if canonical_sha256({k: v for k, v in plan.items() if k != "question_sha256"}) != plan["question_sha256"]:
        raise G6S7Depth2Error("S7D2_PLAN_HASH")
    design_keys = (
        "stage_id", "substage_id", "scientific_role", "question", "separation_rule",
        "observer", "panels", "execution_requirements", "registered_outcomes",
        "stopping_rule", "next_on_candidate", "next_on_discrete", "nonclaims",
    )
    if canonical_sha256({k: plan[k] for k in design_keys}) != plan["scientific_design_sha256"]:
        raise G6S7Depth2Error("S7D2_SCIENTIFIC_DESIGN_HASH")

    auth = plan["authority"]
    sha_keys = (
        "a6_completion_file_sha256", "a6_result_sha256", "a6_scientific_design_sha256",
        "reuse_input_file_sha256", "reuse_input_payload_sha256",
        "executor_decoder_source_sha256", "executor_decoder_package_sha256",
    )
    for key in sha_keys:
        value = auth.get(key)
        if type(value) is not str or len(value) != 64:
            raise G6S7Depth2Error("S7D2_AUTHORITY_SHA:" + key)
    refs = tuple(str(x) for x in auth.get("frozen_seed_refs", ()))
    if len(refs) != 4 or tuple(sorted(refs)) != refs:
        raise G6S7Depth2Error("S7D2_FROZEN_SEEDS")
    if type(auth.get("a6_internal_execution_id")) is not str:
        raise G6S7Depth2Error("S7D2_A6_EXECUTION_ID")

    sep = plan["separation_rule"]
    for key in (
        "marker_operations_allowed", "q_D_allowed_as_science_input", "observer_decode_allowed",
        "exact_parent_reconstruction_allowed", "exact_child_identity_in_signature_allowed",
    ):
        if sep.get(key) is not False:
            raise G6S7Depth2Error("S7D2_SEPARATION_RULE:" + key)
    if sep.get("exact_relation_oracle_allowed") is not True:
        raise G6S7Depth2Error("S7D2_EXACT_RELATION_REQUIRED")

    obs = plan["observer"]
    if obs.get("id") != "ORDINARY_BRANCH_MULTISET_FUTURE_V1" or obs.get("context_count") != 248:
        raise G6S7Depth2Error("S7D2_OBSERVER")
    if obs.get("branch_semantics") != "MULTISET_OF_SUCCESSOR_BLOCKS":
        raise G6S7Depth2Error("S7D2_BRANCH_SEMANTICS")
    ops = tuple(tuple(int(y) for y in x) for x in obs.get("operator_basis", ()))
    if len(ops) != 31 or len(set(ops)) != 31:
        raise G6S7Depth2Error("S7D2_OPERATOR_BASIS")

    req = plan["execution_requirements"]
    if req.get("certified_reuse_mandatory") is not True or req.get("all_collision_members_required") is not True:
        raise G6S7Depth2Error("S7D2_REUSE_OR_COVERAGE")
    if req.get("regenerate_prior_panels") is not False or req.get("recompute_depth1") is not False:
        raise G6S7Depth2Error("S7D2_REDUNDANT_RECOMPUTATION_FORBIDDEN")
    if req.get("sampling_allowed") is not False or req.get("posthoc_member_budget_allowed") is not False:
        raise G6S7Depth2Error("S7D2_SAMPLING_OR_BUDGET_FORBIDDEN")
    if req.get("resume_only_missing") is not True:
        raise G6S7Depth2Error("S7D2_RESUME_RULE")
    return dict(plan)


def _validate_reuse(reuse: Mapping[str, Any], plan: Mapping[str, Any],
                    a6_completion: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if type(reuse) is not dict or reuse.get("schema_id") != REUSE_SCHEMA:
        raise G6S7Depth2Error("S7D2_REUSE_SCHEMA")
    expected_payload = canonical_sha256({k: v for k, v in reuse.items() if k != "reuse_payload_sha256"})
    if reuse.get("reuse_payload_sha256") != expected_payload:
        raise G6S7Depth2Error("S7D2_REUSE_PAYLOAD_HASH")
    auth = plan["authority"]
    if reuse["reuse_payload_sha256"] != auth["reuse_input_payload_sha256"]:
        raise G6S7Depth2Error("S7D2_REUSE_PLAN_BINDING")
    result = (a6_completion.get("result") or {}).get("science_result") or {}
    src = reuse.get("source_execution") or {}
    if result.get("result_sha256") != auth["a6_result_sha256"] or src.get("result_sha256") != auth["a6_result_sha256"]:
        raise G6S7Depth2Error("S7D2_A6_RESULT_BINDING")
    if src.get("internal_execution_id") != auth["a6_internal_execution_id"]:
        raise G6S7Depth2Error("S7D2_A6_EXECUTION_BINDING")
    if result.get("classification") != "BUDGET_INSUFFICIENT_FOR_REGISTERED_DEPTH2_REFINEMENT":
        raise G6S7Depth2Error("S7D2_A6_CLASSIFICATION")
    if result.get("g6_graduated") is not False or result.get("r_series_started") is not False:
        raise G6S7Depth2Error("S7D2_A6_STAGE_SEPARATION")

    expected = {
        "s1": (4520, 363, 307, 4464, 54),
        "higher": (256, 30, 21, 247, 46),
    }
    parsed: dict[str, list[dict[str, Any]]] = {}
    for panel, (panel_count, class_count, multi_count, member_count, max_size) in expected.items():
        obj = reuse.get(panel) or {}
        classes = list(obj.get("collision_classes") or ())
        if len(classes) != multi_count:
            raise G6S7Depth2Error("S7D2_REUSE_CLASS_COUNT:" + panel)
        seen = set(); groups = []; sizes = []
        for c in classes:
            cid = str(c.get("depth1_class_id", "")); members = list(c.get("members") or ())
            if not cid or int(c.get("size", -1)) != len(members) or len(members) <= 1:
                raise G6S7Depth2Error("S7D2_REUSE_CLASS_SHAPE:" + panel)
            rows = []
            for row in members:
                token = str(row.get("state_token", "")); digest = str(row.get("identity_sha256", ""))
                if not token or token in seen or len(digest) != 64 or type(row.get("state")) is not dict:
                    raise G6S7Depth2Error("S7D2_REUSE_MEMBER_STATE:" + panel)
                seen.add(token); rows.append(dict(row))
            rows.sort(key=lambda x: x["state_token"]); groups.append({"depth1_class_id": cid, "members": rows}); sizes.append(len(rows))
        if sum(sizes) != member_count or max(sizes, default=0) != max_size:
            raise G6S7Depth2Error("S7D2_REUSE_MEMBER_COVERAGE:" + panel)
        declared_panel_count = int(obj.get("original_exact_state_count", obj.get("selected_exact_state_count", -1)))
        if declared_panel_count != panel_count or int(obj.get("depth1_class_count", -1)) != class_count or int(obj.get("depth1_multi_class_count", -1)) != multi_count:
            raise G6S7Depth2Error("S7D2_REUSE_PANEL_BINDING:" + panel)
        groups.sort(key=lambda c: c["depth1_class_id"]); parsed[panel] = groups

    rule = reuse.get("reuse_rule") or {}
    for key in ("no_s1_regeneration", "no_higher_regeneration", "no_depth1_recomputation", "exact_state_records_reused", "depth1_partition_groups_reused"):
        if rule.get(key) is not True:
            raise G6S7Depth2Error("S7D2_REUSE_RULE:" + key)
    return parsed


def _partition_memberships(runtime, phase_id: str) -> dict[str, str]:
    db = runtime._phase_root(phase_id) / "partition.sqlite3"
    if not db.is_file():
        raise G6S7Depth2Error("S7D2_PARTITION_DB_MISSING:" + phase_id)
    conn = sqlite3.connect(db)
    try:
        return {str(t): str(c) for t, c in conn.execute("SELECT task_id,class_token FROM task_results ORDER BY task_id")}
    finally:
        conn.close()


def _e2_group_id(panel: str, depth1_class_id: str, members: list[Mapping[str, Any]]) -> str:
    tokens = sorted(str(row["state_token"]) for row in members)
    return "E2G-" + canonical_sha256({
        "panel": str(panel), "initial_a6_depth1_class_id": str(depth1_class_id),
        "member_tokens": tokens,
    })


def _e2_context_normalization(basis_refs: tuple[str, ...],
                              operators: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    op_index = {tuple(op): i for i, op in enumerate(operators)}
    if len(op_index) != len(operators):
        raise G6S7Depth2Error("S7D2_E2_OPERATOR_DUPLICATE")
    execution = tuple((ref, op, "LEFT") for ref in basis_refs for op in operators)
    exec_index = {x: i for i, x in enumerate(execution)}
    rows = []
    aliases: dict[int, list[int]] = {i: [] for i in range(len(execution))}
    scientific = tuple((ref, op, pos) for ref in basis_refs for op in operators for pos in ("LEFT", "RIGHT"))
    for scientific_index, (ref, op, pos) in enumerate(scientific):
        if pos == "LEFT":
            target_op = op
            rule = "DIRECT_LEFT"
        else:
            target_op = (op[1], op[0])
            if target_op not in op_index:
                raise G6S7Depth2Error("S7D2_E2_TRANSPOSE_CLOSURE")
            rule = "RIGHT_TO_TRANSPOSED_LEFT"
        target = (ref, target_op, "LEFT")
        execution_index = exec_index[target]
        aliases[execution_index].append(scientific_index)
        rows.append({
            "scientific_context_index": scientific_index,
            "seed_ref": ref, "operator": list(op), "position": pos,
            "execution_context_index": execution_index,
            "execution_seed_ref": ref, "execution_operator": list(target_op),
            "execution_position": "LEFT", "mapping_rule": rule,
        })
    execution_rows = []
    for i, (ref, op, pos) in enumerate(execution):
        if len(aliases[i]) != 2:
            raise G6S7Depth2Error("S7D2_E2_ALIAS_COVERAGE")
        execution_rows.append({
            "execution_context_index": i, "seed_ref": ref,
            "operator": list(op), "position": pos,
            "scientific_alias_indices": sorted(aliases[i]),
        })
    obj = {
        "schema_id": E2_CONTEXT_NORMALIZATION_SCHEMA,
        "factor_swap_normalization_id": FACTOR_SWAP_NORMALIZATION_ID,
        "scientific_context_count": len(scientific),
        "execution_context_count": len(execution),
        "mapping_applies_independently_to_axes": ["outer", "inner"],
        "rule": "RIGHT(seed,(a,b))->LEFT(seed,(b,a))",
        "scientific_to_execution": rows,
        "execution_coordinates": execution_rows,
    }
    obj["normalization_payload_sha256"] = canonical_sha256(obj)
    return obj


def _partition_class_evidence(runtime, phase_id: str, class_tokens: set[str]) -> dict[str, dict[str, Any]]:
    if not class_tokens:
        return {}
    db = runtime._phase_root(phase_id) / "partition.sqlite3"
    if not db.is_file():
        raise G6S7Depth2Error("S7D2_PARTITION_DB_MISSING:" + phase_id)
    conn = sqlite3.connect(db)
    try:
        out: dict[str, dict[str, Any]] = {}
        tokens = sorted(class_tokens)
        for start in range(0, len(tokens), 500):
            chunk = tokens[start:start+500]
            q = ",".join("?" for _ in chunk)
            for token, sigsha, rep_tid, size, raw in conn.execute(
                f"SELECT class_token,signature_sha256,representative_task_id,size,representative_signature_bytes FROM classes WHERE class_token IN ({q}) ORDER BY class_token",
                chunk,
            ):
                if raw is None:
                    raise G6S7Depth2Error("S7D2_E2_REPRESENTATIVE_BYTES_MISSING:" + str(token))
                b = bytes(raw)
                if hashlib.sha256(b).hexdigest() != str(sigsha):
                    raise G6S7Depth2Error("S7D2_E2_REPRESENTATIVE_BYTES_DIGEST_MISMATCH:" + str(token))
                try:
                    text = b.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise G6S7Depth2Error("S7D2_E2_REPRESENTATIVE_BYTES_NOT_UTF8:" + str(token)) from exc
                out[str(token)] = {
                    "class_token": str(token), "signature_sha256": str(sigsha),
                    "representative_task_id": str(rep_tid), "phase_class_size": int(size),
                    "representative_signature_size_bytes": len(b),
                    "representative_signature_canonical_utf8": text,
                }
        missing = set(tokens) - set(out)
        if missing:
            raise G6S7Depth2Error("S7D2_E2_CLASS_EVIDENCE_MISSING:" + sorted(missing)[0])
        return out
    finally:
        conn.close()


def _refine_parent_prefix_with_e2(groups: list[dict[str, Any]], memberships: Mapping[str, str], *,
                                  panel: str, prefix_context_count: int,
                                  outer_context_index: int, phase_id: str, runtime,
                                  context_normalization: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out: list[dict[str, Any]] = []
    pending_events: list[dict[str, Any]] = []
    witness_tokens: set[str] = set()
    exec_rows = context_normalization.get("execution_coordinates") or []
    if outer_context_index < 0 or outer_context_index >= len(exec_rows):
        raise G6S7Depth2Error("S7D2_E2_OUTER_CONTEXT_INDEX")
    outer_coord = dict(exec_rows[outer_context_index])
    for g in groups:
        if len(g["members"]) <= 1:
            out.append(g)
            continue
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in g["members"]:
            tid = (
                f"S7D2-PARENT-PREFIX-{panel}-P{prefix_context_count:03d}-"
                f"O{outer_context_index:03d}-{row['state_token']}"
            )
            cls = memberships.get(tid)
            if cls is None:
                raise G6S7Depth2Error("S7D2_PARENT_PREFIX_MEMBERSHIP_MISSING:" + tid)
            buckets.setdefault(str(cls), []).append(row)
        ordered = []
        for cls in sorted(buckets):
            rows = sorted(buckets[cls], key=lambda r: str(r["state_token"]))
            child = {"depth1_class_id": g["depth1_class_id"], "members": rows}
            out.append(child)
            ordered.append((cls, child))
        if len(ordered) > 1:
            # A compact exact witness needs only a canonical separating pair.
            # The complete bucket membership is retained separately in the event.
            witness_tokens.update((ordered[0][0], ordered[1][0]))
            pending_events.append({
                "panel": panel, "phase_id": phase_id,
                "initial_a6_depth1_class_id": str(g["depth1_class_id"]),
                "parent_group_id": _e2_group_id(panel, g["depth1_class_id"], g["members"]),
                "component": {
                    "outer_execution_context_index": int(outer_context_index),
                    "outer_execution_coordinate": outer_coord,
                    "inner_execution_prefix_count": int(prefix_context_count),
                    "inner_scientific_coverage_via_normalization": int(prefix_context_count) * 2,
                },
                "buckets": [
                    {
                        "phase_class_token": cls,
                        "child_group_id": _e2_group_id(panel, child["depth1_class_id"], child["members"]),
                        "member_tokens": [str(r["state_token"]) for r in child["members"]],
                    }
                    for cls, child in ordered
                ],
                "canonical_separator_class_tokens": [ordered[0][0], ordered[1][0]],
            })
    evidence = _partition_class_evidence(runtime, phase_id, witness_tokens)
    events: list[dict[str, Any]] = []
    for event in pending_events:
        a, b = event.pop("canonical_separator_class_tokens")
        event["exact_structural_separator"] = [evidence[a], evidence[b]]
        event["split_event_sha256"] = canonical_sha256(event)
        events.append(event)
    out.sort(key=lambda g: (str(g["depth1_class_id"]), str(g["members"][0]["state_token"])))
    events.sort(key=lambda e: (e["panel"], e["initial_a6_depth1_class_id"], e["parent_group_id"]))
    return out, events


def _e2_current_groups(groups: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for panel in ("s1", "higher"):
        for g in groups[panel]:
            members = sorted(g["members"], key=lambda r: str(r["state_token"]))
            rows.append({
                "panel": panel, "initial_a6_depth1_class_id": str(g["depth1_class_id"]),
                "current_group_id": _e2_group_id(panel, g["depth1_class_id"], members),
                "members": [
                    {"state_token": str(r["state_token"]), "identity_sha256": str(r["identity_sha256"])}
                    for r in members
                ],
            })
    rows.sort(key=lambda r: (r["panel"], r["initial_a6_depth1_class_id"], r["current_group_id"]))
    return rows


def _write_e2_progress(out: Path, groups: Mapping[str, list[dict[str, Any]]],
                       split_witnesses: list[dict[str, Any]], *, cursor: Mapping[str, Any],
                       normalization_sha256: str, accepted_source_sha256: str,
                       scientific_design_sha256: str, reuse_payload_sha256: str) -> dict[str, Any]:
    obj = {
        "schema_id": E2_PROGRESS_SCHEMA, "status": "RUNNING",
        "accepted_decoder_source_sha256": accepted_source_sha256,
        "scientific_design_sha256": scientific_design_sha256,
        "reuse_input_payload_sha256": reuse_payload_sha256,
        "context_normalization_payload_sha256": normalization_sha256,
        "cursor": dict(cursor), "groups": _e2_current_groups(groups),
        "split_witnesses": list(split_witnesses),
    }
    obj["progress_payload_sha256"] = canonical_sha256(obj)
    write_json_atomic(out / "G6_S7_DEPTH2_E2_PROGRESS.json", obj)
    return obj


def _write_e2_final(out: Path, groups: Mapping[str, list[dict[str, Any]]],
                    split_witnesses: list[dict[str, Any]], context_normalization: Mapping[str, Any], *,
                    accepted_source_sha256: str, scientific_design_sha256: str,
                    reuse_payload_sha256: str, question_sha256: str) -> dict[str, Any]:
    current = _e2_current_groups(groups)
    memberships = []
    survivors = []
    for g in current:
        members = list(g["members"]); size = len(members); rep = members[0]
        final_id = g["current_group_id"]
        for row in members:
            memberships.append({
                "panel": g["panel"], "state_token": row["state_token"],
                "identity_sha256": row["identity_sha256"],
                "initial_a6_depth1_class_id": g["initial_a6_depth1_class_id"],
                "final_depth2_class_id": final_id, "final_class_size": size,
                "final_representative_state_token": rep["state_token"],
                "survivor_non_singleton": size > 1,
            })
        if size > 1:
            survivors.append({
                "panel": g["panel"], "initial_a6_depth1_class_id": g["initial_a6_depth1_class_id"],
                "final_depth2_class_id": final_id, "class_size": size,
                "representative": rep, "members": members,
            })
    memberships.sort(key=lambda r: (r["panel"], r["state_token"]))
    survivors.sort(key=lambda r: (r["panel"], r["final_depth2_class_id"]))
    memberships_obj = {
        "schema_id": E2_MEMBERSHIPS_SCHEMA, "member_count": len(memberships),
        "rows": memberships,
    }
    memberships_obj["memberships_payload_sha256"] = canonical_sha256(memberships_obj)
    split_obj = {
        "schema_id": E2_SPLIT_WITNESSES_SCHEMA, "split_event_count": len(split_witnesses),
        "events": list(split_witnesses),
    }
    split_obj["split_witnesses_payload_sha256"] = canonical_sha256(split_obj)
    survivor_obj = {
        "schema_id": E2_SURVIVORS_SCHEMA, "survivor_class_count": len(survivors),
        "survivors": survivors,
    }
    survivor_obj["survivors_payload_sha256"] = canonical_sha256(survivor_obj)
    paths = {
        "memberships": out / "G6_S7_DEPTH2_E2_MEMBERSHIPS.json",
        "split_witnesses": out / "G6_S7_DEPTH2_E2_SPLIT_WITNESSES.json",
        "survivors": out / "G6_S7_DEPTH2_E2_SURVIVORS.json",
        "context_normalization": out / "G6_S7_DEPTH2_E2_CONTEXT_NORMALIZATION.json",
    }
    write_json_atomic(paths["memberships"], memberships_obj)
    write_json_atomic(paths["split_witnesses"], split_obj)
    write_json_atomic(paths["survivors"], survivor_obj)
    # The normalization artifact is also written before execution starts; rewrite
    # atomically here to make final packaging independent of an earlier partial file.
    write_json_atomic(paths["context_normalization"], dict(context_normalization))
    files = []
    for logical_name, path in sorted(paths.items()):
        files.append({
            "logical_name": logical_name, "file_name": path.name,
            "sha256": _sha_file(path), "size_bytes": path.stat().st_size,
        })
    manifest = {
        "schema_id": E2_MANIFEST_SCHEMA, "status": "PASS",
        "accepted_decoder_source_sha256": accepted_source_sha256,
        "scientific_design_sha256": scientific_design_sha256,
        "question_sha256": question_sha256,
        "reuse_input_payload_sha256": reuse_payload_sha256,
        "member_count": len(memberships), "split_event_count": len(split_witnesses),
        "survivor_class_count": len(survivors),
        "structural_authority": "EXACT_REPRESENTATIVE_SIGNATURE_BYTES_FOR_CANONICAL_SEPARATOR_PAIR",
        "hash_role": "PROVENANCE_BINDING_ONLY",
        "recursive_zip_nesting": False, "files": files,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    mp = out / "G6_S7_DEPTH2_E2_MANIFEST.json"
    write_json_atomic(mp, manifest)
    return {
        "schema_id": E2_MANIFEST_SCHEMA, "manifest_file_sha256": _sha_file(mp),
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "member_count": len(memberships), "split_event_count": len(split_witnesses),
        "survivor_class_count": len(survivors),
    }


def _prefilter_task(panel: str, row: Mapping[str, Any], *, outer_context_index: int,
                    basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]) -> TaskSpec:
    """Execution-only prefix-1 probe of the frozen depth-2 component.

    This uses the already registered reference evaluator and existing kernel/cache
    services.  It cannot alter the scientific observer: a prefix-1 inequality is
    a prefix of the declared prefix-8 signature, so later coordinates cannot merge
    classes that have already split.
    """
    payload = {
        "state_tree": row["state"], "basis_refs": list(basis_refs),
        "operator_basis": [list(x) for x in operators],
        "outer_context_index": int(outer_context_index),
        "inner_prefix_context_count": 1,
    }
    return TaskSpec(
        task_id=f"S7D2-PREFILTER-{panel}-O{outer_context_index:03d}-{row['state_token']}",
        task_kind="G6_S7_DEPTH2_EXECUTION_ONLY_PREFIX1_PREFILTER",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, float(row["state"].get("n", 1))),
    )


def _refine_prefilter(groups: list[dict[str, Any]], memberships: Mapping[str, str], *,
                      panel: str, outer_context_index: int) -> list[dict[str, Any]]:
    out = []
    for g in groups:
        if len(g["members"]) <= 1:
            out.append(g); continue
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in g["members"]:
            tid = f"S7D2-PREFILTER-{panel}-O{outer_context_index:03d}-{row['state_token']}"
            cls = memberships.get(tid)
            if cls is None:
                raise G6S7Depth2Error("S7D2_PREFILTER_MEMBERSHIP_MISSING:" + tid)
            buckets.setdefault(cls, []).append(row)
        for rows in buckets.values():
            rows.sort(key=lambda r: str(r["state_token"]))
            out.append({"depth1_class_id": g["depth1_class_id"], "members": rows})
    return sorted(out, key=lambda g: (g["depth1_class_id"], str(g["members"][0]["state_token"])))


def _parent_prefix_task(panel: str, row: Mapping[str, Any], *, outer_context_index: int,
                        prefix_context_count: int, basis_refs: tuple[str, ...],
                        operators: tuple[tuple[int, int], ...]) -> TaskSpec:
    """Exact parent-level depth-2 component with no global child task universe.

    The registered evaluator materializes only one parent's outer children inside
    one worker task and returns the exact multiset of cumulative child Sig1
    prefixes.  Task count is therefore bounded by active parent count.
    """
    payload = {
        "state_tree": row["state"], "basis_refs": list(basis_refs),
        "operator_basis": [list(x) for x in operators],
        "outer_context_index": int(outer_context_index),
        "inner_prefix_context_count": int(prefix_context_count),
        "execution_context_normalization": FACTOR_SWAP_NORMALIZATION_ID,
    }
    return TaskSpec(
        task_id=(
            f"S7D2-PARENT-PREFIX-{panel}-P{prefix_context_count:03d}-"
            f"O{outer_context_index:03d}-{row['state_token']}"
        ),
        task_kind="G6_S7_DEPTH2_PARENT_PREFIX_MONOTONE_COMPONENT",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, float(row["state"].get("n", 1)) * float(prefix_context_count)),
    )


def _refine_parent_prefix(groups: list[dict[str, Any]], memberships: Mapping[str, str], *,
                          panel: str, prefix_context_count: int,
                          outer_context_index: int) -> list[dict[str, Any]]:
    out = []
    for g in groups:
        if len(g["members"]) <= 1:
            out.append(g)
            continue
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in g["members"]:
            tid = (
                f"S7D2-PARENT-PREFIX-{panel}-P{prefix_context_count:03d}-"
                f"O{outer_context_index:03d}-{row['state_token']}"
            )
            cls = memberships.get(tid)
            if cls is None:
                raise G6S7Depth2Error("S7D2_PARENT_PREFIX_MEMBERSHIP_MISSING:" + tid)
            buckets.setdefault(cls, []).append(row)
        for rows in buckets.values():
            rows.sort(key=lambda r: str(r["state_token"]))
            out.append({"depth1_class_id": g["depth1_class_id"], "members": rows})
    return sorted(out, key=lambda g: (g["depth1_class_id"], str(g["members"][0]["state_token"])))


def _outer_task(panel: str, row: Mapping[str, Any], *, outer_context_index: int,
                basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]) -> TaskSpec:
    payload = {
        "state_tree": row["state"], "basis_refs": list(basis_refs),
        "operator_basis": [list(x) for x in operators],
        "outer_context_index": int(outer_context_index),
    }
    return TaskSpec(
        task_id=f"S7D2-OUTER-{panel}-O{outer_context_index:03d}-{row['state_token']}",
        task_kind="G6_S7_DEPTH2_OUTER_CHILD_GENERATION",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, float(row["state"].get("n", 1))),
    )


def _outer_store(runtime, phase_id: str) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Return compact exact child rows and parent-task -> child-token incidence.

    The StageRuntime generation store has already established exact structural
    equality by complete canonical bytes.  Keep canonical identity/state as
    compact encoded values in the controller; do not eagerly decode ~O(10^5)
    children into nested Python objects before profile dispatch.
    """
    db = runtime._phase_root(phase_id) / "state_store.sqlite3"
    if not db.is_file():
        raise G6S7Depth2Error("S7D2_OUTER_STORE_MISSING:" + phase_id)
    conn = sqlite3.connect(db)
    try:
        rows: dict[str, dict[str, Any]] = {}
        for token, digest, b, canonical_size, state_json in conn.execute(
            "SELECT state_token,index_digest,canonical_bytes,canonical_size_bytes,state_json "
            "FROM states ORDER BY canonical_bytes,state_token"
        ):
            rows[str(token)] = {
                "state_token": str(token),
                "identity_sha256": str(digest),
                "identity_canonical_bytes": bytes(b),
                "canonical_size_bytes": int(canonical_size),
                "state_json": str(state_json),
            }
        by_task: dict[str, list[str]] = {}
        for tid, token in conn.execute(
            "SELECT task_id,state_token FROM occurrences ORDER BY task_id,occurrence_id"
        ):
            by_task.setdefault(str(tid), []).append(str(token))
    finally:
        conn.close()
    return rows, by_task

def _profile_store_by_exact_bytes(runtime, phase_id: str) -> dict[bytes, tuple[int, ...]]:
    out: dict[bytes, tuple[int, ...]] = {}
    for row in runtime.iter_generated_states(phase_id=phase_id):
        key = bytes(row["identity_canonical_bytes"])
        counts = tuple(int(x) for x in (row["state"].get("profile_counts") or ()))
        if key in out and out[key] != counts:
            raise G6S7Depth2Error("S7D2_PROFILE_STORE_EXACT_IDENTITY_CONFLICT:" + phase_id)
        out[key] = counts
    return out


def _profile_store_by_child_token(runtime, phase_id: str) -> dict[str, tuple[int, ...]]:
    """Execution-only compact join from outer-store token to cumulative profile.

    Exact child structural identity remains the generation-store authority.  The
    outer-store token is merely a stable join key carried in state payload and
    never enters a scientific signature.
    """
    out: dict[str, tuple[int, ...]] = {}
    for row in runtime.iter_generated_states(phase_id=phase_id):
        state = row.get("state") or {}
        token = str(state.get("child_token", ""))
        if not token:
            raise G6S7Depth2Error("S7D2_PROFILE_CHILD_TOKEN_MISSING:" + phase_id)
        counts = tuple(int(x) for x in (state.get("profile_counts") or ()))
        if token in out and out[token] != counts:
            raise G6S7Depth2Error("S7D2_PROFILE_STORE_CHILD_TOKEN_CONFLICT:" + phase_id)
        out[token] = counts
    return out


def _profile_task(panel: str, child_row: Mapping[str, Any], *, outer_context_index: int,
                  start_context_index: int, end_context_index: int,
                  previous_counts: tuple[int, ...], basis_refs: tuple[str, ...],
                  operators: tuple[tuple[int, int], ...]) -> TaskSpec:
    identity_bytes = bytes(child_row["identity_canonical_bytes"])
    try:
        identity_json = identity_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise G6S7Depth2Error("S7D2_OUTER_CHILD_IDENTITY_UTF8") from exc
    state_json = child_row.get("state_json")
    if type(state_json) is not str:
        state = child_row.get("state")
        if type(state) is not dict:
            raise G6S7Depth2Error("S7D2_OUTER_CHILD_STATE_MISSING")
        state_json = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    token = str(child_row["state_token"])
    payload = {
        "child_token": token,
        "child_state_json": state_json,
        "child_identity_json": identity_json,
        # Share immutable frozen observer objects across controller TaskSpecs.
        # Worker serialization may copy a bounded dispatch window only.
        "basis_refs": basis_refs,
        "operator_basis": operators,
        "start_context_index": int(start_context_index),
        "end_context_index": int(end_context_index),
        "previous_counts": tuple(int(x) for x in previous_counts),
    }
    # Canonical identity size is an execution-only load proxy that avoids
    # decoding the tree solely to recover a scheduling weight.
    size_hint = max(1, int(child_row.get("canonical_size_bytes", len(identity_bytes))))
    return TaskSpec(
        task_id=f"S7D2-PROFILE-{panel}-O{outer_context_index:03d}-P{end_context_index:03d}-{token}",
        task_kind="G6_S7_DEPTH2_UNIQUE_CHILD_PROFILE_EXTENSION",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, (float(size_hint) / 2048.0) * float(end_context_index - start_context_index)),
    )

def _assembly_task(panel: str, row: Mapping[str, Any], *, outer_context_index: int,
                   prefix_context_count: int, child_profiles: list[tuple[int, ...]]) -> TaskSpec:
    # Keep one compact immutable string per parent rather than thousands of
    # nested Python integer/list objects resident across the full task panel.
    compact_profiles = json.dumps(child_profiles, separators=(",", ":"), ensure_ascii=False)
    payload = {
        "inner_prefix_context_count": int(prefix_context_count),
        "child_profile_counts_json": compact_profiles,
    }
    return TaskSpec(
        task_id=f"S7D2-ASSEMBLE-{panel}-P{prefix_context_count:03d}-O{outer_context_index:03d}-{row['state_token']}",
        task_kind="G6_S7_DEPTH2_PARENT_PROFILE_MULTISET_ASSEMBLY",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, float(len(child_profiles))),
    )

def _stats(groups: list[dict[str, Any]]) -> dict[str, int]:
    multi = [g for g in groups if len(g["members"]) > 1]
    return {
        "class_count_within_a6_collision_members": len(groups),
        "multi_class_count": len(multi),
        "collision_member_count": sum(len(g["members"]) for g in multi),
        "max_class_size": max((len(g["members"]) for g in groups), default=0),
        "singleton_count_within_a6_collision_members": sum(1 for g in groups if len(g["members"]) == 1),
    }


def run_g6_s7_depth2_completion(*, plan_path: str | Path, artifacts: Mapping[str, str | Path],
                                output_dir: str | Path, accepted_source_sha256: str,
                                internal_execution_id: str, runtime) -> dict[str, Any]:
    require_controller_execution_origin("g6-s7-depth2-completion")
    plan_path = Path(plan_path).resolve(strict=True)
    plan = _validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    paths = {k: Path(v).resolve(strict=True) for k, v in artifacts.items()}
    auth = plan["authority"]
    if accepted_source_sha256 != auth["executor_decoder_source_sha256"]:
        raise G6S7Depth2Error("S7D2_SOURCE_BINDING")
    if _sha_file(paths["a6_completion"]) != auth["a6_completion_file_sha256"]:
        raise G6S7Depth2Error("S7D2_A6_COMPLETION_FILE_BINDING")
    if _sha_file(paths["reuse_input"]) != auth["reuse_input_file_sha256"]:
        raise G6S7Depth2Error("S7D2_REUSE_FILE_BINDING")
    a6_completion = json.loads(paths["a6_completion"].read_text(encoding="utf-8"))
    reuse = json.loads(paths["reuse_input"].read_text(encoding="utf-8"))
    panels = _validate_reuse(reuse, plan, a6_completion)

    basis_refs = tuple(str(x) for x in auth["frozen_seed_refs"])
    operators = tuple(tuple(int(y) for y in x) for x in plan["observer"]["operator_basis"])
    contexts = tuple((ref, op, pos) for ref in basis_refs for op in operators for pos in ("LEFT", "RIGHT"))
    if len(contexts) != 248:
        raise G6S7Depth2Error("S7D2_CONTEXT_COUNT")
    operator_set = set(operators)
    if any((b, a) not in operator_set for a, b in operators):
        raise G6S7Depth2Error("S7D2_FACTOR_SWAP_OPERATOR_TRANSPOSE_CLOSURE")
    execution_contexts = tuple((ref, op, "LEFT") for ref in basis_refs for op in operators)
    if len(execution_contexts) != EXECUTION_CONTEXT_COUNT:
        raise G6S7Depth2Error("S7D2_EXECUTION_CONTEXT_COUNT")

    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    context_normalization = _e2_context_normalization(basis_refs, operators)
    write_json_atomic(out / "G6_S7_DEPTH2_E2_CONTEXT_NORMALIZATION.json", context_normalization)

    groups = {panel: [{"depth1_class_id": c["depth1_class_id"], "members": list(c["members"])} for c in panels[panel]] for panel in ("s1", "higher")}
    initial = {panel: _stats(groups[panel]) for panel in groups}
    phase_count = 0
    task_evaluations = 0
    prefix_rounds = []
    split_witnesses: list[dict[str, Any]] = []

    # Parent-prefix monotone execution.
    #
    # The previously active split path created one global task per distinct exact
    # child.  At O000 that produced 190,641 child tasks for only 4,452 active
    # parents while cross-parent exact-child dedup was ~0.57%.  The already
    # registered reference evaluator is byte-equivalent to that split
    # generation/profile/assembly construction for every frozen cumulative
    # prefix, so use it directly at parent granularity.
    #
    # A25 execution-only schedule: evaluate the complete lossless 124-coordinate
    # child Sig1 representative vector at each outer coordinate before advancing
    # breadth.  The former cumulative 1/8/32/64/124 prefix sweep repaid earlier
    # coordinates (229 inner coordinate evaluations for a P124 survivor) and
    # pruned poorly at P001.  A complete P124 component costs 124 coordinates
    # once, then exact monotone refinement permanently removes singleton parents.
    # The scientific observer remains the frozen 248 contexts via the already
    # certified factor-swap equivalence; later outer coordinates cannot remerge a
    # singleton.
    for prefix_count in INNER_PREFIX_SCHEDULE:
        round_meta = {
            "inner_prefix_context_count": prefix_count,
            "outer_contexts_evaluated": {"s1": 0, "higher": 0},
            "task_evaluations": {"s1": 0, "higher": 0},
            "starting_stats": {p: _stats(groups[p]) for p in groups},
            "execution_strategy": EXECUTION_STRATEGY_ID,
        }
        for outer_idx in range(EXECUTION_CONTEXT_COUNT):
            component_split = False
            for panel in ("s1", "higher"):
                active = sorted(
                    (row for g in groups[panel] if len(g["members"]) > 1 for row in g["members"]),
                    key=lambda r: str(r["state_token"]),
                )
                if not active:
                    continue

                tasks = [
                    _parent_prefix_task(
                        panel, row, outer_context_index=outer_idx,
                        prefix_context_count=prefix_count,
                        basis_refs=basis_refs, operators=operators,
                    )
                    for row in active
                ]
                phase_id = (
                    f"S7D2_{panel.upper()}_P{prefix_count:03d}_"
                    f"O{outer_idx:03d}_PARENT_PREFIX"
                )
                runtime.run_structural_partition(
                    phase_id=phase_id, tasks=tasks,
                    evaluator_ref=D2_COMPONENT_EVALUATOR,
                    requested_workers=4,
                    max_tasks=4464 if panel == "s1" else 247,
                    stream_shard_task_limit=1,
                )
                n_tasks = len(tasks)
                groups[panel], new_split_events = _refine_parent_prefix_with_e2(
                    groups[panel], _partition_memberships(runtime, phase_id),
                    panel=panel, prefix_context_count=prefix_count,
                    outer_context_index=outer_idx, phase_id=phase_id, runtime=runtime,
                    context_normalization=context_normalization,
                )
                split_witnesses.extend(new_split_events)
                component_split = component_split or bool(new_split_events)
                phase_count += 1
                task_evaluations += n_tasks
                round_meta["outer_contexts_evaluated"][panel] += 1
                round_meta["task_evaluations"][panel] += n_tasks
                del tasks

            discrete_now = all(_stats(groups[p])["multi_class_count"] == 0 for p in ("s1", "higher"))
            if component_split or outer_idx in E2_CHECKPOINT_OUTER_INDICES or discrete_now:
                _write_e2_progress(
                    out, groups, split_witnesses,
                    cursor={
                        "inner_prefix_context_count": prefix_count,
                        "completed_outer_execution_context_index": outer_idx,
                        "completed_outer_execution_context_count": outer_idx + 1,
                    },
                    normalization_sha256=context_normalization["normalization_payload_sha256"],
                    accepted_source_sha256=accepted_source_sha256,
                    scientific_design_sha256=plan["scientific_design_sha256"],
                    reuse_payload_sha256=auth["reuse_input_payload_sha256"],
                )
            if discrete_now:
                break

        round_meta["ending_stats"] = {p: _stats(groups[p]) for p in groups}
        prefix_rounds.append(round_meta)
        if all(_stats(groups[p])["multi_class_count"] == 0 for p in ("s1", "higher")):
            break

    final_stats = {p: _stats(groups[p]) for p in groups}
    discrete_early = all(final_stats[p]["multi_class_count"] == 0 for p in ("s1", "higher"))
    full_for_survivors = bool(prefix_rounds and prefix_rounds[-1]["inner_prefix_context_count"] == EXECUTION_CONTEXT_COUNT)
    if not discrete_early and not full_for_survivors:
        raise G6S7Depth2Error("S7D2_INCOMPLETE_FULL_OBSERVER")

    candidate = any(final_stats[p]["multi_class_count"] > 0 for p in ("s1", "higher"))
    if candidate:
        classification = "NONTRIVIAL_ORDINARY_BRANCH_MULTISET_QUOTIENT_CANDIDATE_SURVIVES_COMPLETE_DEPTH2"
        next_authorized = plan["next_on_candidate"]
        interpretation = "At least one exact non-singleton A6 depth-1 class remains non-singleton after the complete ordinary depth-2 successor-block multiset observer. This is a bounded compressed-public-state candidate for S8, not an all-finite congruence or minimality theorem."
    else:
        classification = "ORDINARY_BRANCH_MULTISET_DISCRETE_BY_COMPLETE_DEPTH2_ON_REGISTERED_PANELS"
        next_authorized = plan["next_on_discrete"]
        interpretation = "Monotone exact depth-2 refinement separates every A6 depth-1 collision member on both frozen panels. Later skipped components cannot remerge a singleton. This is bounded evidence, not a global exact-state minimality theorem."

    e2_evidence = _write_e2_final(
        out, groups, split_witnesses, context_normalization,
        accepted_source_sha256=accepted_source_sha256,
        scientific_design_sha256=plan["scientific_design_sha256"],
        reuse_payload_sha256=auth["reuse_input_payload_sha256"],
        question_sha256=plan["question_sha256"],
    )
    _write_e2_progress(
        out, groups, split_witnesses,
        cursor={"status": "FINAL", "phase_count": phase_count},
        normalization_sha256=context_normalization["normalization_payload_sha256"],
        accepted_source_sha256=accepted_source_sha256,
        scientific_design_sha256=plan["scientific_design_sha256"],
        reuse_payload_sha256=auth["reuse_input_payload_sha256"],
    )

    result = {
        "schema_id": RESULT_SCHEMA, "status": "PASS", "stage_id": STAGE_ID, "substage_id": SUBSTAGE_ID,
        "classification": classification, "accepted_decoder_source_sha256": accepted_source_sha256,
        "internal_execution_id": internal_execution_id, "question_sha256": plan["question_sha256"],
        "scientific_design_sha256": plan["scientific_design_sha256"], "a6_result_sha256": auth["a6_result_sha256"],
        "reuse_input_payload_sha256": auth["reuse_input_payload_sha256"],
        "prior_recomputation": {"s1_census_regenerated": False, "higher_holdout_regenerated": False, "depth1_recomputed": False, "carrier_input_mode": "CERTIFIED_REPLAY_ARTIFACT"},
        "observer": {"id": plan["observer"]["id"], "branch_semantics": plan["observer"]["branch_semantics"], "ordinary_context_count": 248, "execution_context_count": EXECUTION_CONTEXT_COUNT, "operator_count": 31, "execution_context_normalization": FACTOR_SWAP_NORMALIZATION_ID, "normalization_rule": "RIGHT(seed,(a,b))->LEFT(seed,(b,a))", "normalization_lossless": True, "inner_prefix_schedule_execution_only": list(INNER_PREFIX_SCHEDULE), "monotone_lazy_refinement": True, "skipped_after_singleton_is_exact": True, "execution_strategy": EXECUTION_STRATEGY_ID, "global_child_task_universe_materialized": False, "primary_chaining": True, "automatic_cold_replay": False},
        "initial_depth1_collision_stats": initial, "final_depth2_stats": final_stats,
        "prefix_rounds": prefix_rounds, "phase_count": phase_count, "parent_component_task_evaluations": task_evaluations,
        "e2_evidence": e2_evidence,
        "full_observer_reached_for_survivors": full_for_survivors, "discrete_proved_early_by_monotonicity": bool(discrete_early and not full_for_survivors),
        "marker_used": False, "q_D_used": False, "observer_decode_used": False, "exact_parent_reconstruction_used": False,
        "exact_child_identity_in_signature": False, "sampling_used": False, "posthoc_member_budget_used": False,
        "all_a6_collision_members_covered": True, "nontrivial_quotient_candidate_present": candidate,
        "recursive_congruence_all_finite_terms_earned": False, "global_minimality_earned": False,
        "arbitrary_g5_leaf_scope_earned": False, "g6_graduated": False, "promotion": False,
        "r_series_started": False, "next_authorized": next_authorized, "nonclaims": plan["nonclaims"],
        "scientific_interpretation": interpretation,
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(out / "G6_S7_DEPTH2_COMPLETION_RESULT.json", result)
    (out / "READ_FIRST.txt").write_text(
        "INFINITY GRID — G6:S7 DEPTH-2 COMPLETION\n\n" + classification +
        "\n\nPrior S1/higher/depth-1 recomputation: NO\nSampling: NO\nPost-hoc member budget: NO\n"
        "Marker/q_D/decode: NOT USED\nR series: NOT STARTED\nG6 graduated: NO\nResult SHA-256: " + result["result_sha256"] + "\n",
        encoding="utf-8",
    )
    return result
