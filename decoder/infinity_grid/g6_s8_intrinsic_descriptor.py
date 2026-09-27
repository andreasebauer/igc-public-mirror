from __future__ import annotations

"""Registered G6:S8 intrinsic-descriptor and deeper-congruence executor.

S8 remains in the S series. It consumes only certified S7 evidence and the
frozen ordinary action basis. Exact state identities are provenance handles;
they never enter scientific descriptor signatures. Deeper refinement is a
monotone refinement of the S7 partition, evaluated one factor-swap-normalized
ordinary outer context at a time so that a first valid split can be made
durable before further work is attempted.
"""

from collections import Counter, defaultdict
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, canonical_text, write_json_atomic
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec, StageRuntimeError

STAGE_ID = "G6:S8"
PLAN_SCHEMA = "IG_G6_S8_ACCEPTED_PREREGISTRATION_V2"
RESULT_SCHEMA = "IG_G6_S8_COMPLETION_RESULT_V1"
EVALUATOR = "infinity_grid.g6_s8_evaluators:s8_recursive_outer_component_evaluator"
CLASS_PREFIX_EVALUATOR = "infinity_grid.g6_s8_evaluators:s8_s7_class_recursive_prefix_comparator_evaluator"
S7_RESULT_SCHEMA = "IG_G6_S7_DEPTH2_COMPLETION_RESULT_V1"
S7_MEMBERSHIP_SCHEMA = "IG_G6_S7_DEPTH2_E2_MEMBERSHIPS_V1"
S7_SURVIVOR_SCHEMA = "IG_G6_S7_DEPTH2_E2_SURVIVORS_V1"
S7_MANIFEST_SCHEMA = "IG_G6_S7_DEPTH2_E2_MANIFEST_V1"
S7_CONTEXT_SCHEMA = "IG_G6_S7_DEPTH2_E2_CONTEXT_NORMALIZATION_V1"
REUSE_SCHEMA = "IG_G6_S7_A6_DEPTH1_CERTIFIED_REUSE_INPUT_V1"


class G6S8Error(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_plan(plan: Mapping[str, Any], *, accepted_source_sha256: str) -> dict[str, Any]:
    required = {
        "schema_id", "stage_id", "date", "status", "question", "authority", "panels",
        "observer", "descriptor_grammar", "forbidden_features", "deeper_refinement",
        "execution_requirements", "registered_outcomes", "stopping_rules", "nonclaims",
        "scientific_design_sha256", "question_sha256",
    }
    if type(plan) is not dict or set(plan) != required:
        raise G6S8Error("S8_PLAN_FIELDS")
    if plan.get("schema_id") != PLAN_SCHEMA or plan.get("stage_id") != STAGE_ID:
        raise G6S8Error("S8_PLAN_SCHEMA")
    if canonical_sha256({k: v for k, v in plan.items() if k != "question_sha256"}) != plan["question_sha256"]:
        raise G6S8Error("S8_PLAN_HASH")
    design_keys = (
        "stage_id", "question", "panels", "observer", "descriptor_grammar", "forbidden_features",
        "deeper_refinement", "execution_requirements", "registered_outcomes", "stopping_rules", "nonclaims",
    )
    if canonical_sha256({k: plan[k] for k in design_keys}) != plan["scientific_design_sha256"]:
        raise G6S8Error("S8_SCIENTIFIC_DESIGN_HASH")

    auth = plan["authority"]
    sha_keys = (
        "s7_result_sha256", "s8_specification_sha256", "parent_decoder_source_sha256",
        "executor_decoder_source_sha256", "executor_decoder_package_sha256",
        "s7_result_file_sha256", "s7_manifest_file_sha256", "s7_memberships_file_sha256",
        "s7_survivors_file_sha256", "s7_context_normalization_file_sha256",
        "reuse_input_file_sha256", "reuse_input_payload_sha256",
    )
    for key in sha_keys:
        value = auth.get(key)
        if type(value) is not str or len(value) != 64:
            raise G6S8Error("S8_AUTHORITY_SHA:" + key)
    if auth["executor_decoder_source_sha256"] != accepted_source_sha256:
        raise G6S8Error("S8_EXECUTOR_SOURCE_BINDING")
    refs = tuple(str(x) for x in auth.get("frozen_seed_refs", ()))
    if refs != tuple(sorted(refs)) or len(refs) != 4:
        raise G6S8Error("S8_FROZEN_SEED_REFS")

    depths = tuple(int(x) for x in plan["deeper_refinement"].get("depth_schedule", ()))
    if depths != (3, 4, 5, 6):
        raise G6S8Error("S8_FROZEN_DEPTH_SCHEDULE")
    if plan["deeper_refinement"].get("no_posthoc_extension") is not True:
        raise G6S8Error("S8_POSTHOC_DEPTH_FORBIDDEN")
    if plan["deeper_refinement"].get("stop_on_first_valid_split") is not True:
        raise G6S8Error("S8_SPLIT_STOP_RULE")

    req = plan["execution_requirements"]
    for key in (
        "all_4711_members_required", "all_1199_non_singleton_classes_required",
        "all_31_ordinary_operators_required_for_claimed_scope", "resume_only_missing",
        "atomic_local_checkpoints", "drive_incremental_save_required",
    ):
        if req.get(key) is not True:
            raise G6S8Error("S8_EXECUTION_REQUIREMENT:" + key)
    for key in ("sampling_allowed", "posthoc_member_budget_allowed"):
        if req.get(key) is not False:
            raise G6S8Error("S8_EXECUTION_FORBIDDEN:" + key)
    if int(req.get("workers", 0)) != 4:
        raise G6S8Error("S8_WORKERS")
    if int(req.get("max_recursive_states_per_task", 0)) < 1000:
        raise G6S8Error("S8_RECURSIVE_STATE_BUDGET")
    if int(req.get("max_exact_relation_calls_per_task", 0)) < 1000:
        raise G6S8Error("S8_RELATION_CALL_BUDGET")
    if int(req.get("stream_shard_task_limit", 0)) != 1:
        raise G6S8Error("S8_STREAM_SHARD_TASK_LIMIT")
    if int(req.get("max_worker_rss_bytes_per_task", 0)) < 134217728:
        raise G6S8Error("S8_WORKER_RSS_BUDGET")
    if req.get("deeper_member_scope") != "ALL_NON_SINGLETON_MEMBERS_SINGLETONS_VACUOUSLY_STABLE":
        raise G6S8Error("S8_DEEPER_MEMBER_SCOPE")
    if tuple(int(x) for x in req.get("recursive_projection_prefix_schedule", ())) != (1, 8, 32, 124):
        raise G6S8Error("S8_RECURSIVE_PREFIX_SCHEDULE")
    if int(req.get("class_batch_size", 0)) != 16:
        raise G6S8Error("S8_CLASS_BATCH_SIZE")
    if req.get("class_order") != "DESCENDING_S7_CLASS_SIZE_THEN_CLASS_ID":
        raise G6S8Error("S8_CLASS_ORDER")
    if req.get("recursive_projection_semantics") != "MONOTONE_EXECUTION_CONTEXT_PROJECTION_P124_FULL_EQUALITY_EQUIVALENT":
        raise G6S8Error("S8_RECURSIVE_PROJECTION_SEMANTICS")

    obs = plan["observer"]
    if obs.get("id") != "ORDINARY_BRANCH_MULTISET_FUTURE_V1":
        raise G6S8Error("S8_OBSERVER_ID")
    if obs.get("branch_semantics") != "ACTION_INDEXED_MULTISET_OF_SUCCESSOR_BLOCKS":
        raise G6S8Error("S8_BRANCH_SEMANTICS")
    if obs.get("operator_count") != 31 or obs.get("scientific_context_count") != 248 or obs.get("execution_context_count") != 124:
        raise G6S8Error("S8_OBSERVER_COUNTS")
    if obs.get("execution_context_normalization") != "FACTOR_SWAP_LEFT_V1":
        raise G6S8Error("S8_NORMALIZATION")
    ops = tuple(tuple(int(y) for y in x) for x in obs.get("operator_basis", ()))
    if len(ops) != 31 or len(set(ops)) != 31:
        raise G6S8Error("S8_OPERATOR_BASIS")
    if any((b, a) not in set(ops) for a, b in ops):
        raise G6S8Error("S8_OPERATOR_TRANSPOSE_CLOSURE")
    return dict(plan)


def _artifact_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_artifact_bindings(plan: Mapping[str, Any], artifacts: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    expected = {
        "s7_result": ("s7_result_file_sha256", S7_RESULT_SCHEMA),
        "s7_manifest": ("s7_manifest_file_sha256", S7_MANIFEST_SCHEMA),
        "s7_memberships": ("s7_memberships_file_sha256", S7_MEMBERSHIP_SCHEMA),
        "s7_survivors": ("s7_survivors_file_sha256", S7_SURVIVOR_SCHEMA),
        "s7_context_normalization": ("s7_context_normalization_file_sha256", S7_CONTEXT_SCHEMA),
        "reuse_input": ("reuse_input_file_sha256", REUSE_SCHEMA),
    }
    if set(artifacts) != set(expected):
        raise G6S8Error("S8_ARTIFACT_SET")
    auth = plan["authority"]
    out: dict[str, dict[str, Any]] = {}
    for logical, (sha_key, schema) in expected.items():
        p = Path(artifacts[logical]).resolve(strict=True)
        if _sha_file(p) != auth[sha_key]:
            raise G6S8Error("S8_ARTIFACT_FILE_HASH:" + logical)
        obj = _artifact_json(p)
        if obj.get("schema_id") != schema:
            raise G6S8Error("S8_ARTIFACT_SCHEMA:" + logical)
        out[logical] = obj
    s7 = out["s7_result"]
    if s7.get("status") != "PASS" or s7.get("result_sha256") != auth["s7_result_sha256"]:
        raise G6S8Error("S8_S7_RESULT_BINDING")
    if s7.get("next_authorized") != "G6:S8_INTRINSIC_DESCRIPTOR_AND_DEEPER_CONGRUENCE_DESIGN":
        raise G6S8Error("S8_NOT_AUTHORIZED_BY_S7")
    manifest = out["s7_manifest"]
    if manifest.get("status") != "PASS" or int(manifest.get("member_count", -1)) != 4711:
        raise G6S8Error("S8_S7_MANIFEST")
    reuse = out["reuse_input"]
    payload = canonical_sha256({k: v for k, v in reuse.items() if k != "reuse_payload_sha256"})
    if reuse.get("reuse_payload_sha256") != payload or payload != auth["reuse_input_payload_sha256"]:
        raise G6S8Error("S8_REUSE_PAYLOAD")
    _validate_context_normalization(plan, out["s7_context_normalization"])
    return out


def _validate_context_normalization(plan: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    if int(context.get("execution_context_count", -1)) != 124 or int(context.get("scientific_context_count", -1)) != 248:
        raise G6S8Error("S8_CONTEXT_COUNTS")
    rows = list(context.get("execution_coordinates") or ())
    if len(rows) != 124:
        raise G6S8Error("S8_CONTEXT_ROWS")
    refs = tuple(str(x) for x in plan["authority"]["frozen_seed_refs"])
    ops = tuple(tuple(int(y) for y in x) for x in plan["observer"]["operator_basis"])
    expected = [(ref, op) for ref in refs for op in ops]
    aliases: list[int] = []
    for idx, (row, want) in enumerate(zip(rows, expected)):
        got = (str(row.get("seed_ref")), tuple(int(x) for x in row.get("operator") or ()))
        if int(row.get("execution_context_index", -1)) != idx or row.get("position") != "LEFT" or got != want:
            raise G6S8Error("S8_CONTEXT_COORDINATE")
        a = tuple(int(x) for x in row.get("scientific_alias_indices") or ())
        if len(a) != 2:
            raise G6S8Error("S8_CONTEXT_ALIAS_ARITY")
        aliases.extend(a)
    if sorted(aliases) != list(range(248)):
        raise G6S8Error("S8_CONTEXT_ALIAS_COVERAGE")


def _reuse_rows(reuse: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for panel, expected in (("s1", 4464), ("higher", 247)):
        groups = list((reuse.get(panel) or {}).get("collision_classes") or ())
        count = 0
        for group in groups:
            d1 = str(group.get("depth1_class_id", ""))
            if not d1:
                raise G6S8Error("S8_REUSE_D1_CLASS:" + panel)
            for member in group.get("members") or ():
                token = str(member.get("state_token", ""))
                if not token or token in rows or type(member.get("state")) is not dict:
                    raise G6S8Error("S8_REUSE_MEMBER:" + panel)
                rows[token] = {
                    "state_token": token,
                    "identity_sha256": str(member.get("identity_sha256", "")),
                    "panel": panel,
                    "state": dict(member["state"]),
                    "certified_depth1_class_id": d1,
                }
                count += 1
        if count != expected:
            raise G6S8Error("S8_REUSE_PANEL_COUNT:" + panel)
    if len(rows) != 4711:
        raise G6S8Error("S8_REUSE_TOTAL_COUNT")
    return rows


def _reference_partition(memberships: Mapping[str, Any], reuse_rows: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    rows = list(memberships.get("rows") or ())
    if int(memberships.get("member_count", -1)) != 4711 or len(rows) != 4711:
        raise G6S8Error("S8_MEMBERSHIP_COUNT")
    ref: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        token = str(row.get("state_token", "")); cls = str(row.get("final_depth2_class_id", "")); panel = str(row.get("panel", ""))
        if token not in reuse_rows or reuse_rows[token]["panel"] != panel or not cls or token in ref:
            raise G6S8Error("S8_MEMBERSHIP_BINDING")
        ref[token] = cls; groups[cls].append(token)
    for cls in groups:
        groups[cls].sort()
    if len(groups) != 1713 or sum(1 for v in groups.values() if len(v) > 1) != 1199:
        raise G6S8Error("S8_REFERENCE_PARTITION_COUNTS")
    return ref, dict(groups)


def _bag(values) -> tuple[tuple[Any, int], ...]:
    c = Counter(values)
    return tuple(sorted(((k, int(v)) for k, v in c.items()), key=lambda x: canonical_text(x[0], pretty=False)))


def _graph_shape(state: Mapping[str, Any]) -> tuple[int, tuple[str, ...], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    n = int(state.get("n", -1)); hs = tuple(str(x) for x in state.get("H_classes") or ())
    edges = tuple(tuple(int(y) for y in x) for x in state.get("edges") or ())
    ops = tuple(tuple(int(y) for y in x) for x in state.get("edge_operators") or ())
    if n < 1 or len(hs) != n or len(edges) != len(ops):
        raise G6S8Error("S8_STATE_SHAPE")
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n and a != b):
            raise G6S8Error("S8_STATE_EDGE")
    return n, hs, edges, ops


def _graph_counts_descriptor(panel: str, state: Mapping[str, Any]) -> tuple[Any, ...]:
    n, hs, edges, ops = _graph_shape(state)
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    return ("GRAPH_COUNTS_V1", panel, n, _bag(hs), _bag(ops), _bag(deg))


def _wl_descriptor(panel: str, state: Mapping[str, Any], *, stable: bool) -> tuple[Any, ...]:
    n, hs, edges, ops = _graph_shape(state)
    adj: list[list[tuple[int, tuple[int, int]]]] = [[] for _ in range(n)]
    for (a, b), op in zip(edges, ops):
        adj[a].append((b, op)); adj[b].append((a, (op[1], op[0])))
    colors = [canonical_sha256(("H", hs[i])) for i in range(n)]
    rounds = n if stable else 1
    prior_partition = None
    actual = 0
    for _ in range(rounds):
        raw = []
        for i in range(n):
            neigh = tuple(sorted(((op, colors[j]) for j, op in adj[i]), key=lambda x: canonical_text(x, pretty=False)))
            raw.append(("WL", colors[i], neigh))
        next_colors = [canonical_sha256(x) for x in raw]
        actual += 1
        groups: dict[str, list[int]] = defaultdict(list)
        for i, color in enumerate(next_colors): groups[color].append(i)
        partition = tuple(sorted(tuple(v) for v in groups.values()))
        colors = next_colors
        if stable and partition == prior_partition:
            break
        prior_partition = partition
    edge_bag = []
    for (a, b), op in zip(edges, ops):
        left = (colors[a], op, colors[b]); right = (colors[b], (op[1], op[0]), colors[a])
        edge_bag.append(min((left, right), key=lambda x: canonical_text(x, pretty=False)))
    base = _graph_counts_descriptor(panel, state)
    return ("LOCAL_WL_STABLE_V1" if stable else "LOCAL_WL1_V1", base, actual, _bag(colors), _bag(edge_bag))


def _descriptor_rows(reuse_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for token in sorted(reuse_rows):
        row = reuse_rows[token]; panel = str(row["panel"]); state = row["state"]
        out[token] = {
            "GRAPH_COUNTS_V1": _graph_counts_descriptor(panel, state),
            "LOCAL_WL1_V1": _wl_descriptor(panel, state, stable=False),
            "LOCAL_WL_STABLE_V1": _wl_descriptor(panel, state, stable=True),
        }
    return out


def _partition_exactness(reference: Mapping[str, str], descriptor_by_token: Mapping[str, Any]) -> dict[str, Any]:
    if set(reference) != set(descriptor_by_token):
        raise G6S8Error("S8_DESCRIPTOR_COVERAGE")
    by_desc: dict[str, list[str]] = defaultdict(list); by_ref: dict[str, list[str]] = defaultdict(list); desc_key: dict[str, str] = {}
    for token in sorted(reference):
        key = canonical_text(descriptor_by_token[token], pretty=False)
        desc_key[token] = key; by_desc[key].append(token); by_ref[reference[token]].append(token)
    under = None
    for key in sorted(by_desc):
        refs: dict[str, list[str]] = defaultdict(list)
        for t in sorted(by_desc[key]): refs[reference[t]].append(t)
        if len(refs) > 1:
            a_ref, b_ref = sorted(refs)[:2]
            under = {"same_descriptor_sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(), "a": sorted(refs[a_ref])[0], "b": sorted(refs[b_ref])[0], "a_s7_class": a_ref, "b_s7_class": b_ref}
            break
    over = None
    for cls in sorted(by_ref):
        buckets: dict[str, list[str]] = defaultdict(list)
        for t in sorted(by_ref[cls]): buckets[desc_key[t]].append(t)
        if len(buckets) > 1:
            ka, kb = sorted(buckets)[:2]
            over = {"s7_class": cls, "a": sorted(buckets[ka])[0], "b": sorted(buckets[kb])[0], "a_descriptor_sha256": hashlib.sha256(ka.encode("utf-8")).hexdigest(), "b_descriptor_sha256": hashlib.sha256(kb.encode("utf-8")).hexdigest()}
            break
    status = "COMPLETE_MATCH" if under is None and over is None else "UNDERREFINES" if under is not None and over is None else "OVERREFINES" if under is None else "FAIL"
    return {"status": status, "s7_class_count": len(by_ref), "descriptor_class_count": len(by_desc), "underrefinement_witness": under, "overrefinement_witness": over}


def _partition_memberships(runtime, phase_id: str) -> dict[str, str]:
    db = runtime._phase_root(phase_id) / "partition.sqlite3"
    if not db.is_file(): raise G6S8Error("S8_PARTITION_DB_MISSING:" + phase_id)
    conn = sqlite3.connect(db)
    try: return {str(t): str(c) for t, c in conn.execute("SELECT task_id,class_token FROM task_results ORDER BY task_id")}
    finally: conn.close()


def _partition_class_evidence(runtime, phase_id: str, class_tokens: set[str]) -> dict[str, dict[str, Any]]:
    if not class_tokens: return {}
    db = runtime._phase_root(phase_id) / "partition.sqlite3"; conn = sqlite3.connect(db)
    try:
        out = {}
        for token in sorted(class_tokens):
            row = conn.execute("SELECT signature_sha256,representative_task_id,size,representative_signature_bytes FROM classes WHERE class_token=?", (token,)).fetchone()
            if row is None or row[3] is None: raise G6S8Error("S8_CLASS_EVIDENCE_MISSING:" + token)
            b = bytes(row[3]); out[token] = {"class_token": token, "signature_sha256": str(row[0]), "representative_task_id": str(row[1]), "phase_class_size": int(row[2]), "representative_signature_size_bytes": len(b), "representative_signature_canonical_utf8": b.decode("utf-8")}
        return out
    finally: conn.close()


def _phase_resource_pause(runtime, phase_id: str) -> dict[str, Any] | None:
    db = runtime._phase_root(phase_id) / "partition.sqlite3"
    if not db.is_file(): return None
    conn = sqlite3.connect(db)
    try:
        for token, raw in conn.execute("SELECT class_token,representative_signature_bytes FROM classes ORDER BY class_token"):
            if raw is not None and b"S8_RESOURCE_PAUSE" in bytes(raw):
                return _partition_class_evidence(runtime, phase_id, {str(token)})[str(token)]
    finally: conn.close()
    return None


def _class_prefix_task(class_id: str, member_tokens: list[str], reuse_rows: Mapping[str, Mapping[str, Any]], *,
                       depth: int, outer_execution_context_index: int, prefix_schedule: tuple[int, ...],
                       basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...],
                       max_states: int, max_relations: int, max_worker_rss_bytes: int) -> TaskSpec:
    members = [{"state_token": t, "state_tree": reuse_rows[t]["state"]} for t in sorted(member_tokens)]
    payload = {
        "s7_class_id": str(class_id), "class_members": members,
        "future_depth": int(depth), "outer_execution_context_index": int(outer_execution_context_index),
        "recursive_prefix_schedule": list(prefix_schedule),
        "basis_refs": list(basis_refs), "operator_basis": [list(x) for x in operators],
        "max_recursive_states": int(max_states), "max_exact_relation_calls": int(max_relations),
        "max_worker_rss_bytes": int(max_worker_rss_bytes),
    }
    return TaskSpec(
        task_id=f"S8CLS-D{depth}-O{outer_execution_context_index:03d}-{canonical_sha256(['S8_CLASS', class_id])[:20]}",
        task_kind="G6_S8_S7_CLASS_RECURSIVE_PREFIX_COMPARISON",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, float(len(members)) * float(max(1, sum(int(r["state"].get("n", 1)) for r in (reuse_rows[t] for t in member_tokens)))) / float(len(members))),
    )


def _phase_task_metrics(runtime, phase_id: str) -> list[dict[str, Any]]:
    db = runtime.runtime_root() / "phases" / phase_id / "partition.sqlite3"
    conn = sqlite3.connect(db)
    try:
        rows = []
        for task_id, metrics_json in conn.execute("SELECT task_id,metrics_json FROM task_results ORDER BY task_id"):
            rows.append({"task_id": str(task_id), "metrics": json.loads(str(metrics_json))})
        return rows
    finally:
        conn.close()


def _ordered_non_singleton_classes(reference_groups: Mapping[str, list[str]]) -> list[tuple[str, list[str]]]:
    rows = [(str(cid), sorted(tokens)) for cid, tokens in reference_groups.items() if len(tokens) > 1]
    rows.sort(key=lambda x: (-len(x[1]), x[0]))
    if len(rows) != 1199:
        raise G6S8Error("S8_NON_SINGLETON_CLASS_COUNT")
    return rows


def _component_task(row: Mapping[str, Any], *, depth: int, outer_execution_context_index: int,
                    basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...],
                    max_states: int, max_relations: int, max_worker_rss_bytes: int) -> TaskSpec:
    payload = {"state_token": row["state_token"], "state_tree": row["state"], "future_depth": int(depth), "outer_execution_context_index": int(outer_execution_context_index), "basis_refs": list(basis_refs), "operator_basis": [list(x) for x in operators], "max_recursive_states": int(max_states), "max_exact_relation_calls": int(max_relations), "max_worker_rss_bytes": int(max_worker_rss_bytes)}
    return TaskSpec(task_id=f"S8D{depth}O{outer_execution_context_index:03d}-{row['panel']}-{row['state_token']}", task_kind=f"G6_S8_RECURSIVE_OUTER_COMPONENT_D{depth}", binding_sha256=canonical_sha256(payload), payload=payload, cost_weight=max(1.0, float(row["state"].get("n", 1)) * (3.0 ** max(0, depth - 2))))


def _task_token_from_component_id(task_id: str, *, depth: int, outer_execution_context_index: int) -> str:
    prefix = f"S8D{depth}O{outer_execution_context_index:03d}-"
    if not task_id.startswith(prefix): raise G6S8Error("S8_COMPONENT_TASK_PREFIX")
    rest = task_id[len(prefix):]; _panel, sep, token = rest.partition("-")
    if not sep or not token: raise G6S8Error("S8_COMPONENT_TASK_PARSE")
    return token


def _first_component_split(reference_groups: Mapping[str, list[str]], active_tokens: set[str], memberships: Mapping[str, str], *, depth: int, outer_execution_context_index: int) -> tuple[dict[str, Any] | None, set[str]]:
    task_class: dict[str, str] = {}
    for tid, cls in memberships.items():
        token = _task_token_from_component_id(tid, depth=depth, outer_execution_context_index=outer_execution_context_index)
        if token in task_class: raise G6S8Error("S8_COMPONENT_DUPLICATE_TOKEN:" + token)
        task_class[token] = str(cls)
    if set(task_class) != active_tokens: raise G6S8Error("S8_COMPONENT_COVERAGE")
    for ref_cls in sorted(reference_groups):
        tokens = [t for t in reference_groups[ref_cls] if t in active_tokens]
        if len(tokens) <= 1: continue
        buckets: dict[str, list[str]] = defaultdict(list)
        for t in sorted(tokens): buckets[task_class[t]].append(t)
        if len(buckets) > 1:
            ca, cb = sorted(buckets)[:2]
            return ({"depth": int(depth), "outer_execution_context_index": int(outer_execution_context_index), "s7_class": ref_cls, "a_state_token": sorted(buckets[ca])[0], "b_state_token": sorted(buckets[cb])[0], "a_component_class_token": ca, "b_component_class_token": cb}, {ca, cb})
    return None, set()


def _compact_phase_summary(phase) -> dict[str, Any]:
    summary = phase.summary; execution = phase.execution_metadata
    return {"task_count": int(summary.get("task_count", summary.get("scientific_task_units", 0))), "class_count": int(summary.get("class_count", summary.get("distinct_class_count", 0))), "collision_class_count": int(summary.get("collision_class_count", summary.get("multi_class_count", 0))), "summary_sha256": canonical_sha256(summary), "execution_sha256": canonical_sha256(execution), "wall_seconds": execution.get("wall_seconds"), "worker_count": execution.get("lease_workers", execution.get("worker_count"))}


def _write_json(out: Path, name: str, obj: Any) -> Path:
    path = out / name; write_json_atomic(path, obj); return path


def _finalize_manifest(out: Path, *, result: Mapping[str, Any]) -> None:
    files = []
    for p in sorted(x for x in out.iterdir() if x.is_file() and x.name not in {"MANIFEST.json", "SHA256SUMS"}):
        files.append({"file_name": p.name, "sha256": _sha_file(p), "size_bytes": p.stat().st_size})
    manifest = {"schema_id": "IG_G6_S8_OUTPUT_MANIFEST_V1", "status": result["status"], "result_sha256": result["result_sha256"], "files": files}
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest); _write_json(out, "MANIFEST.json", manifest)
    rows = [f"{_sha_file(p)}  {p.name}" for p in sorted(x for x in out.iterdir() if x.is_file() and x.name != "SHA256SUMS")]
    (out / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="ascii")


def run_g6_s8(*, plan_path: str | Path, artifacts: Mapping[str, str | Path], output_dir: str | Path,
               accepted_source_sha256: str, internal_execution_id: str, runtime) -> dict[str, Any]:
    require_controller_execution_origin("run_g6_s8")
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    plan = _validate_plan(_artifact_json(plan_path), accepted_source_sha256=accepted_source_sha256)
    resolved = _validate_artifact_bindings(plan, artifacts)
    reuse_rows = _reuse_rows(resolved["reuse_input"])
    reference, reference_groups = _reference_partition(resolved["s7_memberships"], reuse_rows)

    survivors = resolved["s7_survivors"]
    if int(survivors.get("survivor_class_count", -1)) != 1199: raise G6S8Error("S8_SURVIVOR_COUNT")
    survivor_ids = {str(x.get("final_depth2_class_id")) for x in survivors.get("survivors") or ()}
    if survivor_ids != {k for k, v in reference_groups.items() if len(v) > 1}: raise G6S8Error("S8_SURVIVOR_MEMBERSHIP_BINDING")

    _write_json(out, "G6_S8_ACCEPTED_PREREGISTRATION.json", plan)

    desc_rows = _descriptor_rows(reuse_rows)
    descriptor_candidates = {
        "schema_id": "IG_G6_S8_DESCRIPTOR_CANDIDATES_V1", "member_count": 4711,
        "candidate_ids": ["GRAPH_COUNTS_V1", "LOCAL_WL1_V1", "LOCAL_WL_STABLE_V1", "ORDINARY_D1_MULTIPLICITY_V1", "ORDINARY_RECURSIVE_SIGNATURE_V1"],
        "forbidden_features_absent": True,
        "definitions": {
            "GRAPH_COUNTS_V1": "panel + node count + H-class bag + edge-operator bag + degree bag",
            "LOCAL_WL1_V1": "GRAPH_COUNTS_V1 plus one content-addressed operator-labelled identity-free color-refinement round",
            "LOCAL_WL_STABLE_V1": "content-addressed operator-labelled identity-free color refinement to stable vertex partition plus color-edge incidence bag",
            "ORDINARY_D1_MULTIPLICITY_V1": "certified inherited S7 A6 depth-1 ordinary multiplicity partition; not recomputed and not promoted as a new intrinsic graph invariant",
            "ORDINARY_RECURSIVE_SIGNATURE_V1": "monotone S7 partition refined by each action-indexed multiset of lower-depth ordinary signatures",
        },
    }
    descriptor_candidates["payload_sha256"] = canonical_sha256(descriptor_candidates); _write_json(out, "G6_S8_DESCRIPTOR_CANDIDATES.json", descriptor_candidates)

    audits: dict[str, Any] = {}; minimal_witnesses: list[dict[str, Any]] = []
    for candidate in ("GRAPH_COUNTS_V1", "LOCAL_WL1_V1", "LOCAL_WL_STABLE_V1"):
        audit = _partition_exactness(reference, {t: desc_rows[t][candidate] for t in desc_rows}); audits[candidate] = audit
        for kind in ("underrefinement_witness", "overrefinement_witness"):
            if audit.get(kind) is not None: minimal_witnesses.append({"candidate_id": candidate, "witness_kind": kind, **audit[kind]})
    audits["ORDINARY_D1_MULTIPLICITY_V1"] = {"status": "INHERITED_CERTIFIED_PARTITION_NOT_RECOMPUTED", "source": "S7_A6_CERTIFIED_REUSE", "member_count": 4711}
    exactness_obj = {"schema_id": "IG_G6_S8_DESCRIPTOR_EXACTNESS_AUDIT_V1", "member_count": 4711, "s7_class_count": 1713, "candidate_results": audits}
    exactness_obj["payload_sha256"] = canonical_sha256(exactness_obj); _write_json(out, "G6_S8_DESCRIPTOR_EXACTNESS_AUDIT.json", exactness_obj)

    active_tokens = {t for _cls, ts in reference_groups.items() if len(ts) > 1 for t in ts}
    if len(active_tokens) != 4197: raise G6S8Error("S8_ACTIVE_MEMBER_COUNT")
    singleton_tokens = set(reference) - active_tokens
    if len(singleton_tokens) != 514: raise G6S8Error("S8_SINGLETON_MEMBER_COUNT")
    auth = plan["authority"]; basis_refs = tuple(str(x) for x in auth["frozen_seed_refs"])
    operators = tuple(tuple(int(y) for y in x) for x in plan["observer"]["operator_basis"])
    max_states = int(plan["execution_requirements"]["max_recursive_states_per_task"]); max_relations = int(plan["execution_requirements"]["max_exact_relation_calls_per_task"])
    stream_shard_task_limit = int(plan["execution_requirements"]["stream_shard_task_limit"])
    max_worker_rss_bytes = int(plan["execution_requirements"]["max_worker_rss_bytes_per_task"])
    depth_progress: list[dict[str, Any]] = []; late_split = None; resource_pause = None; stable_depths: list[int] = []
    prefix_schedule = tuple(int(x) for x in plan["execution_requirements"]["recursive_projection_prefix_schedule"])
    class_batch_size = int(plan["execution_requirements"]["class_batch_size"])
    ordered_classes = _ordered_non_singleton_classes(reference_groups)
    expected_active = sum(len(tokens) for _, tokens in ordered_classes)
    if expected_active != 4197:
        raise G6S8Error("S8_ACTIVE_NON_SINGLETON_MEMBER_COUNT")

    # Gen32 class-first recursive projection execution.  A task compares a whole
    # certified S7 class and can stop internally at P001/P008/P032.  P124 is an
    # equality-equivalent representation of the full factor-swap-normalized
    # recursive signature.  Batches are execution-only and deterministic.
    for depth in tuple(int(x) for x in plan["deeper_refinement"]["depth_schedule"]):
        context_rows: list[dict[str, Any]] = []
        for outer_index in range(124):
            batch_rows: list[dict[str, Any]] = []
            for batch_start in range(0, len(ordered_classes), class_batch_size):
                batch = ordered_classes[batch_start:batch_start + class_batch_size]
                tasks = [
                    _class_prefix_task(
                        cid, tokens, reuse_rows, depth=depth,
                        outer_execution_context_index=outer_index, prefix_schedule=prefix_schedule,
                        basis_refs=basis_refs, operators=operators, max_states=max_states,
                        max_relations=max_relations, max_worker_rss_bytes=max_worker_rss_bytes,
                    )
                    for cid, tokens in batch
                ]
                batch_index = batch_start // class_batch_size
                phase_id = f"G6_S8_D{depth}_O{outer_index:03d}_CB{batch_index:03d}"
                try:
                    phase = runtime.run_structural_partition(
                        phase_id=phase_id, tasks=tasks, evaluator_ref=CLASS_PREFIX_EVALUATOR,
                        requested_workers=4, max_tasks=len(tasks), stream_shard_task_limit=stream_shard_task_limit,
                    )
                except StageRuntimeError as exc:
                    text = str(exc).upper()
                    if any(x in text for x in ("RESOURCE", "MEMORY", "WORKSPACE", "NO_SPACE", "TIMEOUT", "LEASE")):
                        resource_pause = {"depth": depth, "outer_execution_context_index": outer_index,
                                          "class_batch_index": batch_index, "reason": str(exc), "phase_id": phase_id}
                        break
                    raise
                sentinel = _phase_resource_pause(runtime, phase_id)
                if sentinel is not None:
                    resource_pause = {"depth": depth, "outer_execution_context_index": outer_index,
                                      "class_batch_index": batch_index, "reason": "EVALUATOR_RESOURCE_SENTINEL",
                                      "phase_id": phase_id, "evidence": sentinel}
                    break
                rows = _phase_task_metrics(runtime, phase_id)
                split_rows = [r for r in rows if (r.get("metrics") or {}).get("split_found") is True]
                if split_rows:
                    r = sorted(split_rows, key=lambda x: x["task_id"])[0]
                    m = dict(r["metrics"])
                    late_split = {
                        "depth": int(depth), "outer_execution_context_index": int(outer_index),
                        "class_batch_index": int(batch_index), "phase_id": phase_id,
                        "s7_class": str(m["s7_class_id"]),
                        "a_state_token": str(m["a_state_token"]), "b_state_token": str(m["b_state_token"]),
                        "recursive_prefix_count": int(m["recursive_prefix_count"]),
                        "structural_separator": [m["separator_a_signature"], m["separator_b_signature"]],
                        "projection_semantics": "MONOTONE_EXECUTION_CONTEXT_PROJECTION",
                        "p124_full_equality_equivalent": True,
                    }
                    late_split["witness_sha256"] = canonical_sha256(late_split)
                    minimal_witnesses.append({"candidate_id": "ORDINARY_RECURSIVE_SIGNATURE_V1",
                                              "witness_kind": "late_split", **late_split})
                batch_rows.append({
                    "batch_index": batch_index, "phase_id": phase_id,
                    "class_count": len(batch), "member_count": sum(len(t) for _, t in batch),
                    "partition": _compact_phase_summary(phase), "late_split": late_split,
                })
                progress = {
                    "schema_id": "IG_G6_S8_DEEPER_REFINEMENT_PROGRESS_V1", "depth_schedule": [3,4,5,6],
                    "completed_depths": [x["depth"] for x in depth_progress], "current_depth": depth,
                    "current_outer_execution_context_index": outer_index,
                    "completed_class_batches_current_context": len(batch_rows),
                    "class_batch_count": (len(ordered_classes) + class_batch_size - 1) // class_batch_size,
                    "recursive_prefix_schedule": list(prefix_schedule),
                    "singleton_member_count_vacuously_stable": 514,
                    "active_non_singleton_member_count": 4197, "depths": depth_progress,
                    "current_depth_contexts": context_rows,
                    "current_context_batches": batch_rows,
                    "resource_pause": resource_pause, "late_split": late_split,
                }
                progress["payload_sha256"] = canonical_sha256(progress)
                _write_json(out, "G6_S8_DEEPER_REFINEMENT_PROGRESS.json", progress)
                if late_split is not None:
                    break
            if resource_pause is not None or late_split is not None:
                break
            expected_batches = (len(ordered_classes) + class_batch_size - 1) // class_batch_size
            if len(batch_rows) != expected_batches:
                raise G6S8Error("S8_CLASS_BATCH_SCHEDULE_INCOMPLETE")
            context_rows.append({
                "outer_execution_context_index": outer_index, "class_batch_count": len(batch_rows),
                "class_count": len(ordered_classes), "active_member_count": 4197,
                "recursive_prefix_schedule": list(prefix_schedule), "status": "NO_SPLIT_RELATIVE_TO_S7",
                "batches": batch_rows,
            })
        if resource_pause is not None or late_split is not None:
            break
        if len(context_rows) != 124:
            raise G6S8Error("S8_CONTEXT_SCHEDULE_INCOMPLETE")
        stable_depths.append(depth)
        depth_progress.append({
            "depth": depth, "context_count": 124, "active_member_count": 4197,
            "singleton_member_count_vacuously_stable": 514,
            "recursive_prefix_schedule": list(prefix_schedule),
            "status": "NO_SPLIT_RELATIVE_TO_S7", "contexts": context_rows,
        })
        progress = {
            "schema_id": "IG_G6_S8_DEEPER_REFINEMENT_PROGRESS_V1", "depth_schedule": [3,4,5,6],
            "completed_depths": stable_depths, "current_depth": None,
            "singleton_member_count_vacuously_stable": 514,
            "active_non_singleton_member_count": 4197, "depths": depth_progress,
            "current_depth_contexts": [], "current_context_batches": [],
            "recursive_prefix_schedule": list(prefix_schedule),
            "resource_pause": resource_pause, "late_split": late_split,
        }
        progress["payload_sha256"] = canonical_sha256(progress)
        _write_json(out, "G6_S8_DEEPER_REFINEMENT_PROGRESS.json", progress)

    graph_exact = [k for k, v in audits.items() if isinstance(v, dict) and v.get("status") == "COMPLETE_MATCH"]
    if resource_pause is not None:
        classification = "E_RESOURCE_PAUSE"; status = "PAUSED"; next_authorized = "RESUME_G6_S8_ONLY_MISSING"
    elif late_split is not None:
        classification = "C_LATE_SPLIT"; status = "PASS"; next_authorized = "G6:S8_DESCRIPTOR_REFINEMENT_AFTER_LATE_SPLIT"
    else:
        if stable_depths != [3,4,5,6]: raise G6S8Error("S8_DEPTH_SCHEDULE_INCOMPLETE_WITHOUT_PAUSE")
        classification = "A_DESCRIPTOR_AND_STABLE_CONGRUENCE_CANDIDATE" if graph_exact else "B_STABLE_WITHOUT_INTRINSIC_DESCRIPTOR"
        status = "PASS"; next_authorized = "G6:S8_PROOF_FOCUSED_SUCCESSOR" if classification.startswith("A_") else "G6:S8_DESCRIPTOR_SEARCH_SUCCESSOR"

    congruence = {
        "schema_id": "IG_G6_S8_DEEPER_CONGRUENCE_RESULT_V1", "status": status, "classification": classification,
        "depth_schedule": [3,4,5,6], "completed_depths": stable_depths, "late_split": late_split, "resource_pause": resource_pause,
        "monotone_partition_rule": "START_FROM_S7_DEPTH2_PARTITION_AND_SPLIT_ONLY",
        "singleton_handling": "514_SINGLETON_CLASSES_VACUOUSLY_STABLE_NO_DEEP_TASK_REQUIRED",
        "operator_compatibility": {"operator_count": 31, "scientific_context_count": 248, "execution_context_count": 124, "seed_count": 4, "basis_positions": ["LEFT", "RIGHT"], "execution_normalization": "FACTOR_SWAP_LEFT_V1", "scope": "FROZEN_ORDINARY_OPERATOR_BASIS_ON_CERTIFIED_S7_MEMBERS", "status": "BOUNDED_CHECK_COMPLETE" if status == "PASS" and late_split is None else "STOPPED_ON_SPLIT" if late_split else "PAUSED", "all_depth_theorem_earned": False},
    }
    congruence["payload_sha256"] = canonical_sha256(congruence); _write_json(out, "G6_S8_DEEPER_CONGRUENCE_RESULT.json", congruence)

    witness_obj = {"schema_id": "IG_G6_S8_MINIMAL_WITNESSES_V1", "witness_count": len(minimal_witnesses), "witnesses": minimal_witnesses}; witness_obj["payload_sha256"] = canonical_sha256(witness_obj); _write_json(out, "G6_S8_MINIMAL_WITNESSES.json", witness_obj)
    obligations = {"schema_id": "IG_G6_S8_PROOF_OBLIGATIONS_V1", "classification": classification, "obligations": [
        {"id": "PO-01", "statement": "Prove the recursive ordinary signature is invariant under the frozen anonymous-interface quotient.", "status": "OPEN_FOR_FORMAL_PROOF"},
        {"id": "PO-02", "statement": "Prove factor-swap normalization preserves all 248 scientific contexts at every recursive depth.", "status": "INHERITED_COMPUTATIONAL_PREMISE_REQUIRES_S8_PROOF_REFERENCE"},
        {"id": "PO-03", "statement": "If no split occurs through depth 6, prove a closure/induction principle sufficient to extend equality to every finite ordinary context.", "status": "OPEN"},
        {"id": "PO-04", "statement": "Prove any promoted finite descriptor has an operator update law independent of exact child identity.", "status": "OPEN"},
        {"id": "PO-05", "statement": "Independent replay must reproduce source, manifest and science result hashes before theorem promotion.", "status": "REQUIRED"}], "all_finite_congruence_theorem_earned": False, "g6_graduated": False}
    obligations["payload_sha256"] = canonical_sha256(obligations); _write_json(out, "G6_S8_PROOF_OBLIGATIONS.json", obligations)

    theorem_text = ("# G6:S8 theorem candidate\n\n" f"Status: {classification}\n\n" "Candidate statement only: the S7 ordinary depth-2 partition remains unchanged by every preregistered depth-3 through depth-6 ordinary action-indexed multiset refinement in the frozen carrier scope, if and only if the completed evidence reports no late split.\n\n" "Finite execution is evidence, not an all-finite proof. G6 is not graduated by this file.\n")
    (out / "G6_S8_THEOREM_CANDIDATE.md").write_text(theorem_text, encoding="utf-8")

    completion = {"schema_id": RESULT_SCHEMA, "status": status, "stage_id": STAGE_ID, "classification": classification, "scope": "CERTIFIED_S7_4711_MEMBER_PANEL_AND_FROZEN_31_OPERATOR_ORDINARY_BASIS", "accepted_decoder_source_sha256": accepted_source_sha256, "internal_execution_id": internal_execution_id, "question_sha256": plan["question_sha256"], "scientific_design_sha256": plan["scientific_design_sha256"], "descriptor": {"intrinsic_graph_candidate_results": audits, "exact_intrinsic_graph_candidates": graph_exact, "recursive_signature_id": "ORDINARY_RECURSIVE_SIGNATURE_V1", "stable_bounded_depths": stable_depths}, "exactness": "MONOTONE_S7_PARTITION_UNCHANGED_AT_ALL_COMPLETED_PREREGISTERED_DEPTHS" if stable_depths else "NO_DEEPER_STABILITY_PROMOTED", "operator_compatibility": congruence["operator_compatibility"], "depth_evidence": {"scheduled": [3,4,5,6], "completed": stable_depths, "late_split": late_split, "resource_pause": resource_pause}, "proof_status": "OPEN_PROOF_OBLIGATIONS_NO_ALL_FINITE_THEOREM", "g6_graduated": False, "global_minimality_earned": False, "all_finite_congruence_theorem_earned": False, "marker_used": False, "q_D_used": False, "observer_decode_used": False, "exact_parent_reconstruction_used": False, "sampling_used": False, "r_series_started": False, "nonclaims": plan["nonclaims"], "next_authorized": next_authorized}
    completion["result_sha256"] = canonical_sha256(completion); _write_json(out, "G6_S8_COMPLETION_RESULT.json", completion)
    (out / "READ_FIRST.txt").write_text("INFINITY GRID — G6:S8 INTRINSIC DESCRIPTOR AND DEEPER CONGRUENCE\n\n" f"Status: {status}\nClassification: {classification}\nResult SHA-256: {completion['result_sha256']}\n\n" "No marker, q_D, observer decode, exact-parent reconstruction, sampling or R-series operation was used.\nFinite deeper evidence is not an all-finite congruence theorem and does not graduate G6.\n", encoding="utf-8")
    _finalize_manifest(out, result=completion); return completion
