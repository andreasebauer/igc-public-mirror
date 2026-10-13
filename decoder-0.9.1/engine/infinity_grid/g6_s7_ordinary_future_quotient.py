from __future__ import annotations

"""Registered G6:S7 ordinary branch-sensitive public quotient reconnaissance.

S7 is pre-R and marker/reconstruction free.  It regenerates the complete exact
S1 census, freezes the inherited public state D=(f,m), and asks whether ordinary
31-operator composition exposes a smaller behavioral quotient than exact state
when branching is observed as complete action-indexed multisets of lower-depth
behavioral blocks.  Exact child identity is used only transiently by the exact
relation kernel and never appears in a scientific signature.

The registered run is deliberately bounded.  It computes depth-1 branch
multiset signatures for the full S1 census and an independently fixed higher
holdout, then refines *all* members of non-singleton classes to depth 2 when the
predeclared collision-member budget permits.  It cannot graduate G6 or prove an
all-finite congruence theorem.
"""

import hashlib
import itertools
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

STAGE_ID = "G6:S7"
PLAN_SCHEMA = "IG_G6_S7_ORDINARY_BRANCH_CONGRUENCE_PREREGISTRATION_V3"
RESULT_SCHEMA = "IG_G6_S7_ORDINARY_BRANCH_CONGRUENCE_RESULT_V2"
GEN_S1_EVALUATOR = "infinity_grid.g6_controller_evaluators:g6_s1_universe_generation_evaluator"
GEN_HIGHER_EVALUATOR = "infinity_grid.g6_controller_evaluators:g6_s3_axis_a_generation_evaluator"
PUBLIC_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_public_state_evaluator"
BRANCH_COMPONENT_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_ordinary_branch_count_component_evaluator"
BRANCH_EVALUATOR = "infinity_grid.g6_s7_evaluators:s7_ordinary_branch_relation_evaluator"


class G6S7Error(RuntimeError):
    pass


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_id", "stage_id", "date", "status", "scientific_role", "question",
        "authority", "separation_rule", "observer", "phases", "budgets",
        "registered_outcomes", "stopping_rule", "next_on_candidate", "next_on_discrete",
        "nonclaims", "preregistration_lineage", "scientific_design_sha256", "question_sha256",
    }
    if type(plan) is not dict or set(plan) != required:
        raise G6S7Error("S7_PLAN_FIELDS")
    if plan.get("schema_id") != PLAN_SCHEMA or plan.get("stage_id") != STAGE_ID:
        raise G6S7Error("S7_PLAN_SCHEMA")
    base = {k: v for k, v in plan.items() if k != "question_sha256"}
    if canonical_sha256(base) != plan["question_sha256"]:
        raise G6S7Error("S7_PLAN_HASH")
    design_keys = (
        "stage_id", "scientific_role", "question", "separation_rule", "observer", "phases",
        "budgets", "registered_outcomes", "stopping_rule", "next_on_candidate",
        "next_on_discrete", "nonclaims",
    )
    design = {k: plan[k] for k in design_keys}
    if canonical_sha256(design) != plan["scientific_design_sha256"]:
        raise G6S7Error("S7_SCIENTIFIC_DESIGN_HASH")
    lineage = plan["preregistration_lineage"]
    if lineage.get("change_type") != "SCIENTIFIC_DESIGN_CORRECTION_BEFORE_FIRST_S7_EXECUTION":
        raise G6S7Error("S7_PREREGISTRATION_LINEAGE")
    if lineage.get("no_prior_s7_science") is not True:
        raise G6S7Error("S7_PRIOR_SCIENCE_GUARD")
    sep = plan["separation_rule"]
    for key in ("marker_operations_allowed", "q_D_allowed_as_science_input", "observer_decode_allowed",
                "exact_parent_reconstruction_allowed", "exact_child_identity_in_signature_allowed"):
        if sep.get(key) is not False:
            raise G6S7Error("S7_SEPARATION_RULE:" + key)
    if sep.get("exact_relation_oracle_allowed") is not True:
        raise G6S7Error("S7_EXACT_RELATION_ORACLE_REQUIRED")
    obs = plan["observer"]
    if obs.get("id") != "ORDINARY_BRANCH_MULTISET_FUTURE_V1" or obs.get("context_count") != 248:
        raise G6S7Error("S7_OBSERVER")
    if obs.get("max_future_depth") != 2 or obs.get("branch_semantics") != "MULTISET_OF_SUCCESSOR_BLOCKS":
        raise G6S7Error("S7_BRANCH_SEMANTICS")
    ops = tuple(tuple(int(y) for y in x) for x in obs.get("operator_basis", ()))
    if len(ops) != 31 or len(set(ops)) != 31:
        raise G6S7Error("S7_OPERATOR_BASIS")
    refs = tuple(str(x) for x in plan["authority"].get("frozen_seed_refs", ()))
    if tuple(sorted(refs)) != refs or len(refs) != 4:
        raise G6S7Error("S7_SEED_REFS")
    return dict(plan)


def _tree_task(prefix: str, row: Mapping[str, Any], *, depth: int,
               basis_refs: tuple[str, ...], operators: tuple[tuple[int, int], ...]) -> TaskSpec:
    state = row["state"]
    n = int(state.get("n", 1))
    token = str(row["state_token"])
    payload = {
        "state_token": token,
        "state_tree": state,
        "future_depth": int(depth),
        "basis_refs": list(basis_refs),
        "operator_basis": [list(x) for x in operators],
    }
    return TaskSpec(
        task_id=f"{prefix}-{token}",
        task_kind=f"G6_S7_ORDINARY_BRANCH_MULTISET_D{int(depth)}",
        binding_sha256=canonical_sha256(payload),
        payload=payload,
        cost_weight=max(1.0, float(n) * (1.0 if depth == 1 else 8.0)),
    )


def _partition_memberships(runtime, phase_id: str) -> dict[str, str]:
    db = runtime._phase_root(phase_id) / "partition.sqlite3"
    if not db.is_file():
        raise G6S7Error("S7_PARTITION_DB_MISSING:" + phase_id)
    conn = sqlite3.connect(db)
    try:
        return {str(t): str(c) for t, c in conn.execute("SELECT task_id,class_token FROM task_results ORDER BY task_id")}
    finally:
        conn.close()


def _component_batch_task(prefix: str, row: Mapping[str, Any], *, batch_index: int,
                          contexts: list[tuple[str, tuple[int, int], str]],
                          basis_refs: tuple[str, ...]) -> TaskSpec:
    token = str(row["state_token"]); state = row["state"]
    payload = {
        "state_token": token, "state_tree": state, "basis_refs": list(basis_refs),
        "contexts": [
            {"basis_ref": ref, "operator": list(op), "position": pos}
            for ref, op, pos in contexts
        ],
        "batch_index": int(batch_index),
    }
    return TaskSpec(
        task_id=f"{prefix}B{int(batch_index):02d}-{token}",
        task_kind="G6_S7_D1_BRANCH_MULTIPLICITY_COMPONENT_BATCH",
        binding_sha256=canonical_sha256(payload), payload=payload,
        cost_weight=max(1.0, float(state.get("n", 1)) * len(contexts)),
    )


def _lazy_depth1_partition(runtime, *, panel_tag: str, rows: list[Mapping[str, Any]],
                           public_tasks: list[TaskSpec], public_phase_id: str,
                           base_task_prefix: str, basis_refs: tuple[str, ...],
                           operators: tuple[tuple[int, int], ...], member_budget: int,
                           max_tasks: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute the frozen 248-component depth-1 partition by monotone lazy refinement.

    Contexts are processed in the original frozen order in execution-only batches of
    at most eight. Later batches are skipped only after a class is already singleton;
    omitted components cannot merge a singleton. Intersecting the exact batch
    partitions is therefore exactly equivalent to materializing the full observer.
    """
    row_by_token = {str(r["state_token"]): r for r in rows}
    if len(row_by_token) != len(rows):
        raise G6S7Error("S7_LAZY_DUPLICATE_STATE_TOKEN")
    pub_members = _partition_memberships(runtime, public_phase_id)
    initial: dict[str, list[str]] = {}
    for task in public_tasks:
        token = str(task.payload["state_token"])
        cls = pub_members.get(task.task_id)
        if cls is None:
            raise G6S7Error("S7_LAZY_PUBLIC_MEMBERSHIP_MISSING")
        initial.setdefault(cls, []).append(token)
    groups = [sorted(v) for _k, v in sorted(initial.items())]
    contexts = [(ref, op, pos) for ref in basis_refs for op in operators for pos in ("LEFT", "RIGHT")]
    if len(contexts) != 248:
        raise G6S7Error("S7_LAZY_CONTEXT_COUNT")
    batch_size = 8
    batches = [contexts[i:i+batch_size] for i in range(0, len(contexts), batch_size)]
    rounds = []
    component_evaluations = 0
    relation_profile_calls = 0
    contexts_materialized = 0
    for batch_index, batch in enumerate(batches):
        active = sorted(t for g in groups if len(g) > 1 for t in g)
        if not active:
            break
        tasks = [_component_batch_task(
            base_task_prefix, row_by_token[t], batch_index=batch_index,
            contexts=batch, basis_refs=basis_refs,
        ) for t in active]
        phase_id = f"{panel_tag}_D1_COMPONENT_BATCH_{batch_index:02d}"
        res = runtime.run_structural_partition(
            phase_id=phase_id, tasks=tasks, evaluator_ref=BRANCH_COMPONENT_EVALUATOR,
            requested_workers=4, max_tasks=max_tasks,
        )
        memb = _partition_memberships(runtime, phase_id)
        new_groups: list[list[str]] = []
        for g in groups:
            if len(g) <= 1:
                new_groups.append(g); continue
            buckets: dict[str, list[str]] = {}
            for token in g:
                tid = f"{base_task_prefix}B{batch_index:02d}-{token}"
                c = memb.get(tid)
                if c is None:
                    raise G6S7Error("S7_LAZY_COMPONENT_MEMBERSHIP_MISSING")
                buckets.setdefault(c, []).append(token)
            new_groups.extend(sorted((sorted(v) for v in buckets.values()), key=lambda x: x[0]))
        groups = sorted(new_groups, key=lambda x: x[0])
        component_evaluations += len(tasks)
        relation_profile_calls += len(tasks) * len(batch)
        contexts_materialized += len(batch)
        rounds.append({
            "batch_index": batch_index,
            "context_index_start": batch_index * batch_size,
            "context_count": len(batch), "active_state_count": len(tasks),
            "class_count_after": len(groups),
            "multi_class_count_after": sum(1 for g in groups if len(g) > 1),
            "phase_execution": res.execution_metadata,
        })
    multi = [g for g in groups if len(g) > 1]
    member_count = sum(len(g) for g in multi)
    ids = [] if member_count > int(member_budget) else [
        f"{base_task_prefix}-{t}" for g in multi for t in g
    ]
    first = multi[0] if multi else None
    collision = {
        "multi_class_count": len(multi), "collision_member_count": member_count,
        "max_class_size": max((len(g) for g in multi), default=0),
        "member_budget": int(member_budget), "budget_exceeded": member_count > int(member_budget),
        "member_task_ids": ids,
        "first_multi_class": None if first is None else {
            "class_token": "LAZY:" + canonical_sha256(["S7_LAZY_CLASS", first]),
            "size": len(first), "representative_task_id": f"{base_task_prefix}-{first[0]}",
        },
    }
    science = {
        "schema_id": "IG_DECODER_G6_S7_MONOTONE_DEPTH1_PARTITION_V1",
        "phase_id": panel_tag + "_D1_MONOTONE_COMPLETE_OBSERVER",
        "task_count": len(rows), "class_count": len(groups),
        "multi_class_count": len(multi), "max_class_size": max((len(g) for g in groups), default=0),
        "complete_observer_context_count": 248,
        "partition_equivalent_to_full_248_component_observer": True,
        "structural_equality_authority": "EXACT_COMPONENT_SIGNATURE_BYTES_AND_INTERSECTION",
        "hash_equality_never_decides_scientific_equality": True,
    }
    science["summary_sha256"] = canonical_sha256(science)
    execution = {
        "method": "MONOTONE_LAZY_COMPONENT_REFINEMENT",
        "component_batch_size": batch_size,
        "component_batches_materialized": len(rounds),
        "contexts_materialized": contexts_materialized,
        "component_state_batch_evaluations": component_evaluations,
        "exact_relation_profile_calls": relation_profile_calls,
        "full_materialization_exact_relation_calls": len(rows) * 248,
        "saved_exact_relation_profile_calls": len(rows) * 248 - relation_profile_calls,
        "rounds": rounds,
        "completion_order_not_science": True,
        "worker_count_not_science": True,
    }
    return science, execution, collision


def _partition_collision_info(runtime, phase_id: str, *, member_budget: int) -> dict[str, Any]:
    root = runtime._phase_root(phase_id)
    db = root / "partition.sqlite3"
    if not db.is_file():
        raise G6S7Error("S7_PARTITION_DB_MISSING:" + phase_id)
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT COUNT(*),COALESCE(SUM(size),0),COALESCE(MAX(size),0) FROM classes WHERE size>1"
        ).fetchone()
        multi_class_count = int(row[0]); member_count = int(row[1]); max_class_size = int(row[2])
        first = conn.execute(
            "SELECT class_token,size,representative_task_id FROM classes WHERE size>1 ORDER BY class_token LIMIT 1"
        ).fetchone()
        ids: list[str] = []
        if member_count <= int(member_budget):
            ids = [str(r[0]) for r in conn.execute(
                "SELECT tr.task_id FROM task_results tr JOIN classes c ON tr.class_token=c.class_token "
                "WHERE c.size>1 ORDER BY tr.task_id"
            )]
        return {
            "multi_class_count": multi_class_count,
            "collision_member_count": member_count,
            "max_class_size": max_class_size,
            "member_budget": int(member_budget),
            "budget_exceeded": bool(member_count > int(member_budget)),
            "member_task_ids": ids,
            "first_multi_class": None if first is None else {
                "class_token": str(first[0]), "size": int(first[1]), "representative_task_id": str(first[2]),
            },
        }
    finally:
        conn.close()


def _refinement_tasks(base_tasks: list[TaskSpec], collision_ids: list[str], *, prefix: str) -> list[TaskSpec]:
    by_id = {t.task_id: t for t in base_tasks}
    out = []
    for tid in collision_ids:
        if tid not in by_id:
            raise G6S7Error("S7_COLLISION_TASK_NOT_FOUND:" + tid)
        p = dict(by_id[tid].payload)
        p["future_depth"] = 2
        out.append(TaskSpec(
            task_id=f"{prefix}-{tid}",
            task_kind="G6_S7_ORDINARY_BRANCH_MULTISET_D2",
            binding_sha256=canonical_sha256(p), payload=p,
            cost_weight=max(8.0, float(p["state_tree"].get("n", 1)) * 8.0),
        ))
    return out


def _panel_disposition(o1_summary: Mapping[str, Any], o2_summary: Mapping[str, Any] | None,
                       collision: Mapping[str, Any]) -> str:
    if int(o1_summary["multi_class_count"]) == 0:
        return "DISCRETE_AT_DEPTH1"
    if collision.get("budget_exceeded"):
        return "DEPTH2_BUDGET_EXCEEDED"
    if o2_summary is None:
        raise G6S7Error("S7_DEPTH2_MISSING")
    return "NONTRIVIAL_SURVIVES_DEPTH2" if int(o2_summary["multi_class_count"]) > 0 else "DISCRETE_BY_DEPTH2"


def run_g6_s7(*, plan_path: str | Path, artifacts: Mapping[str, str | Path], output_dir: str | Path,
               accepted_source_sha256: str, internal_execution_id: str, runtime) -> dict[str, Any]:
    require_controller_execution_origin("g6-s7-ordinary-branch-congruence")
    plan_path = Path(plan_path).resolve(strict=True)
    plan = _validate_plan(json.loads(plan_path.read_text(encoding="utf-8")))
    paths = {k: Path(v).resolve(strict=True) for k, v in artifacts.items()}

    auth = plan["authority"]
    if accepted_source_sha256 != auth["executor_decoder_source_sha256"]:
        raise G6S7Error("S7_SOURCE_BINDING")
    if _sha_file(paths["master_prereg"]) != auth["g6_master_file_sha256"]:
        raise G6S7Error("S7_MASTER_BINDING")
    if _sha_file(paths["post_r0_handoff"]) != auth["post_r0_handoff_file_sha256"]:
        raise G6S7Error("S7_HANDOFF_BINDING")

    basis_refs = tuple(str(x) for x in auth["frozen_seed_refs"])
    operators = tuple(tuple(int(y) for y in x) for x in plan["observer"]["operator_basis"])
    budgets = plan["budgets"]

    # Phase 1: complete exact S1 state regeneration from all 4x4x31 ordinary seed compositions.
    gen_tasks = []
    for l in basis_refs:
        for r in basis_refs:
            for op in operators:
                payload = {"left_ref": l, "right_ref": r, "operator": list(op)}
                gen_tasks.append(TaskSpec(
                    task_id=f"S1-{l}-{r}-{int(op[0])}-{int(op[1])}",
                    task_kind="G6_S7_S1_EXACT_REGEN",
                    binding_sha256=canonical_sha256(payload), payload=payload, cost_weight=1.0,
                ))
    if len(gen_tasks) != 496:
        raise G6S7Error("S7_S1_TASK_COUNT")
    s1_gen = runtime.run_content_indexed_generation(
        phase_id="S7_S1_EXACT_CENSUS", tasks=gen_tasks, evaluator_ref=GEN_S1_EVALUATOR,
        requested_workers=4, max_tasks=496, max_generated_occurrences=25000,
    )
    s1_count = int(s1_gen.summary["distinct_exact_state_count"])
    if s1_count != 4520:
        classification = "AUTHORITY_OR_REGENERATION_MISMATCH"
        result = {
            "schema_id": RESULT_SCHEMA, "status": "REVIEW_REQUIRED", "stage_id": STAGE_ID,
            "classification": classification, "accepted_decoder_source_sha256": accepted_source_sha256,
            "internal_execution_id": internal_execution_id, "question_sha256": plan["question_sha256"],
            "s1_generation": {"science": s1_gen.summary, "execution": s1_gen.execution_metadata},
            "expected_s1_exact_state_count": 4520, "observed_s1_exact_state_count": s1_count,
            "marker_used": False, "q_D_used": False, "observer_decode_used": False,
            "exact_child_identity_in_signature": False, "g6_graduated": False,
            "promotion": False, "r_series_started": False, "next_authorized": None,
            "nonclaims": plan["nonclaims"],
        }
        result["result_sha256"] = canonical_sha256(result)
        out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
        write_json_atomic(out / "G6_S7_RESULT.json", result)
        return result

    s1_rows = list(runtime.iter_generated_states(phase_id="S7_S1_EXACT_CENSUS"))
    s1_public_tasks = [_tree_task("S1PUB", r, depth=1, basis_refs=basis_refs, operators=operators) for r in s1_rows]
    # Public-only evaluator ignores future-depth/action fields but task bindings stay fully declared.
    s1_public = runtime.run_structural_partition(
        phase_id="S7_S1_INHERITED_PUBLIC_PARTITION", tasks=s1_public_tasks,
        evaluator_ref=PUBLIC_EVALUATOR, requested_workers=4, max_tasks=4520,
    )
    s1_d1_tasks = [_tree_task("S1D1", r, depth=1, basis_refs=basis_refs, operators=operators) for r in s1_rows]
    s1_d1_science, s1_d1_execution, s1_collision = _lazy_depth1_partition(
        runtime, panel_tag="S7_S1_BRANCH_MULTISET", rows=s1_rows, public_tasks=s1_public_tasks,
        public_phase_id="S7_S1_INHERITED_PUBLIC_PARTITION", base_task_prefix="S1D1",
        basis_refs=basis_refs, operators=operators,
        member_budget=int(budgets["s1_depth2_collision_member_budget"]), max_tasks=4520,
    )
    s1_d2 = None
    if s1_collision["multi_class_count"] and not s1_collision["budget_exceeded"]:
        tasks = _refinement_tasks(s1_d1_tasks, s1_collision["member_task_ids"], prefix="S1D2")
        s1_d2 = runtime.run_structural_partition(
            phase_id="S7_S1_BRANCH_MULTISET_DEPTH2", tasks=tasks, evaluator_ref=BRANCH_EVALUATOR,
            requested_workers=4, max_tasks=int(budgets["s1_depth2_collision_member_budget"]),
            stream_shard_task_limit=1,
        )

    # Independent higher holdout: all 64 ordered triples under the existing C00 Axis-A generator;
    # first 256 distinct exact states in canonical-byte order, fixed before any S1 quotient result.
    higher_tasks = []
    for triple in itertools.product(basis_refs, repeat=3):
        payload = {"triple": list(triple)}
        higher_tasks.append(TaskSpec(
            task_id="H-" + "-".join(triple), task_kind="G6_S7_HIGHER_C00_HOLDOUT_REGEN",
            binding_sha256=canonical_sha256(payload), payload=payload, cost_weight=3.0,
        ))
    higher_gen = runtime.run_content_indexed_generation(
        phase_id="S7_HIGHER_HOLDOUT_CENSUS", tasks=higher_tasks, evaluator_ref=GEN_HIGHER_EVALUATOR,
        requested_workers=4, max_tasks=64, max_generated_occurrences=100000,
    )
    higher_rows = list(itertools.islice(runtime.iter_generated_states(phase_id="S7_HIGHER_HOLDOUT_CENSUS"), 256))
    if len(higher_rows) != 256:
        raise G6S7Error(f"S7_HIGHER_HOLDOUT_COUNT:{len(higher_rows)}")
    higher_public_tasks = [_tree_task("HOLDPUB", r, depth=1, basis_refs=basis_refs, operators=operators) for r in higher_rows]
    higher_public = runtime.run_structural_partition(
        phase_id="S7_HIGHER_HOLDOUT_INHERITED_PUBLIC_PARTITION", tasks=higher_public_tasks,
        evaluator_ref=PUBLIC_EVALUATOR, requested_workers=4, max_tasks=256,
    )
    higher_d1_tasks = [_tree_task("HOLD1", r, depth=1, basis_refs=basis_refs, operators=operators) for r in higher_rows]
    higher_d1_science, higher_d1_execution, higher_collision = _lazy_depth1_partition(
        runtime, panel_tag="S7_HIGHER_HOLDOUT_BRANCH_MULTISET", rows=higher_rows, public_tasks=higher_public_tasks,
        public_phase_id="S7_HIGHER_HOLDOUT_INHERITED_PUBLIC_PARTITION", base_task_prefix="HOLD1",
        basis_refs=basis_refs, operators=operators,
        member_budget=int(budgets["higher_depth2_collision_member_budget"]), max_tasks=256,
    )
    higher_d2 = None
    if higher_collision["multi_class_count"] and not higher_collision["budget_exceeded"]:
        tasks = _refinement_tasks(higher_d1_tasks, higher_collision["member_task_ids"], prefix="HOLD2")
        higher_d2 = runtime.run_structural_partition(
            phase_id="S7_HIGHER_HOLDOUT_BRANCH_MULTISET_DEPTH2", tasks=tasks, evaluator_ref=BRANCH_EVALUATOR,
            requested_workers=4, max_tasks=int(budgets["higher_depth2_collision_member_budget"]),
            stream_shard_task_limit=1,
        )

    s1_disp = _panel_disposition(s1_d1_science, None if s1_d2 is None else s1_d2.summary, s1_collision)
    higher_disp = _panel_disposition(higher_d1_science, None if higher_d2 is None else higher_d2.summary, higher_collision)
    budget_hit = "BUDGET_EXCEEDED" in s1_disp or "BUDGET_EXCEEDED" in higher_disp
    candidate = s1_disp == "NONTRIVIAL_SURVIVES_DEPTH2" or higher_disp == "NONTRIVIAL_SURVIVES_DEPTH2"
    if budget_hit:
        classification = "BUDGET_INSUFFICIENT_FOR_REGISTERED_DEPTH2_REFINEMENT"
        status = "REVIEW_REQUIRED"
        next_authorized = None
    elif candidate:
        classification = "NONTRIVIAL_ORDINARY_BRANCH_MULTISET_QUOTIENT_CANDIDATE_SURVIVES_DEPTH2"
        status = "PASS"
        next_authorized = plan["next_on_candidate"]
    else:
        classification = "ORDINARY_BRANCH_MULTISET_DISCRETE_BY_DEPTH2_ON_REGISTERED_PANELS"
        status = "PASS"
        next_authorized = plan["next_on_discrete"]

    result = {
        "schema_id": RESULT_SCHEMA,
        "status": status,
        "stage_id": STAGE_ID,
        "classification": classification,
        "accepted_decoder_source_sha256": accepted_source_sha256,
        "internal_execution_id": internal_execution_id,
        "question_sha256": plan["question_sha256"],
        "observer_id": plan["observer"]["id"],
        "branch_semantics": plan["observer"]["branch_semantics"],
        "ordinary_context_count": 248,
        "ordinary_operator_count": 31,
        "marker_used": False,
        "q_D_used": False,
        "observer_decode_used": False,
        "exact_parent_reconstruction_used": False,
        "exact_child_identity_in_signature": False,
        "scalar_branch_counts_are_sole_future_signature": False,
        "s1_generation": {"science": s1_gen.summary, "execution": s1_gen.execution_metadata},
        "s1_inherited_public_partition": {"science": s1_public.summary, "execution": s1_public.execution_metadata},
        "s1_depth1_partition": {"science": s1_d1_science, "execution": s1_d1_execution},
        "s1_depth1_collision_info": s1_collision,
        "s1_depth2_partition": None if s1_d2 is None else {"science": s1_d2.summary, "execution": s1_d2.execution_metadata},
        "s1_disposition": s1_disp,
        "higher_holdout_generation": {"science": higher_gen.summary, "execution": higher_gen.execution_metadata},
        "higher_holdout_selected_state_count": 256,
        "higher_inherited_public_partition": {"science": higher_public.summary, "execution": higher_public.execution_metadata},
        "higher_depth1_partition": {"science": higher_d1_science, "execution": higher_d1_execution},
        "higher_depth1_collision_info": higher_collision,
        "higher_depth2_partition": None if higher_d2 is None else {"science": higher_d2.summary, "execution": higher_d2.execution_metadata},
        "higher_disposition": higher_disp,
        "nontrivial_quotient_candidate_present": bool(candidate),
        "recursive_congruence_all_finite_terms_earned": False,
        "global_minimality_earned": False,
        "arbitrary_g5_leaf_scope_earned": False,
        "g6_graduated": False,
        "promotion": False,
        "r_series_started": False,
        "next_authorized": next_authorized,
        "nonclaims": plan["nonclaims"],
        "scientific_interpretation": (
            "A non-singleton class survives the complete ordinary action-indexed successor-block multiset observer through registered depth 2 on at least one preregistered panel. This is a genuine compressed-public-state candidate for S8, but not an all-finite congruence or minimality theorem."
            if candidate and not budget_hit else
            "The registered depth-2 refinement could not be completed for every non-singleton depth-1 class within the preregistered collision-member budget; no S7 quotient conclusion is promoted."
            if budget_hit else
            "The complete ordinary action-indexed successor-block multiset observer separates every tested exact state by depth 1 or depth 2 on both registered panels. This is bounded evidence for ordinary-future discreteness, not a global exact-state minimality theorem."
        ),
    }
    result["result_sha256"] = canonical_sha256(result)
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out / "G6_S7_RESULT.json", result)
    (out / "READ_FIRST.txt").write_text(
        "INFINITY GRID — G6:S7 ORDINARY BRANCH-SENSITIVE QUOTIENT\n\n" + classification +
        "\n\nMarker/reconstruction operations: NOT USED\nExact child identity in scientific signature: NO\n"
        "Branch semantics: action-indexed structural multisets of successor blocks\nResult SHA-256: " + result["result_sha256"] + "\n",
        encoding="utf-8",
    )
    return result
