from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.controller_only_fixture import stage_handler
from infinity_grid.execution import TaskSpec, WorkerTaskError
from infinity_grid.v05_chain import (
    CHAIN_SCHEMA_V1,
    CHAIN_SCHEMA_V2,
    ChainExecutionResult,
    ScientificChainController,
    ScientificChainValidationError,
    seal_chain_registration,
)
from infinity_grid.v05_stage_architecture import (
    StageArchitectureViolation,
    audit_module_source,
)
from infinity_grid.v05_stage_runtime import StageRuntimeError, StageScienceRuntime
from infinity_grid.v05_execution_authority import EngineeringAuthority, ExecutionAuthorityError, digest

SHA0 = "0" * 64
SHA1 = "1" * 64
QUESTION = "2" * 64


def _stage(*, kind="DECODER_STAGE", workers=4, collision=False, stage_id="G6:SX"):
    execution = (
        {"handler_key": "fixture", "parameters": {
            "values": list(range(60)), "modulus": 5, "workers": workers,
            "collision_mode": collision,
            "reducer_memory_budget_bytes": 512 * 1024 * 1024,
            "workspace_budget_bytes": 16 * 1024 * 1024,
        }}
        if kind == "DECODER_STAGE"
        else {"executor_key": "legacy", "parameters": {}}
    )
    return {
        "stage_id": stage_id, "series": "S", "stage_kind": kind,
        "question_ref": "controller-only-fixture", "question_sha256": QUESTION,
        "depends_on": [], "execution": execution,
        "result_contract": {
            "artifact_logical_name": "result.json", "outcome_pointer": "/outcome",
            "allowed_outcomes": ["PASS"],
        },
        "transitions": {"PASS": {"action": "END", "next_stage": None, "reason": "FIXTURE_DONE"}},
        "auto_run": True, "promotion_effect": "NONE",
    }


def _reg(*, schema=CHAIN_SCHEMA_V2, kind="DECODER_STAGE", chain_id="V2-CONTROLLER-ONLY-TEST", workers=4, collision=False, hierarchy="G", level=6, stage_id="G6:SX"):
    return seal_chain_registration({
        "schema_id": schema,
        "chain_id": chain_id,
        "subject": {"hierarchy": hierarchy, "level": level, "parent_ref": "ENGINEERING_FIXTURE" if hierarchy == "ENGINEERING" else "G5_GRADUATED"},
        "release_line": "v0.5 / 0.50",
        "mode": "CONTROLLED_S_THEN_ADAPTIVE_R",
        "parent_authority": {
            "authority_ref": "fixture", "science_sha256": SHA0,
            "verification_sha256": SHA1, "status": "GRADUATED",
        },
        "stages": [_stage(kind=kind, workers=workers, collision=collision, stage_id=stage_id)],
        "budgets": {"max_stage_executions": 4, "max_chain_wall_seconds": 120, "default_workers": workers},
        "durability": {"fsync_each_transition": True, "stage_commits": "APPEND_ONLY", "external_mirror_required": False},
        "authority_policy": {
            "automatic_promotion": False, "require_verified_parent": True,
            "novelty_policy": "REVIEW_REQUIRED", "changed_assumptions_policy": "REVIEW_REQUIRED",
            "unregistered_outcome_policy": "REVIEW_REQUIRED",
            "r_science_policy": "ADAPTIVE_ONLY_WITHIN_PREREGISTERED_BRANCHES",
        },
    })


def _runtime(*, chain_dir: Path, chain_id: str, stage_id: str, question_sha256: str, **kwargs):
    chain_dir=Path(chain_dir); chain_dir.mkdir(parents=True, exist_ok=True)
    bindings={
        "chain_id":str(chain_id), "registration_sha256":"1"*64, "stage_id":str(stage_id),
        "handler_key":"fixture", "handler_ref":"infinity_grid.controller_only_fixture:stage_handler",
        "question_sha256":str(question_sha256), "source_sha256":"3"*64, "handler_source_sha256":"4"*64,
        "parameters_sha256":"5"*64, "authority_sha256":"6"*64, "dependencies_sha256":"7"*64,
        "evidence_store_id":digest({"chain_dir":str(chain_dir.resolve(strict=True))}), "run_id":"a"*32,
    }
    permit=EngineeringAuthority(True).begin(bindings)
    return StageScienceRuntime(chain_dir=chain_dir, chain_id=chain_id, stage_id=stage_id,
        question_sha256=question_sha256, _execution_permit=permit, **kwargs)


def _tasks(n=60, modulus=5):
    out=[]
    for value in range(n):
        payload={"value": value, "modulus": modulus}
        out.append(TaskSpec(
            task_id=f"v{value:06d}", task_kind="FIXTURE",
            binding_sha256=canonical_sha256({"q": QUESTION, "payload": payload}),
            payload=payload,
        ))
    return out


def test_architecture_gate_rejects_historical_private_g6_execution_modules():
    for rel in (
        "infinity_grid/g6_s1_repair.py",
        "infinity_grid/g6_s2_repaired.py",
        "infinity_grid/g6_s3_repaired.py",
    ):
        r=audit_module_source(Path(__file__).parents[1] / rel)
        assert r["status"] == "FAIL"
        assert any(x["kind"] in {"FORBIDDEN_IMPORT", "FORBIDDEN_EXECUTION_CALL", "FORBIDDEN_DURABLE_WRITE"} for x in r["violations"])
    assert audit_module_source(Path(__file__).parents[1] / "infinity_grid/controller_only_fixture.py")["status"] == "PASS"
    assert audit_module_source(Path(__file__).parents[1] / "infinity_grid/g6_s3_controller_stage.py")["status"] == "PASS"


def test_v2_registration_forbids_prepared_executor():
    with pytest.raises(ScientificChainValidationError, match="forbid PREPARED_EXECUTOR"):
        _reg(schema=CHAIN_SCHEMA_V2, kind="PREPARED_EXECUTOR")


def test_new_v1_registration_disabled_and_historical_executor_is_inspect_only(tmp_path):
    legacy=_reg(schema=CHAIN_SCHEMA_V1, kind="PREPARED_EXECUTOR", chain_id="LEGACY-V1-FIXTURE")
    assert legacy["schema_id"] == CHAIN_SCHEMA_V1
    c=ScientificChainController(tmp_path, allow_legacy_new_registrations=True, engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match="LEGACY_EXECUTOR_DISABLED"):
        c.register_executor("legacy", lambda st,root: ChainExecutionResult({"outcome":"PASS"}))


def test_v2_controller_only_fixture_runs_through_stage_runtime(tmp_path):
    reg=_reg(workers=2, hierarchy="ENGINEERING", level=0, stage_id="ENG:SX", chain_id="ENG-CONTROLLER-ONLY-TEST")
    c=ScientificChainController(tmp_path, engineering_only=True)
    c.register_stage_handler("fixture", stage_handler)
    out=c.run(reg)
    assert out["status"] == "COMPLETE"
    commit=json.loads((tmp_path/reg["chain_id"] / "stage_commits" / "ENG__SX.json").read_text())
    part=commit["result"]["partition"]
    assert part["task_count"] == 60
    assert part["class_count"] == 5
    assert part["multi_class_count"] == 5
    assert part["max_class_size"] == 12
    db=tmp_path/reg["chain_id"] / "decoder_stage_runtime" / "ENG__SX" / "phases" / "PARTITION" / "partition.sqlite3"
    assert db.is_file()


def test_one_worker_equals_four_worker_science(tmp_path):
    def run(root: Path, workers: int):
        rt=_runtime(
            chain_dir=root, chain_id="SAME", stage_id="G6:SX", question_sha256=QUESTION,
            default_workers=workers, memory_budget_bytes=512*1024*1024,
            workspace_budget_bytes=16*1024*1024,
        )
        return rt.run_structural_partition(
            phase_id="P", tasks=_tasks(),
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
            requested_workers=workers,
        ).summary
    aroot=tmp_path/"a"; broot=tmp_path/"b"; aroot.mkdir(); broot.mkdir()
    assert run(aroot,1) == run(broot,4)


def test_engine_owned_resume_reuses_committed_tasks_exactly(tmp_path, monkeypatch):
    root=tmp_path/"resume"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="RESUME", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=4, memory_budget_bytes=512*1024*1024,
        workspace_budget_bytes=16*1024*1024,
    )
    monkeypatch.setenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK", "STAGE_RUNTIME_ABORT_TEST")
    monkeypatch.setenv("IG_V05_ENGINEERING_ABORT_AFTER_COMMITS", "40")
    with pytest.raises((StageRuntimeError, WorkerTaskError)):
        rt.run_structural_partition(
            phase_id="P", tasks=_tasks(),
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
            requested_workers=4,
        )
    db=root/"decoder_stage_runtime"/"G6__SX"/"phases"/"P"/"partition.sqlite3"
    with sqlite3.connect(db) as conn:
        committed=int(conn.execute("select count(*) from task_results").fetchone()[0])
    assert committed == 40
    monkeypatch.delenv("IG_V05_ENGINEERING_ABORT_AFTER_COMMITS")
    monkeypatch.delenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK")
    resumed=rt.run_structural_partition(
        phase_id="P", tasks=_tasks(),
        evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
        requested_workers=4,
    )
    clean_root=tmp_path/"clean"; clean_root.mkdir()
    clean=_runtime(
        chain_dir=clean_root, chain_id="RESUME", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=512*1024*1024,
        workspace_budget_bytes=16*1024*1024,
    ).run_structural_partition(
        phase_id="P", tasks=_tasks(), evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
        requested_workers=1,
    )
    assert resumed.summary == clean.summary
    assert resumed.execution_metadata["task_count"] == 20


def test_deliberate_digest_collision_uses_structural_fallback(tmp_path, monkeypatch):
    root=tmp_path/"collision"; root.mkdir()
    monkeypatch.setenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK", "STAGE_RUNTIME_DIGEST_COLLISION_TEST")
    rt=_runtime(
        chain_dir=root, chain_id="COLLISION", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=2, memory_budget_bytes=512*1024*1024,
        workspace_budget_bytes=16*1024*1024,
    )
    tasks=_tasks(8, modulus=99)
    out=rt.run_structural_partition(
        phase_id="P", tasks=tasks,
        evaluator_ref="infinity_grid.controller_only_fixture:collision_partition_evaluator",
        requested_workers=2,
    )
    assert out.summary["task_count"] == 8
    assert out.summary["class_count"] == 8
    assert out.summary["multi_class_count"] == 0
    witnesses=list((root/"decoder_stage_runtime"/"G6__SX"/"phases"/"P"/"digest_collision_witnesses").glob("*.json"))
    assert len(witnesses) == 7


def test_runtime_workspace_budget_is_engine_owned_and_fail_closed(tmp_path):
    root=tmp_path/"budget"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="BUDGET", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=512*1024*1024, workspace_budget_bytes=1,
    )
    with pytest.raises((StageRuntimeError, WorkerTaskError)):
        rt.run_structural_partition(
            phase_id="P", tasks=_tasks(40), evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
            requested_workers=1,
        )

def test_legacy_full_signature_shards_stream_into_compact_runtime_store(tmp_path):
    legacy=tmp_path/"legacy"; legacy.mkdir()
    for shard_i in range(3):
        rows=[]
        for j in range(4):
            value=shard_i*4+j
            rows.append({"state_id":f"s{value:03d}", "signature":{"bucket":value%3,"payload":[value%3]*40}})
        base={
            "schema_id":"LEGACY_FIXTURE_V1", "level":"L1", "chunk_index":shard_i,
            "state_ids":[r["state_id"] for r in rows], "rows":rows,
        }
        base["science_sha256"]=canonical_sha256(base)
        (legacy/f"{shard_i:05d}.json").write_text(json.dumps(base,sort_keys=True,separators=(",",":")))
    root=tmp_path/"import"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="IMPORT", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=512*1024*1024, workspace_budget_bytes=16*1024*1024,
    )
    out=rt.import_legacy_partition_shards(
        phase_id="LEGACY", shard_root=legacy, expected_schema_id="LEGACY_FIXTURE_V1", expected_level="L1",
    )
    assert out.summary["task_count"]==12
    assert out.summary["class_count"]==3
    assert out.summary["multi_class_count"]==3
    assert out.summary["max_class_size"]==4
    assert out.execution_metadata["legacy_shards_verified"]==3
    assert out.execution_metadata["evidence_bytes"] < 1024*1024

def test_register_stage_handler_rejects_historical_private_executor_module(tmp_path):
    from infinity_grid.g6_s3_repaired import g6_s3_repaired_executor
    c=ScientificChainController(tmp_path)
    with pytest.raises(StageArchitectureViolation):
        c.register_stage_handler("forbidden.old.s3", g6_s3_repaired_executor)

def test_e2_digest_collision_partition_binding_is_worker_order_independent(tmp_path, monkeypatch):
    monkeypatch.setenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK", "STAGE_RUNTIME_DIGEST_COLLISION_TEST")
    def run(root, workers):
        root.mkdir()
        tasks=[]
        for value in range(32):
            payload={"value":value,"modulus":99}
            tasks.append(TaskSpec(
                task_id=f"v{value:06d}", task_kind="FIXTURE",
                binding_sha256=canonical_sha256({"q":QUESTION,"payload":payload}),
                payload=payload, cost_weight=float((value % 7) + 1),
            ))
        rt=_runtime(
            chain_dir=root, chain_id="COLLISION-ORDER", stage_id="G6:SX", question_sha256=QUESTION,
            default_workers=workers, memory_budget_bytes=512*1024*1024,
            workspace_budget_bytes=32*1024*1024,
        )
        return rt.run_structural_partition(
            phase_id="P", tasks=tasks,
            evaluator_ref="infinity_grid.controller_only_fixture:collision_partition_evaluator",
            requested_workers=workers,
        ).summary
    one=run(tmp_path/"one",1)
    four=run(tmp_path/"four",4)
    assert one == four
    assert one["class_count"] == 32
    assert one["partition_bindings_sha256"] == four["partition_bindings_sha256"]
    assert one["deterministic_collision_class_ordering"] == "EXACT_CANONICAL_BYTES_ASCENDING_V1"

def test_e2_reduce_finalize_interruption_reuses_all_task_commits(tmp_path, monkeypatch):
    root=tmp_path/"reduce-resume"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="REDUCE-RESUME", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=4, memory_budget_bytes=512*1024*1024,
        workspace_budget_bytes=32*1024*1024,
    )
    monkeypatch.setenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK", "STAGE_RUNTIME_REDUCE_ABORT_TEST")
    monkeypatch.setenv("IG_V05_ENGINEERING_ABORT_AFTER_REDUCE_FINALIZE", "1")
    with pytest.raises(StageRuntimeError, match="ABORT_AFTER_REDUCE_FINALIZE"):
        rt.run_structural_partition(
            phase_id="P", tasks=_tasks(80, modulus=7),
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
            requested_workers=4,
        )
    db=root/"decoder_stage_runtime"/"G6__SX"/"phases"/"P"/"partition.sqlite3"
    with sqlite3.connect(db) as conn:
        assert int(conn.execute("select count(*) from task_results").fetchone()[0]) == 80
    monkeypatch.delenv("IG_V05_ENGINEERING_ABORT_AFTER_REDUCE_FINALIZE")
    monkeypatch.delenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK")
    resumed=rt.run_structural_partition(
        phase_id="P", tasks=_tasks(80, modulus=7),
        evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
        requested_workers=4,
    )
    clean_root=tmp_path/"reduce-clean"; clean_root.mkdir()
    clean=_runtime(
        chain_dir=clean_root, chain_id="REDUCE-RESUME", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=512*1024*1024,
        workspace_budget_bytes=32*1024*1024,
    ).run_structural_partition(
        phase_id="P", tasks=_tasks(80, modulus=7),
        evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
        requested_workers=1,
    )
    assert resumed.summary == clean.summary
    assert resumed.execution_metadata["backend"] == "REUSE"
    assert resumed.execution_metadata["task_count"] == 0

def test_e2_stream_execution_uses_bounded_microshards(tmp_path):
    root=tmp_path/"microshards"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="MICROSHARDS", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=4, memory_budget_bytes=512*1024*1024,
        workspace_budget_bytes=64*1024*1024,
    )
    out=rt.run_structural_partition(
        phase_id="P", tasks=_tasks(600, modulus=17),
        evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",
        requested_workers=4,
    )
    assert out.summary["task_count"] == 600
    assert out.execution_metadata["stream_shard_task_limit"] == 24
    assert out.execution_metadata["max_stream_shard_tasks"] <= 24
    assert out.execution_metadata["stream_shard_count"] >= 25
    assert out.execution_metadata["max_inflight_shards"] == 8
    assert out.execution_metadata["max_observed_inflight_shards"] <= 8
    assert out.execution_metadata["completed_stream_shards"] == out.execution_metadata["stream_shard_count"]

def test_e2_admission_rejects_stale_question_and_task_scope(tmp_path):
    root=tmp_path/"scope"; root.mkdir()
    tasks=_tasks(12, modulus=3)
    rt=_runtime(
        chain_dir=root, chain_id="SCOPE", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=512*1024*1024, workspace_budget_bytes=32*1024*1024,
    )
    rt.run_structural_partition(
        phase_id="P", tasks=tasks,
        evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator", requested_workers=1,
    )
    changed_question=_runtime(
        chain_dir=root, chain_id="SCOPE", stage_id="G6:SX", question_sha256="a"*64,
        default_workers=1, memory_budget_bytes=512*1024*1024, workspace_budget_bytes=32*1024*1024,
    )
    with pytest.raises(StageRuntimeError, match="durable scope mismatch"):
        changed_question.run_structural_partition(
            phase_id="P", tasks=tasks,
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator", requested_workers=1,
        )
    altered=list(tasks)
    payload={"value":999,"modulus":3}
    altered[0]=TaskSpec(
        task_id=altered[0].task_id, task_kind="FIXTURE",
        binding_sha256=canonical_sha256({"q":QUESTION,"payload":payload}), payload=payload,
    )
    with pytest.raises(StageRuntimeError, match="durable scope mismatch"):
        rt.run_structural_partition(
            phase_id="P", tasks=altered,
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator", requested_workers=1,
        )


def test_e2_memory_budget_fails_at_admission_before_task_commit(tmp_path):
    root=tmp_path/"membudget"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="MEM", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=1, workspace_budget_bytes=32*1024*1024,
    )
    with pytest.raises(StageRuntimeError, match="memory budget exceeded"):
        rt.run_structural_partition(
            phase_id="P", tasks=_tasks(20),
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator", requested_workers=1,
        )
    db=root/"decoder_stage_runtime"/"G6__SX"/"phases"/"P"/"partition.sqlite3"
    with sqlite3.connect(db) as conn:
        assert int(conn.execute("select count(*) from task_results").fetchone()[0]) == 0


def test_e2_workspace_accounting_fails_closed_on_unknown_phase_entry(tmp_path):
    root=tmp_path/"unknown"; root.mkdir()
    rt=_runtime(
        chain_dir=root, chain_id="UNKNOWN", stage_id="G6:SX", question_sha256=QUESTION,
        default_workers=1, memory_budget_bytes=512*1024*1024, workspace_budget_bytes=32*1024*1024,
    )
    phase=rt.runtime_root/"phases"/"P"; phase.mkdir(parents=True)
    (phase/"science-stage-private-shard.bin").write_bytes(b"forbidden")
    with pytest.raises(StageRuntimeError, match="unexpected runtime workspace file"):
        rt.run_structural_partition(
            phase_id="P", tasks=_tasks(4),
            evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator", requested_workers=1,
        )

def test_e2_content_indexed_state_store_deduplicates_exact_payloads_and_resumes(tmp_path):
    root=tmp_path/"state-store"; root.mkdir()
    occurrences=[]
    for i in range(500):
        k=i%25
        occurrences.append({
            "occurrence_id":f"o{i:06d}",
            "state":{"kind":"fixture-tree","vertices":k+3,"labels":[k%7,(k*3)%11],"edges":[[j,j+1] for j in range(k+2)]},
            "metadata":{"source_row":i},
        })
    rt=_runtime(
        chain_dir=root,chain_id="STATE-STORE",stage_id="G6:SX",question_sha256=QUESTION,
        default_workers=1,memory_budget_bytes=512*1024*1024,workspace_budget_bytes=64*1024*1024,
    )
    first=rt.build_content_indexed_state_store(phase_id="STATE",occurrences=occurrences,max_occurrences=500)
    assert first.summary["raw_occurrence_count"]==500
    assert first.summary["distinct_exact_state_count"]==25
    assert first.summary["duplicate_occurrence_count"]==475
    assert first.summary["state_payload_compression_ratio"] > 19.0
    assert first.summary["digest_equality_never_decides_state_equality"] is True
    resumed=rt.build_content_indexed_state_store(phase_id="STATE",occurrences=list(reversed(occurrences)),max_occurrences=500)
    assert resumed.summary==first.summary
    assert resumed.execution_metadata["committed_occurrences_this_invocation"]==0


def test_e2_content_indexed_state_store_digest_collision_uses_exact_bytes(tmp_path,monkeypatch):
    monkeypatch.setenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK","STATE_STORE_DIGEST_COLLISION_TEST")
    occurrences=[]
    for i in range(12):
        occurrences.append({
            "occurrence_id":f"c{i:03d}","state":{"exact":i,"payload":[i]*5},
            "_test_state_digest_override":"e"*64,
        })
    def run(root,items):
        root.mkdir()
        rt=_runtime(
            chain_dir=root,chain_id="STATE-COLLISION",stage_id="G6:SX",question_sha256=QUESTION,
            default_workers=1,memory_budget_bytes=512*1024*1024,workspace_budget_bytes=64*1024*1024,
        )
        return rt.build_content_indexed_state_store(phase_id="STATE",occurrences=items,max_occurrences=20).summary
    a=run(tmp_path/"a",occurrences)
    b=run(tmp_path/"b",list(reversed(occurrences)))
    assert a==b
    assert a["distinct_exact_state_count"]==12
    assert a["state_store_bindings_sha256"]==b["state_store_bindings_sha256"]
    assert a["exact_canonical_bytes_are_equality_authority"] is True

def test_e2_partition_execution_reports_phase_and_science_kernel_telemetry(tmp_path):
    root=tmp_path/"telemetry"; root.mkdir()
    rt=_runtime(
        chain_dir=root,chain_id="TELEMETRY",stage_id="G6:SX",question_sha256=QUESTION,
        default_workers=4,memory_budget_bytes=512*1024*1024,workspace_budget_bytes=64*1024*1024,
    )
    out=rt.run_structural_partition(
        phase_id="P",tasks=_tasks(40,modulus=5),
        evaluator_ref="infinity_grid.controller_only_fixture:partition_evaluator",requested_workers=4,
    )
    timings=out.execution_metadata["phase_timing_seconds"]
    for key in ("admission","engine_execution_including_stream_reduce","controller_classification",
                "controller_commit","finalize_verify","runtime_call_total_to_summary"):
        assert key in timings and timings[key] >= 0
    totals=out.execution_metadata["worker_metric_totals"]
    assert totals["engine_evaluator_wall_seconds"] >= 0
    assert totals["engine_signature_encode_wall_seconds"] >= 0
    assert totals["value"] == sum(range(40))
    status=json.loads((root/"decoder_stage_runtime"/"G6__SX"/"phases"/"P"/"telemetry"/"RUNTIME_STATUS.json").read_text())
    assert status["status"]=="COMPLETE"
    assert status["detail"]["actual_workers"]==4
    assert "phase_timing_seconds" in status["detail"]
