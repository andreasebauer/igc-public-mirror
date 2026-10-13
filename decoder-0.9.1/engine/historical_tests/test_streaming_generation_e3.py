from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.structural_encoding import structural_canonical_sha256
from infinity_grid.v05_stage_runtime import StageScienceRuntime, StageRuntimeError
from infinity_grid.v05_execution_authority import EngineeringAuthority, digest
from infinity_grid.g6_s3_repaired import _axis_a_task
from infinity_grid.g6_controller_evaluators import g6_s3_axis_a_generation_evaluator


def _runtime(root: Path) -> StageScienceRuntime:
    root.mkdir(parents=True, exist_ok=True)
    bindings={
        "chain_id":"E3TEST", "registration_sha256":"1"*64, "stage_id":"G6:S3R",
        "handler_key":"fixture", "handler_ref":"infinity_grid.controller_only_fixture:stage_handler",
        "question_sha256":"8"*64, "source_sha256":"3"*64, "handler_source_sha256":"4"*64,
        "parameters_sha256":"5"*64, "authority_sha256":"6"*64, "dependencies_sha256":"7"*64,
        "evidence_store_id":digest({"chain_dir":str(root.resolve(strict=True))}), "run_id":"b"*32,
    }
    permit=EngineeringAuthority(True).begin(bindings)
    return StageScienceRuntime(
        chain_dir=root, chain_id="E3TEST", stage_id="G6:S3R", question_sha256="8" * 64,
        default_workers=4, memory_budget_bytes=1024**3, workspace_budget_bytes=100 * 1024**2,
        _execution_permit=permit,
    )


def _tasks(n: int) -> list[TaskSpec]:
    out=[]
    for i in range(n):
        payload={"value":i}
        out.append(TaskSpec(task_id=f"t{i:03d}", task_kind="GEN_FIXTURE",
                            binding_sha256=canonical_sha256(payload), payload=payload, cost_weight=1))
    return out


def test_streaming_generation_dedup_and_resume(tmp_path: Path):
    rt=_runtime(tmp_path/'one')
    r=rt.run_content_indexed_generation(
        phase_id='GEN', tasks=_tasks(4), evaluator_ref='infinity_grid.controller_only_fixture:generation_evaluator',
        requested_workers=1, max_tasks=4, max_generated_occurrences=8)
    assert r.summary['raw_generated_occurrence_count']==8
    assert r.summary['distinct_exact_state_count']==6
    rows=list(rt.iter_generated_states(phase_id='GEN'))
    assert len(rows)==6
    r2=rt.run_content_indexed_generation(
        phase_id='GEN', tasks=_tasks(4), evaluator_ref='infinity_grid.controller_only_fixture:generation_evaluator',
        requested_workers=1, max_tasks=4, max_generated_occurrences=8)
    assert r2.summary==r.summary
    assert r2.execution_metadata['committed_generation_tasks_this_invocation']==0


def test_streaming_generation_one_vs_four_workers(tmp_path: Path):
    r1=_runtime(tmp_path/'w1').run_content_indexed_generation(
        phase_id='GEN', tasks=_tasks(12), evaluator_ref='infinity_grid.controller_only_fixture:generation_evaluator', requested_workers=1)
    r4=_runtime(tmp_path/'w4').run_content_indexed_generation(
        phase_id='GEN', tasks=_tasks(12), evaluator_ref='infinity_grid.controller_only_fixture:generation_evaluator', requested_workers=4)
    for key in ('raw_generated_occurrence_count','distinct_exact_state_count','generated_state_set_sha256','generation_occurrence_bindings_sha256'):
        assert r1.summary[key]==r4.summary[key]


def test_streaming_generation_abort_resume(tmp_path: Path, monkeypatch):
    rt=_runtime(tmp_path/'abort')
    monkeypatch.setenv('IG_V05_ENGINEERING_FAULT_INJECTION_ACK','STAGE_RUNTIME_GENERATION_ABORT_TEST')
    monkeypatch.setenv('IG_V05_ENGINEERING_ABORT_AFTER_GENERATION_TASKS','2')
    with pytest.raises(StageRuntimeError, match='ENGINEERING_GENERATION_ABORT'):
        rt.run_content_indexed_generation(
            phase_id='GEN', tasks=_tasks(5), evaluator_ref='infinity_grid.controller_only_fixture:generation_evaluator', requested_workers=1)
    monkeypatch.delenv('IG_V05_ENGINEERING_ABORT_AFTER_GENERATION_TASKS')
    monkeypatch.delenv('IG_V05_ENGINEERING_FAULT_INJECTION_ACK')
    r=rt.run_content_indexed_generation(
        phase_id='GEN', tasks=_tasks(5), evaluator_ref='infinity_grid.controller_only_fixture:generation_evaluator', requested_workers=1)
    assert r.summary['task_count']==5
    assert r.summary['raw_generated_occurrence_count']==10
    assert r.execution_metadata['committed_generation_tasks_this_invocation']==3


def test_streaming_generation_digest_collision_uses_exact_bytes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv('IG_V05_ENGINEERING_FAULT_INJECTION_ACK','GENERATION_DIGEST_COLLISION_TEST')
    rt=_runtime(tmp_path/'collision')
    r=rt.run_content_indexed_generation(
        phase_id='GEN', tasks=_tasks(2), evaluator_ref='infinity_grid.controller_only_fixture:collision_generation_evaluator', requested_workers=1)
    assert r.summary['distinct_exact_state_count']==2
    db=rt.runtime_root/'phases'/'GEN'/'state_store.sqlite3'
    conn=sqlite3.connect(db)
    try:
        rows=list(conn.execute('SELECT state_token,canonical_bytes FROM states ORDER BY class_index'))
    finally:
        conn.close()
    assert [x[0] for x in rows]==['e'*64+':0','e'*64+':1']
    assert rows[0][1] != rows[1][1]


def test_g6_axis_a_generation_matches_historical_reference():
    triple=('D2_PATH','D2_PATH','D2_PATH')
    old=_axis_a_task(triple)
    new=g6_s3_axis_a_generation_evaluator({'triple':list(triple)})
    old_ids={r['state_id'] for r in old['records']}
    new_ids={structural_canonical_sha256(r['identity']) for r in new['states']}
    assert old_ids==new_ids
