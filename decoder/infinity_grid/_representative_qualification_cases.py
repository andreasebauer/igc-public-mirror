"""Engineering qualification assertions; called only with a native stage runtime."""

import sqlite3

import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
import infinity_grid.v05_stage_runtime as stage_runtime

EVAL = "infinity_grid.controller_only_fixture:partition_evaluator"




def _tasks():
    out = []
    for value in range(257):
        payload = {"value": value, "modulus": 1000}
        out.append(TaskSpec(
            task_id=f"u{value:04d}", task_kind="REP_BYTES_FIXTURE",
            binding_sha256=canonical_sha256(payload), payload=payload, cost_weight=1.0,
        ))
    payload = {"value": 256, "modulus": 1000}
    out.append(TaskSpec(
        task_id="z-duplicate-0256", task_kind="REP_BYTES_FIXTURE",
        binding_sha256=canonical_sha256({"duplicate": payload}), payload=payload, cost_weight=1.0,
    ))
    return out


def _db(runtime, phase):
    return runtime.runtime_root / "phases" / phase / "partition.sqlite3"


def test_old_partition_schema_migrates_additively(tmp_path):
    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
    conn.execute("CREATE TABLE task_results (task_id TEXT PRIMARY KEY, payload_sha256 TEXT NOT NULL, signature_sha256 TEXT NOT NULL, class_token TEXT NOT NULL, outcome_count INTEGER NOT NULL, metrics_json TEXT NOT NULL, locator_json TEXT, committed_utc TEXT NOT NULL)")
    conn.execute("CREATE TABLE classes (class_token TEXT PRIMARY KEY, signature_sha256 TEXT NOT NULL, class_index INTEGER NOT NULL, representative_task_id TEXT NOT NULL, size INTEGER NOT NULL)")
    conn.commit(); conn.close()
    migrated = stage_runtime.StageScienceRuntime._open_db(db)
    try:
        columns = {str(r[1]) for r in migrated.execute("PRAGMA table_info(classes)")}
        assert "representative_signature_bytes" in columns
    finally:
        migrated.close()


def test_257_distinct_then_duplicate_needs_no_controller_reopen(monkeypatch, tmp_path, runtime):
    out = runtime.run_structural_partition(
        phase_id="SERIAL257", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
    )
    assert out.summary["task_count"] == 258
    assert out.summary["class_count"] == 257
    assert out.summary["multi_class_count"] == 1
    assert out.summary["max_class_size"] == 2
    assert out.execution_metadata["controller_representative_fallback_evaluations"] == 0
    assert out.execution_metadata["representative_signature_storage"] == "SQLITE_CLASSES_BLOB_V1"
    conn = sqlite3.connect(_db(runtime, "SERIAL257"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM classes WHERE representative_signature_bytes IS NULL").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM classes WHERE size=2").fetchone()[0] == 1
    finally:
        conn.close()


def test_cold_multicore_over_256_classes_keeps_exact_grouping_and_durable_bytes(monkeypatch, tmp_path, runtime):
    out = runtime.run_structural_partition(
        phase_id="MULTI257", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=4, max_tasks=258, stream_shard_task_limit=1,
    )
    assert out.summary["class_count"] == 257
    assert out.summary["max_class_size"] == 2
    assert out.execution_metadata["workers"] == 4
    assert out.execution_metadata["controller_representative_fallback_evaluations"] == 0
    conn = sqlite3.connect(_db(runtime, "MULTI257"))
    try:
        assert conn.execute("SELECT COUNT(*) FROM classes WHERE representative_signature_bytes IS NULL").fetchone()[0] == 0
    finally:
        conn.close()


def test_partial_resume_reuses_durable_rep_bytes_without_reopen(monkeypatch, tmp_path, runtime):
    monkeypatch.setenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK", "STAGE_RUNTIME_ABORT_TEST")
    monkeypatch.setenv("IG_V05_ENGINEERING_ABORT_AFTER_COMMITS", "257")
    with pytest.raises(stage_runtime.StageRuntimeError, match="ABORT_AFTER_257_COMMITS"):
        runtime.run_structural_partition(
            phase_id="PARTIAL", tasks=_tasks(), evaluator_ref=EVAL,
            requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
        )
    monkeypatch.delenv("IG_V05_ENGINEERING_ABORT_AFTER_COMMITS")
    monkeypatch.delenv("IG_V05_ENGINEERING_FAULT_INJECTION_ACK")
    out = runtime.run_structural_partition(
        phase_id="PARTIAL", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
    )
    assert out.summary["class_count"] == 257
    assert out.summary["max_class_size"] == 2
    assert out.execution_metadata["controller_representative_fallback_evaluations"] == 0
    first_hash = out.summary["partition_bindings_sha256"]
    again = runtime.run_structural_partition(
        phase_id="PARTIAL", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
    )
    assert again.summary["partition_bindings_sha256"] == first_hash
    assert again.execution_metadata["controller_representative_fallback_evaluations"] == 0


def test_legacy_null_rep_binds_view_once_backfills_then_reuses(monkeypatch, tmp_path, runtime):
    baseline = runtime.run_structural_partition(
        phase_id="LEGACY", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
    )
    db = _db(runtime, "LEGACY")
    conn = sqlite3.connect(db)
    try:
        token, rep_tid = conn.execute("SELECT class_token,representative_task_id FROM classes WHERE size=2").fetchone()
        conn.execute("DELETE FROM task_results WHERE task_id='z-duplicate-0256'")
        conn.execute("UPDATE classes SET size=1, representative_signature_bytes=NULL WHERE class_token=?", (token,))
        conn.commit()
    finally:
        conn.close()

    calls = {"bind": 0}
    original_bind = stage_runtime._bind_controller_fallback_kernel_view
    def checked_bind(ref, scope):
        calls["bind"] += 1
        return original_bind(ref, scope)
    monkeypatch.setattr(stage_runtime, "_bind_controller_fallback_kernel_view", checked_bind)

    resumed = runtime.run_structural_partition(
        phase_id="LEGACY", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
    )
    assert resumed.summary["partition_bindings_sha256"] == baseline.summary["partition_bindings_sha256"]
    assert resumed.execution_metadata["controller_representative_fallback_evaluations"] == 1
    assert calls["bind"] == 1
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT representative_signature_bytes IS NOT NULL FROM classes WHERE class_token=?", (token,)).fetchone()[0] == 1
    finally:
        conn.close()
    completed = runtime.run_structural_partition(
        phase_id="LEGACY", tasks=_tasks(), evaluator_ref=EVAL,
        requested_workers=1, max_tasks=258, stream_shard_task_limit=1,
    )
    assert completed.summary["partition_bindings_sha256"] == baseline.summary["partition_bindings_sha256"]
    assert completed.execution_metadata["controller_representative_fallback_evaluations"] == 0
