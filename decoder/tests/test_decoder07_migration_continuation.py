from __future__ import annotations

import json
import lzma
from pathlib import Path

import pytest

from infinity_grid import __version__, candidate_workflow, chunked_save, continuation, portable_registry as project
from infinity_grid import project_migration, submission as sub
from infinity_grid.canon import canonical_sha256
from infinity_grid.v05_controller_event_loop import _source_ids


SOURCE = Path(__file__).resolve().parents[1]


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _qualified_target(tmp_path):
    raw = sub._archive(sub._tree(SOURCE))
    archive = tmp_path / "target.zip"; archive.write_bytes(raw)
    source_sha, package_sha = _source_ids(SOURCE)
    record = candidate_workflow.make_candidate(
        version=__version__, parent_source_sha256="0" * 64,
        candidate_source_sha256=source_sha,
        changed_files=["infinity_grid/project_migration.py"],
        contracts=["project.local.migration"],
        test_results=[{"name": "fixture", "outcome": "PASS"}], reason="migration unit fixture")
    saved = candidate_workflow.save_candidate(tmp_path / "candidates", record)
    qualification = Path(saved["path"])
    return archive, source_sha, package_sha, qualification


def _old_source(tmp_path):
    old = tmp_path / "old" / "infinity_grid"; old.mkdir(parents=True)
    (old / "_version.py").write_text("__version__ = '0.6.0'\n")
    (old / "producer.py").write_text("VALUE = 'historical'\n")
    return old.parent


def _migration_request(root, tmp_path):
    head, release = project.current(root, "release")
    archive, source_sha, package_sha, qualification = _qualified_target(tmp_path)
    return {"schema_id": project_migration.REQUEST_SCHEMA,
        "project_id": project.read(root / "PROJECT.json")["project_id"],
        "expected_release_head": head, "expected_old_engine": release["engine"],
        "target_archive": str(archive), "target_archive_sha256": sub._sha(archive.read_bytes()),
        "target_source_sha256": source_sha, "target_package_sha256": package_sha,
        "qualification": str(qualification), "qualification_sha256": sub._sha(qualification.read_bytes()),
        "authorization": {"operator": "fixture", "decision": "AUTHORIZE_PROJECT_LOCAL_MIGRATION",
                          "scope": "ONE_PROJECT_NO_SHARED_POINTER"},
        "reason": "adopt qualified local engine"}


def test_project_local_migration_is_append_only_and_idempotent(tmp_path):
    store = tmp_path / "project"
    project.initialize(store, "migration fixture", _old_source(tmp_path))
    root = store / "coordination"; before = project.snapshot(root)
    request = _migration_request(root, tmp_path)
    result = project_migration.migrate(root, request)
    assert result["status"] == "PROJECT_MIGRATED"
    head, release = project.current(root, "release")
    assert head == result["release_head"] and release["version"] == __version__
    assert release["previous"] == request["expected_release_head"]
    assert project.blob(root, request["expected_old_engine"])
    retry = project_migration.migrate(root, request)
    assert retry["status"] == "PROJECT_ALREADY_MIGRATED" and retry["release_head"] == head
    assert before != project.snapshot(root)


def test_project_local_migration_rejects_bad_old_target_and_qualification(tmp_path):
    store = tmp_path / "project"; project.initialize(store, "negative fixture", _old_source(tmp_path))
    root = store / "coordination"; request = _migration_request(root, tmp_path)
    bad = dict(request, expected_old_engine="f" * 64)
    with pytest.raises(sub.SubmissionError, match="MIGRATION_OLD_ENGINE_MISMATCH"):
        project_migration.migrate(root, bad)
    bad = dict(request, target_source_sha256="e" * 64)
    with pytest.raises(sub.SubmissionError, match="MIGRATION_TARGET_SOURCE_HASH"):
        project_migration.migrate(root, bad)
    q = Path(request["qualification"]); q.write_bytes(q.read_bytes() + b" ")
    with pytest.raises(sub.SubmissionError, match="MIGRATION_QUALIFICATION_FILE_HASH"):
        project_migration.migrate(root, request)


def test_migration_refuses_branch_label_qualification_without_mutation(tmp_path):
    store = tmp_path / "project"
    project.initialize(store, "version mismatch fixture", _old_source(tmp_path))
    root = store / "coordination"
    request = _migration_request(root, tmp_path)
    q = Path(request["qualification"])
    record = json.loads(q.read_text())
    record["version"] = "0.8lib"
    record["record_sha256"] = canonical_sha256({k: v for k, v in record.items() if k != "record_sha256"})
    _json(q, record)
    request["qualification_sha256"] = sub._sha(q.read_bytes())
    before = project.snapshot(root)
    with pytest.raises(sub.SubmissionError, match="MIGRATION_TARGET_NOT_QUALIFIED"):
        project_migration.migrate(root, request)
    assert project.snapshot(root) == before


def _plan(count=124):
    body = {"schema_id": continuation.PLAN_SCHEMA, "campaign_id": "WORKER02",
            "actions": [{"action_id": f"A{i:03d}", "source_sha256": f"{i:064x}",
                         "inputs_sha256": f"{i + 1000:064x}", "budget_bytes": 10}
                        for i in range(count)]}
    return continuation.seal_plan(body)


def _completed(plan, count):
    return [{"action_id": action["action_id"], "source_sha256": action["source_sha256"],
             "inputs_sha256": action["inputs_sha256"], "result_sha256": f"{i + 2000:064x}",
             "evidence_role": "worker_result", "evidence_sha256": f"{i + 3000:064x}",
             "evidence_size_bytes": 5, "status": "COMPLETED"}
            for i, action in enumerate(plan["actions"][:count])]


def test_worker02_continues_only_112_to_124_suffix(tmp_path):
    plan = _plan(); prefix = continuation.create_prefix(plan, _completed(plan, 112))
    request = {"previous_capture_id": "capture-old", "previous_attempt_id": "attempt-1",
               "previous_status": "INTERRUPTED", "new_capture_id": "capture-new",
               "new_attempt_id": "attempt-2", "max_cumulative_bytes": 1000,
               "additional_role_usage": {"second_step": 12},
               "max_cumulative_roles": {"second_step": 12},
               "reason": "resume verified missing suffix"}
    result = continuation.continue_plan(tmp_path, plan, prefix, request)
    assert result["record"]["completed_action_ids"] == [f"A{i:03d}" for i in range(112)]
    assert result["record"]["selected_action_ids"] == [f"A{i:03d}" for i in range(112, 124)]
    assert result["record"]["continuation"]["output_namespace"] == "captures/capture-new/attempts/attempt-2/segments"
    retry = continuation.continue_plan(tmp_path, plan, prefix, request)
    assert retry["record"] == result["record"]


def test_continuation_refuses_gap_corruption_failed_and_budget(tmp_path):
    plan = _plan(4); rows = _completed(plan, 2)
    rows[1]["action_id"] = "A003"
    with pytest.raises(sub.SubmissionError, match="CONTINUATION_PREFIX_NOT_ORDERED"):
        continuation.create_prefix(plan, rows)
    prefix = continuation.create_prefix(plan, _completed(plan, 2))
    prefix["rows"][0]["result_sha256"] = "f" * 64
    request = {"previous_capture_id": "old", "previous_attempt_id": "one", "previous_status": "PAUSED",
               "new_capture_id": "new", "new_attempt_id": "two", "max_cumulative_bytes": 100,
               "additional_role_usage": {"second_step": 2},
               "max_cumulative_roles": {"second_step": 2},
               "reason": "resume"}
    with pytest.raises(sub.SubmissionError, match="CONTINUATION_PREFIX_SEAL"):
        continuation.continue_plan(tmp_path, plan, prefix, request)
    prefix = continuation.create_prefix(plan, _completed(plan, 2))
    with pytest.raises(sub.SubmissionError, match="CONTINUATION_TERMINAL_STATUS"):
        continuation.continue_plan(tmp_path, plan, prefix, dict(request, previous_status="FAILED"))
    with pytest.raises(sub.SubmissionError, match="CONTINUATION_CUMULATIVE_BUDGET"):
        continuation.continue_plan(tmp_path, plan, prefix, dict(request, max_cumulative_bytes=1))
    with pytest.raises(sub.SubmissionError, match="CONTINUATION_CUMULATIVE_ROLE_BUDGET"):
        continuation.continue_plan(tmp_path, plan, prefix, dict(request, max_cumulative_roles={"second_step": 1}))


def test_chunked_save_resume_readback_and_reconstruction(tmp_path):
    source = tmp_path / "large.bin"; source.write_bytes(bytes(range(251)) * 43)
    parts = tmp_path / "parts"; manifest = chunked_save.pack(source, parts, 1024)
    assert len(manifest["parts"]) > 4
    readback = tmp_path / "readback"; readback.mkdir()
    for row in manifest["parts"][:-1]:
        (readback / row["name"]).write_bytes((parts / row["name"]).read_bytes())
    assert [r["name"] for r in chunked_save.pending(manifest, readback)] == [manifest["parts"][-1]["name"]]
    last = manifest["parts"][-1]; (readback / last["name"]).write_bytes((parts / last["name"]).read_bytes())
    output = tmp_path / "rebuilt.bin"
    receipt = chunked_save.verify_and_reconstruct(manifest, readback, output)
    assert output.read_bytes() == source.read_bytes()
    assert receipt["reconstructed_sha256"] == manifest["source_sha256"]
    (readback / manifest["parts"][0]["name"]).write_bytes(b"corrupt")
    with pytest.raises(sub.SubmissionError, match="CHUNK_SAVE_PARTS_PENDING"):
        chunked_save.verify_and_reconstruct(manifest, readback, tmp_path / "bad.bin")


def test_historical_xz_chunk_transport_reconstructs(tmp_path):
    data = bytes(range(199)) * 101
    compressed = lzma.compress(data); parts = tmp_path / "xz"; parts.mkdir()
    rows = []
    for index, start in enumerate(range(0, len(compressed), 197)):
        raw = compressed[start:start + 197]; path = parts / f"part-{index:05d}"; path.write_bytes(raw)
        rows.append({"drive_file_id": f"fixture-{index}", "sha256": sub._sha(raw), "size_bytes": len(raw)})
    transport = {"schema_id": "IG_DECODER_TRANSPORT_MANIFEST_V1", "encoding": "XZ_CHUNKS",
                 "object": {"sha256": sub._sha(data), "size_bytes": len(data)}, "parts": rows}
    target = tmp_path / "restored.bin"
    receipt = chunked_save.verify_xz_transport(transport, parts, target)
    assert target.read_bytes() == data and receipt["reconstructed_sha256"] == sub._sha(data)
