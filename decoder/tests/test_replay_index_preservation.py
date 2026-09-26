"""Registered engineering tests; no scientific result or authority is created."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from infinity_grid.canon import write_json_atomic
from infinity_grid.replay_dag_runner import ReplayDagRunner, ReplayDagRunnerError
from infinity_grid.replay_reference_data import ReplayReferenceDataStore, ReferenceDataError
from infinity_grid.replay_index_lock import replay_index_lock
from test_p2a_replay_dag_runner import manifest, result
from test_p3_replay_reference_data import fixtures


ROOT = Path(__file__).parents[1]
OLD_HASHES = {
    "replay_dag_runner": "a76cec5465495b569a0757957534dc107c3955515cb617e1014d55aaca9f00dd",
    "replay_reference_data": "e150ab83215197be848edd520d6597377dc386748eaddeef0ed53baf2653d100",
}


def old_module(name):
    path = ROOT / "tests/fixtures/dev79_index_reproduction" / f"{name}.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == OLD_HASHES[name]
    spec = importlib.util.spec_from_file_location(f"infinity_grid._dev79_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner_at(root, cls=ReplayDagRunner):
    return cls.create(manifest(), root, root_run_id="INDEX-REGRESSION",
                      dataset_root={"state": "EMPTY", "sha256": "0" * 64})


def test_exact_dev79_reference_source_reproduces_stale_writer_loss(tmp_path):
    cls = old_module("replay_reference_data").ReplayReferenceDataStore
    first = cls.initialize(tmp_path)
    stale = cls(tmp_path)
    a, b = fixtures()[:2]
    first.put(a)
    stale.put(b)
    reopened = cls(tmp_path)
    assert reopened.manifest["record_ids"] == [b["record_id"]]
    assert (tmp_path / "records" / f"{a['record_sha256']}.json").exists()


def test_exact_dev79_runner_source_reproduces_stale_writer_loss(tmp_path):
    cls = old_module("replay_dag_runner").ReplayDagRunner
    first = runner_at(tmp_path, cls)
    stale = cls.resume(manifest(), tmp_path)
    action = first.next_action()
    first.record_node_result(result(action))
    stale.next_action()
    assert cls.resume(manifest(), tmp_path).state["completed_node_ids"] == []
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 1


def test_reference_stale_instances_refresh_without_losing_committed_records(tmp_path):
    first = ReplayReferenceDataStore.initialize(tmp_path)
    stale = ReplayReferenceDataStore(tmp_path)
    a, b = fixtures()[:2]
    first.put(a)
    stale.put(b)
    assert ReplayReferenceDataStore(tmp_path).manifest["record_ids"] == [a["record_id"], b["record_id"]]
    # A stale reader also verifies the actual persisted index.
    first.verify()
    assert first.manifest == stale.manifest


@pytest.mark.parametrize("operation", ["next", "result", "decision"])
def test_runner_stale_instances_refuse_before_writing(tmp_path, operation):
    first = runner_at(tmp_path)
    stale = ReplayDagRunner.resume(manifest(), tmp_path)
    action = first.next_action()
    first.record_node_result(result(action))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ReplayDagRunnerError, match="STALE_RUNNER_STATE"):
        if operation == "next":
            stale.next_action()
        elif operation == "result":
            stale.record_node_result(result(action))
        else:
            stale.apply_external_decision({})
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert ReplayDagRunner.resume(manifest(), tmp_path).state == first.state


def test_reference_valid_old_index_is_refused_without_rewriting_evidence(tmp_path):
    store = ReplayReferenceDataStore.initialize(tmp_path)
    old = deepcopy(store.manifest)
    store.put(fixtures()[0])
    write_json_atomic(tmp_path / "MANIFEST.json", old)
    before = {p.name: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ReferenceDataError, match="REFERENCE_INDEX_ROLLBACK"):
        ReplayReferenceDataStore(tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_imported_reference_records_also_detect_index_rollback(tmp_path):
    store = ReplayReferenceDataStore.initialize(tmp_path)
    old = deepcopy(store.manifest)
    row = fixtures()[0]
    # Existing frontier imports copy records and their manifest, without commits.
    write_json_atomic(tmp_path / "records" / f"{row['record_sha256']}.json", row)
    write_json_atomic(tmp_path / "MANIFEST.json", old)
    with pytest.raises(ReferenceDataError, match="REFERENCE_INDEX_ROLLBACK_OR_INTERRUPTED_PUT"):
        ReplayReferenceDataStore(tmp_path)


def test_runner_valid_old_index_is_refused_without_rewriting_evidence(tmp_path):
    runner = runner_at(tmp_path)
    old = deepcopy(runner.state)
    runner.record_node_result(result(runner.next_action()))
    write_json_atomic(tmp_path / "runner_state.json", old)
    before = {p.name: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ReplayDagRunnerError, match="RUNNER_INDEX_ROLLBACK_OR_INTERRUPTED_ACCEPT"):
        ReplayDagRunner.resume(manifest(), tmp_path)
    assert before == {p.name: p.read_bytes() for p in tmp_path.rglob("*.json")}


def test_reference_recovers_crash_after_manifest_before_commit_exactly_once(tmp_path, monkeypatch):
    store = ReplayReferenceDataStore.initialize(tmp_path)
    row = fixtures()[0]
    def crash(_transaction):
        raise RuntimeError("CRASH_BEFORE_COMMIT")
    monkeypatch.setattr(store, "_commit", crash)
    with pytest.raises(RuntimeError, match="CRASH_BEFORE_COMMIT"):
        store.put(row)
    proposed = (tmp_path / "MANIFEST.json").read_bytes()
    resumed = ReplayReferenceDataStore(tmp_path)
    assert (tmp_path / "MANIFEST.json").read_bytes() == proposed
    assert resumed.manifest["record_ids"] == [row["record_id"]]
    assert resumed.put(row) == row
    assert len(list((tmp_path / "commits").glob("*.json"))) == 1


def test_competing_operations_fail_busy_before_mutating_indexes(tmp_path):
    refs = ReplayReferenceDataStore.initialize(tmp_path / "refs")
    runner = runner_at(tmp_path / "runner")
    for root, error, operation in [
        (refs.root, ReferenceDataError, lambda: refs.put(fixtures()[0])),
        (runner.state_dir, ReplayDagRunnerError, runner.next_action),
    ]:
        before = {p.name: p.read_bytes() for p in root.rglob("*.json")}
        with replay_index_lock(root, error):
            with pytest.raises(error, match="REPLAY_INDEX_BUSY"):
                operation()
        assert before == {p.name: p.read_bytes() for p in root.rglob("*.json")}


def test_preserved_p9_frontier_is_compatible_and_unchanged(tmp_path):
    archive = ROOT / "tests/fixtures/dev79_index_reproduction/P9_CAPTURED_STATE_OBJECT_V1.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == "b360397d2593cfee3086ac4b23560c6ab8de2418d6d607d89844463a00766cfc"
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            relative = Path(name)
            assert not relative.is_absolute() and ".." not in relative.parts
        z.extractall(tmp_path)
    state_path, = tmp_path.rglob("runner_state.json")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
    runner = ReplayDagRunner.resume(manifest(), state_path.parent)
    store = ReplayReferenceDataStore(state_path.parent.parent / "replay_reference_data")
    assert len(runner.state["completed_node_ids"]) == 30
    assert len(store.manifest["record_ids"]) == 133
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*.json")}
