import hashlib
import json
from pathlib import Path

import pytest

from infinity_grid.replay_node_in_executor import NODE_EXECUTOR_BINDINGS, NODE_IDS
from infinity_grid.replay_p6_frontier_import import P6_STATE_OBJECT_SHA256
from infinity_grid.replay_reference_data import empty_manifest
from infinity_grid.v05_stage_runtime import ReplayRootPaused, StageRuntimeError, StageScienceRuntime

RES = Path(__file__).resolve().parents[1] / 'infinity_grid/resources/replay'
MANIFEST = RES / 'L0_UPWARD_SRCF_ASSERTION_DAG_V9.json'
CATALOGUE = RES / 'L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json'
FRONTIER = RES / 'P6_CAPTURED_STATE_OBJECT_V1.zip'


def parameters(injection='NONE'):
    return {'operation': 'EXECUTE_NODE_IN_P7',
            'manifest_input': 'compiled_replay_manifest',
            'manifest_file_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            'manifest_dag_sha256': json.loads(MANIFEST.read_text())['dag_sha256'],
            'reference_catalogue_input': 'replay_reference_catalogue',
            'reference_catalogue_file_sha256': hashlib.sha256(CATALOGUE.read_bytes()).hexdigest(),
            'p6_frontier_input': 'p6_captured_state_object',
            'p6_frontier_file_sha256': P6_STATE_OBJECT_SHA256,
            'dataset_root_sha256': empty_manifest()['manifest_sha256'],
            'root_run_id': 'L0-G8-REPLAY-ROOT-V1', 'authorized_through_layer': 'G8',
            'next_wait_layer': 'GLOBAL', 'node_executor_bindings': NODE_EXECUTOR_BINDINGS,
            'pilot_fault_injection': injection}


def artifacts(frontier=FRONTIER):
    return {'compiled_replay_manifest': str(MANIFEST),
            'replay_reference_catalogue': str(CATALOGUE),
            'p6_captured_state_object': str(frontier)}


def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr('infinity_grid.v05_origin_guard.require_registered_output', lambda *a, **k: None)
    candidate = object.__new__(StageScienceRuntime)
    candidate._root = tmp_path / 'registered-output'
    candidate._root.mkdir()
    candidate._require_execution = lambda: None
    candidate.publish_json = lambda name, obj: {'logical_name': name, 'sha256': '0' * 64}
    return candidate


def test_fresh_node_in_and_cold_resume(tmp_path, monkeypatch):
    candidate = runtime(tmp_path, monkeypatch)
    result = candidate.run_registered_replay_root(parameters(), artifacts())
    assert result['outcome'] == 'NODE_IN_PILOT_COMPLETE'
    assert result['science_executed'] is True
    assert result['fresh_empty_root_science_node_count'] == 1
    assert result['runner_action']['node_id'] == 'IG/SCOUT_HISTORICAL/S/ASSERTION_MAPPED_STRUCTURE'
    assert len(result['completed_node_ids']) == 20
    state = candidate._root / 'replay_runner/runner_state.json'
    original = state.read_bytes()
    resumed = candidate.run_registered_replay_root(parameters(), artifacts())
    assert resumed['runner_state_sha256'] == result['runner_state_sha256']
    assert state.read_bytes() == original
    checkpoint_hash = json.loads(original)['accepted_checkpoint_sha256_by_node'][NODE_IDS[1]]
    checkpoint = json.loads((candidate._root / 'replay_runner/checkpoints' / (checkpoint_hash + '.json')).read_text())
    assert checkpoint['acceptance']['mode'] == 'AUTOMATIC_FRESH_RECOMPUTATION'
    assert checkpoint['acceptance']['node_result']['fresh_metrics']['regression_exact_cases'] == 43008


def test_injected_fresh_mismatch_stops_at_r(tmp_path, monkeypatch):
    candidate = runtime(tmp_path, monkeypatch)
    with pytest.raises(ReplayRootPaused):
        candidate.run_registered_replay_root(parameters(NODE_IDS[1]), artifacts())
    state = json.loads((candidate._root / 'replay_runner/runner_state.json').read_text())
    assert len(state['completed_node_ids']) == 16
    assert state['active_audit_capsule_sha256'] is not None


def test_frontier_tamper_fails_closed(tmp_path, monkeypatch):
    candidate = runtime(tmp_path, monkeypatch)
    bad = tmp_path / 'changed.zip'
    bad.write_bytes(FRONTIER.read_bytes() + b'changed')
    with pytest.raises(StageRuntimeError, match='P7_FRONTIER_FILE_HASH'):
        candidate.run_registered_replay_root(parameters(), artifacts(bad))
