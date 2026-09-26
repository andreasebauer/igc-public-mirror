from __future__ import annotations

"""Cut-down contract tests. All execution happens in the step-2 registered job."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.v05_execution_authority import ExecutionAuthorityError
from infinity_grid.v05_origin_guard import registered_workspace_scope, require_controller_execution_origin, current_execution_context_snapshot
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.execution import ExecutionPolicy, _choose_start_method


def _save_job(root, job):
    body = {k:v for k,v in job.items() if k != 'registration_sha256'}
    job = dict(body,registration_sha256=canonical_sha256(body))
    write_json_atomic(root/'registry'/(job['job_id']+'.json'),job)
    return job


def _workspace(tmp_path, *, workers=1, name='work'):
    from infinity_grid import submission as sub
    source = Path(loop.__file__).resolve().parents[1]
    record = sub.capture_record(source.parent)
    logical = ('saved_stage_one_prerun' if name == 'fresh' else
               'saved_stage_outcome' if name == 'outcome' else
               'saved_stage_four' if workers == 4 else 'saved_stage_one')
    row = next(x for x in record['job']['input_artifacts'] if x['logical_name'] == logical)
    archive = source.parent/'runtime/intake/artifacts'/(row['sha256']+'.bin')
    root = tmp_path/name
    loop.restore_workspace(archive, root, row['sha256'])
    return root, sub.capture_record(root)['job']


def _exact_relation(root):
    db = next((root/'runtime/runs').rglob('partition.sqlite3'))
    with sqlite3.connect(db) as conn:
        rows = list(conn.execute('SELECT t.task_id,c.representative_signature_bytes FROM task_results t JOIN classes c ON t.class_token=c.class_token ORDER BY t.task_id'))
    return rows


def _scientific_files(root):
    """Digest existing scientific state; retained refusal records are administrative."""
    paths = list((root/'runtime/runs').rglob('partition.sqlite3'))
    paths += list((root/'runtime/intake/completed').glob('*.json'))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def test_unregistered_job_is_rejected(tmp_path):
    root,job = _workspace(tmp_path)
    before = _scientific_files(root)
    with pytest.raises(loop.ControllerLoopError,match='JOB_NOT_REGISTERED'):
        loop.run_workspace_job(root,'UNREGISTERED')
    assert _scientific_files(root) == before


def test_changed_source_is_rejected_before_execution(tmp_path):
    root,job = _workspace(tmp_path)
    before = _scientific_files(root)
    with (root/'source/infinity_grid/controller_only_fixture.py').open('a') as f: f.write('\n# changed\n')
    with pytest.raises(loop.ControllerLoopError,match='WORKSPACE_SOURCE_MISMATCH'):
        loop.run_workspace_job(root,job['job_id'])
    assert _scientific_files(root) == before


def test_changed_registration_is_rejected(tmp_path):
    root,job = _workspace(tmp_path)
    job['execution']['parameters']['modulus'] = 8
    write_json_atomic(root/'registry'/(job['job_id']+'.json'),job)
    with pytest.raises(loop.ControllerLoopError,match='JOB_REGISTRATION_MISMATCH'):
        loop.run_workspace_job(root,job['job_id'])


def test_changed_input_bytes_are_rejected(tmp_path):
    root,job = _workspace(tmp_path)
    raw = b'{"certified":true}\n'; sha = hashlib.sha256(raw).hexdigest()
    artifact = root/'runtime/intake/artifacts'/(sha+'.json')
    artifact.parent.mkdir(parents=True, exist_ok=True); artifact.write_bytes(raw)
    job['input_artifacts'] = [{'logical_name':'fixture','sha256':sha}]
    job = _save_job(root,job)
    assert loop.validate_workspace_job(root,job['job_id'])['artifacts']['fixture'] == artifact
    artifact.write_bytes(b'{"certified":false}\n')
    with pytest.raises(loop.ControllerLoopError,match='PASSIVE_ARTIFACT_RESOLUTION'):
        loop.run_workspace_job(root,job['job_id'])


def test_registration_not_supervisor_credentials_opens_scope(tmp_path):
    from infinity_grid.invocation import InvocationRefused
    root,job = _workspace(tmp_path)
    with pytest.raises(ExecutionAuthorityError): require_controller_execution_origin('direct')
    with pytest.raises(InvocationRefused, match='RECORDED_ATTEMPT_REQUIRED'):
        with registered_workspace_scope(root,job['job_id']):
            raise AssertionError('registration must not grant runtime authority')
    with pytest.raises(ExecutionAuthorityError): require_controller_execution_origin('after')
    assert not (root/'RECOVERY_CAPSULE.json').exists()


def test_one_and_four_workers_give_identical_exact_relation(tmp_path):
    one,j1 = _workspace(tmp_path,workers=1,name='one')
    four,j4 = _workspace(tmp_path,workers=4,name='four')
    a = loop.run_workspace_job(one,j1['job_id']); b = loop.run_workspace_job(four,j4['job_id'])
    assert a['status'] == b['status'] == 'COMPLETED'
    assert _exact_relation(one) == _exact_relation(four)
    assert a['result']['partition']['class_count'] == b['result']['partition']['class_count'] == 7
    # Inspect the actual shared-runtime execution record, not a requested-worker claim.
    metas = [json.loads(p.read_text())['execution'] for p in (four/'runtime/runs').rglob('SUMMARY.json')]
    assert metas and any(m.get('workers') == 4 for m in metas)


def test_completed_job_is_reused_and_record_is_byte_identical(tmp_path):
    root,job = _workspace(tmp_path)
    first = loop.run_workspace_job(root,job['job_id'])
    file = next((root/'runtime/intake/completed').glob('*.json')); before = file.read_bytes()
    again = loop.run_workspace_job(root,job['job_id'])
    assert again['reused'] is True and first['result_sha256'] == again['result_sha256']
    assert file.read_bytes() == before
    assert len(list((root/'runtime/attempts').rglob('*.json'))) == 1


def test_interrupted_checkpoint_restores_elsewhere_and_resumes(tmp_path,monkeypatch):
    # Use a genuinely saved pre-execution export. Never rewind a completion,
    # its prepared record, terminal proof, or immutable project event history.
    root,job = _workspace(tmp_path, name='fresh')
    assert not list((root/'runtime/intake/completed').glob('*.json'))
    assert not list((root/'runtime/intake/prepared_completions').glob('*.json'))
    assert not list((root/'runtime/attempts').rglob('*.json'))
    monkeypatch.setenv('IG_V05_ENGINEERING_FAULT_INJECTION_ACK','STAGE_RUNTIME_ABORT_TEST')
    monkeypatch.setenv('IG_V05_ENGINEERING_ABORT_AFTER_COMMITS','5')
    with pytest.raises(Exception,match='ABORT_AFTER_5_COMMITS'):
        loop.run_workspace_job(root,job['job_id'])
    db = next((root/'runtime/runs').rglob('partition.sqlite3'))
    with sqlite3.connect(db) as conn:
        before = dict(conn.execute('SELECT task_id,committed_utc FROM task_results'))
    assert len(before) == 5
    monkeypatch.delenv('IG_V05_ENGINEERING_FAULT_INJECTION_ACK')
    monkeypatch.delenv('IG_V05_ENGINEERING_ABORT_AFTER_COMMITS')
    snap = loop.export_workspace(root,tmp_path/'paused.zip')
    restored = tmp_path/'different-root'
    loop.restore_workspace(snap['path'],restored,snap['sha256'])
    done = loop.run_workspace_job(restored,job['job_id'])
    assert done['status'] == 'COMPLETED'
    after_db = next((restored/'runtime/runs').rglob('partition.sqlite3'))
    with sqlite3.connect(after_db) as conn:
        after = dict(conn.execute('SELECT task_id,committed_utc FROM task_results'))
    assert len(after) == 48 and all(after[k] == v for k,v in before.items())


def test_workspace_lock_refuses_conflicting_runner(tmp_path):
    root,job = _workspace(tmp_path)
    with loop._workspace_lock(root):
        with pytest.raises(loop.ControllerLoopError,match='WORKSPACE_BUSY'):
            loop.run_workspace_job(root,job['job_id'])
        with pytest.raises(loop.ControllerLoopError,match='WORKSPACE_BUSY'):
            loop.export_workspace(root,tmp_path/'busy.zip')


def test_wrong_archive_checksum_is_rejected(tmp_path):
    root,job = _workspace(tmp_path)
    snap = loop.export_workspace(root,tmp_path/'snapshot.zip')
    with pytest.raises(loop.ControllerLoopError,match='RESTORE_ARCHIVE_HASH'):
        loop.restore_workspace(snap['path'],tmp_path/'new-root','0'*64)
    assert not (tmp_path/'new-root').exists()


def test_completed_output_mutation_is_not_reused(tmp_path):
    root,job = _workspace(tmp_path)
    loop.run_workspace_job(root,job['job_id'])
    output = next((root/'runtime/runs').rglob('*.json'))
    output.write_text('{}\n')
    with pytest.raises(loop.ControllerLoopError,match='COMPLETION_EVIDENCE_MISMATCH'):
        loop.run_workspace_job(root,job['job_id'])


def test_unregistered_scientific_outcome_is_not_committed(tmp_path):
    root,job = _workspace(tmp_path, name='outcome')
    with pytest.raises(loop.ControllerLoopError,match='UNREGISTERED_SCIENTIFIC_OUTCOME'):
        loop.run_workspace_job(root,job['job_id'])
    assert not list((root/'runtime/intake/completed').glob('*.json'))


def test_explicit_runner_start_method_overrides_auto(monkeypatch):
    monkeypatch.setenv('IG_DECODER_START_METHOD','spawn')
    assert _choose_start_method(ExecutionPolicy(start_method='AUTO')) == 'spawn'
    assert _choose_start_method(ExecutionPolicy(start_method='fork')) == 'fork'
