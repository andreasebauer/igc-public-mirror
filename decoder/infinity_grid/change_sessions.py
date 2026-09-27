"""Recorded development in one local store; shared ownership is a later layer.

Candidates are data here. Only the retained Decoder validation scheduler loads
their tests, in child processes. Hash checks detect changes to retained evidence;
they are not filesystem access controls against unrestricted same-user Python.
"""
from __future__ import annotations

import ast
import difflib
import json
from pathlib import Path
import re
import time
import tempfile

from . import submission as sub
from .canon import canonical_sha256, write_json_atomic
from . import v05_controller_event_loop as loop

Error = sub.SubmissionError
SCHEMA = 'IG_DECODER_CHANGE_SESSION_V1'


def _sealed(body, field='record_sha256'):
    return dict(body, **{field: canonical_sha256(body)})


def _verified(path, field='record_sha256'):
    record = sub._read(path)
    if canonical_sha256({k:v for k,v in record.items() if k != field}) != record.get(field):
        raise Error('CHANGE_RECORD_MISMATCH', str(path))
    return record


def _put_json(store, value, role):
    return sub._put_object(store, sub._json_bytes(value), '.json', role)


def _immutable(path, record):
    raw = sub._json_bytes(record)
    if path.exists():
        if path.read_bytes() != raw:
            raise Error('RETAINED_RECORD_CONFLICT', str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)


def _immutable_bytes(path, raw):
    """Create an immutable byte record, or verify an identical retry.

    Recovery tools use this for already-sealed historical bytes.  It never
    rewrites a retained record and therefore cannot manufacture missing
    history.
    """
    if path.exists():
        if path.read_bytes() != raw:
            raise Error('RETAINED_RECORD_CONFLICT', str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(raw)


def source_version(source):
    """Read the canonical literal without importing the candidate package."""
    tree = ast.parse((Path(source)/'infinity_grid/_version.py').read_text())
    values = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets)
              and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)]
    if len(values) != 1 or not re.fullmatch(r'(?:0\.[67]\.\d+(?:\.dev\d+)?|0\.8\.\d+(?:\.dev\d+)?\+lib)', values[0]):
        raise Error('CANONICAL_VERSION_REQUIRED')
    return values[0]


def _requirements(value):
    if (not isinstance(value, dict) or set(value) != {'statement', 'nodes'}
            or not isinstance(value['statement'], str) or not value['statement'].strip()
            or not isinstance(value['nodes'], list) or not value['nodes']
            or any(not isinstance(n, str) for n in value['nodes'])
            or len(set(value['nodes'])) != len(value['nodes'])):
        raise Error('CHANGE_REQUIREMENTS_REQUIRED')
    for node in value['nodes']:
        path = sub._relative(node.split('::', 1)[0])
        if not str(path).startswith('tests/') or path.suffix != '.py':
            raise Error('CHANGE_VALIDATION_NODES_ONLY')
    return value


def _completion(workspace, *, require_pass=True):
    """Verify the controller record AND its evidence without trusting a flag."""
    root = Path(workspace)
    rec = sub.capture_record(root)
    job = rec['job']
    admission = loop.validate_workspace_job(root, job['job_id'], check_loaded=False)
    # This reads sealed historical evidence; it does not execute the producer.
    # A new candidate run has its own captured environment admission.
    sub.require_saved(root, job['job_id'], check_environment=False)
    request = admission['request']; rid = request['request_id']
    done = _verified(root/'runtime/intake/completed'/f'{rid}.json', 'completion_sha256')
    if (done.get('schema_id') != 'IG_DECODER_WORKSPACE_COMPLETION_V1'
            or done.get('source_sha256') != job['source_sha256']
            or done.get('registration_sha256') != job['registration_sha256']
            or done.get('request_id') != rid
            or done.get('request_sha256') != canonical_sha256(request)
            or done.get('result_sha256') != canonical_sha256(done.get('result'))):
        raise Error('CHANGE_COMPLETION_BINDING')
    # There is one registered job per capture and exactly one matching evidence tree.
    base = root/('runtime/sealed' if done.get('evidence_protocol') else 'runtime/runs')
    roots = [p for p in base.glob('*') if p.is_dir()
             and loop._evidence_rows(p) == done.get('evidence')]
    if done.get('evidence_protocol') and loop.verified_completion(admission) != done:
        raise Error('CHANGE_COMPLETION_EVIDENCE')
    if len(roots) != 1 or not done.get('evidence'):
        raise Error('CHANGE_COMPLETION_EVIDENCE')
    from .preservation import terminal_completion_proof
    if not terminal_completion_proof(root, done):
        raise Error('CHANGE_COMPLETION_PENDING_TERMINAL_CHECKPOINT')
    result = done['result']
    passed = (done['status'] == 'COMPLETED' and
              (result.get('status') == 'PASS' if job['execution']['kind'] == 'VALIDATION'
               else result.get('outcome') == 'PASS' and result.get('validation', {}).get('status') == 'PASS'))
    if require_pass and not passed:
        raise Error('CHANGE_TESTS_NOT_PASSED', next_action='Keep this failure; capture and test a repaired revision.')
    return done


def _session(store, change_id):
    if not isinstance(change_id, str) or not loop._JOB_ID.fullmatch(change_id):
        raise Error('CHANGE_IDENTIFIER')
    folder = store/'changes'/change_id
    row = _verified(folder/'SESSION.json')
    if row.get('schema_id') != SCHEMA or row.get('change_id') != change_id:
        raise Error('CHANGE_SESSION_BINDING')
    return folder, row


def _object(store, obj):
    raw = (store/'objects'/sub._relative(obj['object_name'])).read_bytes()
    if sub._sha(raw) != obj['sha256'] or len(raw) != obj['size_bytes']:
        raise Error('CHANGE_OBJECT_MISMATCH', obj['object_name'])
    return raw


def _event(store, operation, payload):
    event = _sealed({'schema_id': 'IG_DECODER_CHANGE_EVENT_V1', 'operation': operation,
                     'recorded_ns': time.time_ns(), 'details': payload})
    _immutable(store/'change_events'/(event['record_sha256']+'.json'), event)
    return event


def open_change(store_root, request):
    store = Path(store_root).resolve(); store.mkdir(parents=True, exist_ok=True)
    request = sub._read(request) if isinstance(request, (str, Path)) else request
    fields = {'schema_id', 'change_id', 'parent_workspace', 'purpose', 'requirements', 'environment', 'resources'}
    if set(request) != fields or request['schema_id'] != 'IG_DECODER_CHANGE_OPEN_V1':
        raise Error('CHANGE_OPEN_FIELDS')
    cid = request['change_id']
    if not isinstance(cid, str) or not loop._JOB_ID.fullmatch(cid) or not str(request['purpose']).strip():
        raise Error('CHANGE_IDENTIFIER_OR_PURPOSE')
    _requirements(request['requirements']); sub._validate_environment(request['environment'])
    parent = Path(request['parent_workspace']).resolve(strict=True)
    done = _completion(parent)
    rec = sub.capture_record(parent)
    if any(o['role'] == 'project_source' for o in rec['objects']):
        raise Error('CHANGE_PARENT_ENGINE_ONLY')
    from . import portable_registry as project
    parent_project, parent_binding = project.locate(parent)
    if not (store/'coordination/PROJECT.json').exists():
        raw=project.snapshot(parent_project)
        archive=store/'PARENT_PROJECT.zip';archive.write_bytes(raw)
        project.import_history(store,archive,sub._sha(raw))
    project_head,project_release=project.current(store/'coordination','release')
    if project_release['engine']!=sub._sha(project.engine_object(parent/'source')):
        raise Error('PROJECT_CHANGE_PARENT_NOT_CURRENT')
    with loop._workspace_lock(store):
        folder = store/'changes'/cid
        if folder.exists():
            raise Error('CHANGE_ALREADY_EXISTS', cid, 'Use change-status, diff or capture-revision on the existing session.')
        files = sub._tree(parent/'source')
        obj = sub._put_object(store, sub._archive(files), '.zip', 'change_parent_source')
        baseline = {'version': source_version(parent/'source'), 'source_sha256': rec['workspace']['source_sha256'],
                    'package_sha256': rec['workspace']['package_sha256'], 'source_object': obj,
                    'completion_sha256': done['completion_sha256']}
        pointer_path = store/'CURRENT_LOCAL.json'
        if not pointer_path.exists():
            pointer = _sealed({'schema_id': 'IG_DECODER_LOCAL_DEVELOPMENT_POINTER_V1', 'epoch': 0,
                               'target': baseline, 'previous_pointer_sha256': None,
                               'scope': 'ISOLATED_LOCAL_DEVELOPMENT_NOT_SHARED_AUTHORITY'})
            _immutable(store/'pointers'/(pointer['record_sha256']+'.json'), pointer)
            write_json_atomic(pointer_path, pointer)
        pointer = _verified(pointer_path)
        if pointer['target']['source_sha256'] != baseline['source_sha256']:
            raise Error('CHANGE_PARENT_NOT_CURRENT', next_action='Reconcile with CURRENT_LOCAL before opening another change.')
        # Normalize environment paths into preserved content objects at open time.
        env = dict(request['environment']); artifacts = []
        for row in env['artifacts']:
            raw = Path(row['path']).read_bytes()
            if sub._sha(raw) != row['sha256']:
                raise Error('CHANGE_ENVIRONMENT_OBJECT_MISMATCH')
            saved = sub._put_object(store, raw, '.bin', 'environment:'+row['logical_name'])
            artifacts.append({'logical_name': row['logical_name'], 'sha256': saved['sha256'], 'object_name': saved['object_name']})
        env['artifacts'] = artifacts
        body = {'schema_id': SCHEMA, 'change_id': cid, 'purpose': request['purpose'],
                'parent': baseline, 'parent_pointer_sha256': pointer['record_sha256'],
                'project_parent_head': project_head,
                'requirements': request['requirements'], 'environment': env, 'resources': request['resources']}
        session = _sealed(body); _immutable(folder/'SESSION.json', session)
        _put_json(store, session, 'change_session')
        _event(store, 'OPEN', {'change_id': cid, 'session_sha256': session['record_sha256']})
        return {'status': 'CHANGE_OPEN', 'session': session, 'next_action': 'Edit a working source copy, then capture-revision.'}


def _archive_files(raw):
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise Error('CHANGE_ARCHIVE_DUPLICATE')
        return {sub._relative(n).as_posix(): z.read(n) for n in names}


def _delta(before, after):
    rows = []
    for name in sorted(set(before) | set(after)):
        a, b = before.get(name), after.get(name)
        if a == b:
            continue
        rows.append({'path': name, 'before_sha256': sub._sha(a) if a is not None else None,
                     'after_sha256': sub._sha(b) if b is not None else None,
                     'change': 'ADDED' if a is None else 'DELETED' if b is None else 'MODIFIED'})
    return rows


def _revision(folder, revision_id):
    if not isinstance(revision_id, str) or not re.fullmatch(r'[0-9a-f]{64}', revision_id):
        raise Error('CHANGE_REVISION_IDENTIFIER')
    return _verified(folder/'revisions'/(revision_id+'.json'))


def capture_revision(store_root, change_id, candidate, requirements=None, reason=''):
    store = Path(store_root).resolve(strict=True)
    with loop._workspace_lock(store):
        folder, session = _session(store, change_id)
        files = sub._tree(candidate)
        obj = sub._put_object(store, sub._archive(files), '.zip', 'candidate_source')
        # Preserve proposed bytes even if the subsequent requirements check refuses.
        _event(store, 'CANDIDATE_CAPTURED', {'change_id': change_id, 'source_object': obj})
        previous_path = folder/'LATEST.json'
        previous = _verified(previous_path)['revision_id'] if previous_path.exists() else None
        old = _revision(folder, previous)['requirements'] if previous else session['requirements']
        if isinstance(requirements, (str, Path)):
            requirements = sub._read(requirements)
        req = _requirements(requirements if requirements is not None else old)
        if previous:
            prev = _revision(folder, previous)
            if prev['source_object']['sha256'] == obj['sha256'] and prev['requirements'] == req:
                binding, workspace = _binding(store, folder, previous)
                state = sub.save_status(workspace)
                return {'status': state['status'], 'revision_id': previous, 'version': prev['version'],
                        'changed_files': prev['changed_files'], 'capture': state, 'reused_revision': True}
        req_change = req != old
        if req_change and not reason.strip():
            raise Error('REQUIREMENT_CHANGE_REASON_REQUIRED', next_action='Supply the reason; the old requirements and failure remain retained.')
        for node in req['nodes']:
            if node.split('::', 1)[0] not in files:
                raise Error('CHANGE_TEST_FILE_MISSING', node)
        sid, pid = loop._source_ids(Path(candidate))
        parent_files = _archive_files(_object(store, session['parent']['source_object']))
        comparison = _archive_files(_object(store, _revision(folder, previous)['source_object'])) if previous else parent_files
        test_changes = [r for r in _delta(comparison, files) if r['path'].startswith('tests/') or r['path'].endswith('conftest.py')]
        if test_changes and not reason.strip():
            raise Error('TEST_CODE_CHANGE_REASON_REQUIRED', next_action='Record why the checks changed; prior test bytes and failures remain retained.')
        body = {'schema_id': 'IG_DECODER_CHANGE_REVISION_V1', 'change_id': change_id,
                'session_sha256': session['record_sha256'], 'parent': session['parent'],
                'source_sha256': sid, 'package_sha256': pid, 'version': source_version(candidate),
                'source_object': obj, 'requirements': req,
                'requirements_change': {'before': old, 'after': req, 'reason': reason} if req_change else None,
                'test_code_changes': test_changes, 'test_code_change_reason': reason if test_changes else None,
                'changed_files': _delta(parent_files, files)}
        # Identical source/requirements is a reused revision, independent of chronology.
        revision = _sealed(body); rid = revision['record_sha256']
        _immutable(folder/'revisions'/(rid+'.json'), revision)
        rev_obj = _put_json(store, revision, 'change_revision')
        runner = Path(__file__).resolve().parents[1]
        session_obj = _put_json(store, session, 'change_session')
        env = dict(session['environment'], artifacts=[{'logical_name': row['logical_name'], 'sha256': row['sha256'],
                    'path': str(store/'objects'/row['object_name'])} for row in session['environment']['artifacts']])
        spec = {'schema_id': sub.SPEC_SCHEMA, 'job_id': 'CHANGE.'+rid[:24],
                'engine_source': str(runner), 'project_source': None,
                'question': {'stage_id': 'DECODER:ENGINEERING', 'description': req['statement'],
                             'outcomes': ['PASS', 'FAIL'], 'stopping_rule': 'Run only the frozen validation nodes once; retain failure and permit a new revision.'},
                'execution': {'kind': 'STAGE', 'handler_ref': 'infinity_grid.change_validation:validate_revision',
                              'evaluator_refs': [], 'parameters': {'revision_id': rid, 'workers': session['resources']['workers'],
                                                                'wall_seconds_max': session['resources']['wall_seconds_max']}},
                'resources': session['resources'], 'environment': env,
                'inputs': [{'logical_name': 'candidate_source', 'path': str(store/'objects'/obj['object_name']), 'sha256': obj['sha256']},
                           {'logical_name': 'change_revision', 'path': str(store/'objects'/rev_obj['object_name']), 'sha256': rev_obj['sha256']},
                           {'logical_name': 'change_session', 'path': str(store/'objects'/session_obj['object_name']), 'sha256': session_obj['sha256']},
                           {'logical_name': 'change_parent_source', 'path': str(store/'objects'/session['parent']['source_object']['object_name']),
                            'sha256': session['parent']['source_object']['sha256']}],
                'output_contract': {'requirements': req, 'revision_id': rid, 'acceptance': 'Retained controller completion and matching logs; candidate flags are not evidence.'}}
        # Validation belongs to the loaded runner's project, not the historical
        # parent release. Candidate/session inputs retain the parent binding;
        # activation still verifies that parent's exact release head separately.
        # The child store also avoids re-locking the change-session store.
        from . import portable_registry as project
        execution_store = store/'execution'
        if not (execution_store/'coordination/PROJECT.json').exists():
            project.initialize(execution_store, 'Recorded change validation', runner)
        state = sub.capture(execution_store, spec)
        binding = _sealed({'revision_id': rid, 'capture_id': state['capture_id'], 'job_id': state['job_id']})
        _immutable(folder/'bindings'/(rid+'.json'), binding)
        write_json_atomic(previous_path, _sealed({'revision_id': rid}))
        _event(store, 'REVISION_REGISTERED', {'change_id': change_id, 'revision_id': rid, 'previous_revision_id': previous,
                                          'capture_id': state['capture_id']})
        return {'status': state['status'], 'revision_id': rid, 'version': revision['version'],
                'changed_files': revision['changed_files'], 'capture': state}


def _original_binding(store, folder, rid):
    binding = _verified(folder/'bindings'/(rid+'.json'))
    if binding['revision_id'] != rid or not re.fullmatch(r'[0-9a-f]{64}', binding['capture_id']):
        raise Error('CHANGE_TEST_BINDING')
    workspace = store/'execution/captures'/binding['capture_id']
    rec = sub.capture_record(workspace)
    if (rec['job']['job_id'] != binding['job_id'] or
            rec['job']['execution'].get('handler_ref') != 'infinity_grid.change_validation:validate_revision' or
            rec['job']['execution'].get('parameters', {}).get('revision_id') != rid):
        raise Error('CHANGE_TEST_BINDING')
    return binding, workspace


def _binding(store, folder, rid):
    original, workspace = _original_binding(store, folder, rid)
    from .change_preservation import selected_binding
    return selected_binding(store, folder, rid, original, workspace)


def record_test_completion(store_root, change_id, revision_id):
    """Record an already executed retained-controller result; never run it."""
    store = Path(store_root).resolve(strict=True)
    with loop._workspace_lock(store):
        folder, session = _session(store, change_id)
        revision = _revision(folder, revision_id)
        binding, workspace = _binding(store, folder, revision_id)
        done = _completion(workspace, require_pass=False)
        if (done['result'].get('revision_id') != revision_id or
                done['result'].get('candidate_source_sha256') != revision['source_sha256'] or
                done['result'].get('requirements_sha256') != canonical_sha256(revision['requirements'])):
            raise Error('CHANGE_COMPLETION_CANDIDATE_BINDING')
        receipt = _sealed({'schema_id': 'IG_DECODER_CHANGE_TEST_RECEIPT_V1',
            'revision_id': revision_id, 'capture_id': binding['capture_id'],
            'completion_sha256': done['completion_sha256'],
            'outcome': done['result'].get('outcome'),
            'requirements_sha256': canonical_sha256(revision['requirements'])})
        _immutable(folder/'test_receipts'/(revision_id+'.json'), receipt)
        return {'status': 'PASS' if receipt['outcome'] == 'PASS' else 'VALIDATION_FAILED',
                'receipt': receipt, 'reused_execution': True}


def _control_hash(store):
    paths = [store/'CURRENT_LOCAL.json', *(store/'changes').rglob('*.json'), *(store/'pointers').glob('*.json')]
    return canonical_sha256({p.relative_to(store).as_posix(): sub._sha(p.read_bytes()) for p in paths if p.is_file()})


def test_revision(store_root, change_id, revision_id):
    store = Path(store_root).resolve(strict=True)
    with loop._workspace_lock(store):
        folder, session = _session(store, change_id); revision = _revision(folder, revision_id)
        binding, workspace = _binding(store, folder, revision_id)
        _object(store, revision['source_object'])
        control = _control_hash(store)
        _event(store, 'TEST_REQUESTED', {'change_id': change_id, 'revision_id': revision_id})
        try:
            result = loop.run_workspace_job(workspace, binding['job_id'])
            if _control_hash(store) != control:
                raise Error('CHANGE_CONTROL_STATE_MODIFIED_DURING_TEST')
            completion = _completion(workspace, require_pass=False)
            receipt = _sealed({'schema_id': 'IG_DECODER_CHANGE_TEST_RECEIPT_V1', 'revision_id': revision_id,
                               'capture_id': binding['capture_id'], 'completion_sha256': completion['completion_sha256'],
                               'outcome': completion['result'].get('outcome'), 'requirements_sha256': canonical_sha256(revision['requirements'])})
            _immutable(folder/'test_receipts'/(revision_id+'.json'), receipt)
            _event(store, 'TEST_FINISHED', {'change_id': change_id, **receipt})
            return {'status': 'PASS' if receipt['outcome'] == 'PASS' else 'VALIDATION_FAILED',
                    'receipt': receipt, 'completion': result}
        except BaseException as exc:
            _event(store, 'TEST_PAUSED', {'change_id': change_id, 'revision_id': revision_id, 'reason': str(exc)})
            raise


def diff_revision(store_root, change_id, revision_id):
    store = Path(store_root).resolve(strict=True); folder, session = _session(store, change_id)
    revision = _revision(folder, revision_id)
    before = _archive_files(_object(store, session['parent']['source_object']))
    after = _archive_files(_object(store, revision['source_object']))
    lines = []
    for row in revision['changed_files']:
        name = row['path']
        try:
            a = before.get(name, b'').decode('utf-8'); b = after.get(name, b'').decode('utf-8')
            lines.extend(difflib.unified_diff(a.splitlines(True), b.splitlines(True), fromfile='parent/'+name, tofile='candidate/'+name))
        except UnicodeDecodeError:
            lines.append('Binary revision: '+name+'\n')
    return {'revision_id': revision_id, 'changed_files': revision['changed_files'],
            'requirements_change': revision['requirements_change'], 'unified_diff': ''.join(lines)}


def _passed_revision(store, folder, rid):
    revision = _revision(folder, rid)
    _object(store, revision['source_object'])
    binding, workspace = _binding(store, folder, rid)
    receipt_path = folder/'test_receipts'/(rid+'.json')
    if not receipt_path.is_file():
        raise Error('CHANGE_TEST_RECEIPT_REQUIRED', next_action='Use test for this exact saved revision first.')
    receipt = _verified(receipt_path)
    done = _completion(workspace)
    if (receipt.get('revision_id') != rid or receipt.get('outcome') != 'PASS' or
            receipt.get('capture_id') != binding['capture_id'] or
            receipt.get('completion_sha256') != done['completion_sha256'] or
            receipt.get('requirements_sha256') != canonical_sha256(revision['requirements']) or
            done['result'].get('revision_id') != rid or
            done['result'].get('candidate_source_sha256') != revision['source_sha256']):
        raise Error('CHANGE_TEST_RECEIPT_MISMATCH')
    return revision, receipt


def _activation_execution_files(workspace):
    """Retain proof and checkpoint references, without duplicating payload objects."""
    workspace = Path(workspace)
    excluded = ('source/', 'runtime/intake/artifacts/', 'durability/outbox/objects/',
                'durability/base_objects/', 'coordination/')
    packet = {}
    for p in workspace.rglob('*'):
        rel = p.relative_to(workspace).as_posix()
        if p.is_file() and not rel.startswith(excluded) and not p.name.startswith('.'):
            packet['execution/'+rel] = p.read_bytes()
    packet['ACTIVATION_PACKAGE_SCOPE.json'] = sub._json_bytes({
        'scope': 'Exact activation proof, source/input references, checkpoint manifests and existing receipts.',
        'excluded_payload_prefixes': list(excluded),
        'standalone_recovery_archive': False,
        'recovery': 'Use native preserve export and its exact dependency closure; pending saves remain pending.'})
    return packet


def prepare_activation(store_root, change_id, revision_id):
    store = Path(store_root).resolve(strict=True)
    with loop._workspace_lock(store):
        folder, session = _session(store, change_id)
        rev, receipt = _passed_revision(store, folder, revision_id)
        current = _verified(store/'CURRENT_LOCAL.json')
        if current['record_sha256'] != session['parent_pointer_sha256']:
            raise Error('CHANGE_PARENT_MOVED', next_action='Reconcile with the current development parent and open a new session.')
        _, workspace = _binding(store, folder, revision_id)
        packet = {'SESSION.json': sub._json_bytes(session), 'REVISION.json': sub._json_bytes(rev),
                  'TEST_RECEIPT.json': sub._json_bytes(receipt), 'CURRENT_BEFORE.json': sub._json_bytes(current),
                  'CAPTURE.json': (workspace/'CAPTURE.json').read_bytes()}
        packet.update(_activation_execution_files(workspace))
        from .change_preservation import activation_amendment_files
        packet.update(activation_amendment_files(folder, revision_id))
        evidence_obj = sub._put_object(store, sub._archive(packet), '.zip', 'activation_evidence')
        prepared = _sealed({'schema_id': 'IG_DECODER_PREPARED_ACTIVATION_V1', 'change_id': change_id,
                            'revision_id': revision_id, 'parent_pointer_sha256': current['record_sha256'],
                            'receipt_sha256': receipt['record_sha256'], 'target': {'version': rev['version'],
                              'source_sha256': rev['source_sha256'], 'package_sha256': rev['package_sha256'],
                              'source_object': rev['source_object'], 'completion_sha256': receipt['completion_sha256']},
                            'evidence_object': evidence_obj, 'scope': 'ISOLATED_LOCAL_DEVELOPMENT_NOT_SHARED_AUTHORITY'})
        _immutable(folder/'prepared'/(prepared['record_sha256']+'.json'), prepared)
        _event(store, 'PREPARE_ACTIVATION', {'change_id': change_id, 'prepared_sha256': prepared['record_sha256']})
        return {'status': 'PREPARED_SAVE_REQUIRED', 'prepared': prepared,
                'pending_objects': [dict(evidence_obj, local_object_path=str(store/'objects'/evidence_obj['object_name']))],
                'next_action': 'Save this evidence object; use change confirm-save before activate.'}


def confirm_object_save(store_root, object_name, readback, drive_file_id):
    store = Path(store_root).resolve(strict=True)
    path = store/'objects'/sub._relative(object_name)
    raw = path.read_bytes(); digest = sub._sha(raw)
    if path.name.split('.')[0] != digest or Path(readback).read_bytes() != raw:
        raise Error('SAVE_READBACK_MISMATCH')
    if not re.fullmatch(r'[A-Za-z0-9_-]{10,200}', drive_file_id):
        raise Error('DRIVE_FILE_ID_REQUIRED')
    receipt = {'schema_id': 'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V1', 'provider': 'google_drive',
               'sha256': digest, 'size_bytes': len(raw), 'drive_file_id': drive_file_id,
               'raw_readback_verified': True, 'recorded_ns': time.time_ns(),
               'scope': 'EXPLICIT_CONNECTOR_OPERATOR_ATTESTATION_NOT_REMOTE_AUTHENTICATION'}
    receipt['receipt_sha256'] = canonical_sha256(receipt)
    _immutable(store/'receipts'/digest/(receipt['receipt_sha256']+'.json'), receipt)
    return {'status': 'SAVED', 'receipt': receipt}


def _move_pointer(store, current, target, operation, details):
    new = _sealed({'schema_id': current['schema_id'], 'epoch': current['epoch']+1, 'target': target,
                   'previous_pointer_sha256': current['record_sha256'], 'scope': current['scope']})
    _immutable(store/'pointers'/(new['record_sha256']+'.json'), new)
    # History object is written first. A crash before replace keeps the old pointer.
    write_json_atomic(store/'CURRENT_LOCAL.json', new)
    _event(store, operation, {'previous': current['record_sha256'], 'current': new['record_sha256'], **details})
    return {'status': operation, 'current': new}


def activate(store_root, change_id, prepared_sha256):
    store = Path(store_root).resolve(strict=True)
    if not re.fullmatch(r'[0-9a-f]{64}', prepared_sha256):
        raise Error('CHANGE_PREPARED_IDENTIFIER')
    with loop._workspace_lock(store):
        folder, session = _session(store, change_id)
        prepared = _verified(folder/'prepared'/(prepared_sha256+'.json'))
        current = _verified(store/'CURRENT_LOCAL.json')
        if prepared['change_id'] != change_id or current['record_sha256'] != prepared['parent_pointer_sha256']:
            raise Error('CHANGE_PARENT_MOVED')
        rev, receipt = _passed_revision(store, folder, prepared['revision_id'])
        if receipt['record_sha256'] != prepared['receipt_sha256'] or rev['source_sha256'] != prepared['target']['source_sha256']:
            raise Error('CHANGE_PREPARATION_MISMATCH')
        obj = prepared['evidence_object']; _object(store, obj)
        if not any(sub._valid_receipt(p, obj) for p in (store/'receipts'/obj['sha256']).glob('*.json')):
            raise Error('ACTIVATION_SAVE_REQUIRED', next_action='Save and confirm the prepared evidence object before activation.')
        from . import portable_registry as project
        if 'project_parent_head' not in session:
            raise Error('PROJECT_CHANGE_MIGRATION_REQUIRED')
        _,validation_workspace=_binding(store,folder,prepared['revision_id'])
        with tempfile.TemporaryDirectory(prefix='ig-project-activation-') as directory:
            candidate=Path(directory)
            sub._write_files(candidate,_archive_files(_object(store,rev['source_object'])))
            project.promote(store/'coordination',session['project_parent_head'],candidate,validation_workspace,session['purpose'])
        return _move_pointer(store, current, prepared['target'], 'ACTIVATED_LOCAL', {'change_id': change_id, 'prepared_sha256': prepared_sha256})


def rollback(store_root, expected_current, reason):
    store = Path(store_root).resolve(strict=True)
    if not reason.strip():
        raise Error('ROLLBACK_REASON_REQUIRED')
    with loop._workspace_lock(store):
        current = _verified(store/'CURRENT_LOCAL.json')
        if current['record_sha256'] != expected_current:
            raise Error('ROLLBACK_CURRENT_MOVED')
        previous_id = current['previous_pointer_sha256']
        if previous_id is None:
            raise Error('ROLLBACK_NO_PREVIOUS_VERSION')
        previous = _verified(store/'pointers'/(previous_id+'.json'))
        _object(store, previous['target']['source_object'])
        from . import portable_registry as project
        project.rollback_release(store/'coordination',current['target']['source_object']['sha256'],previous['target']['source_object']['sha256'],reason)
        return _move_pointer(store, current, previous['target'], 'ROLLED_BACK_LOCAL', {'reason': reason})


def status(store_root, change_id):
    store = Path(store_root).resolve(strict=True); folder, session = _session(store, change_id)
    revisions = [_verified(p) for p in sorted((folder/'revisions').glob('*.json'))]
    return {'session': session, 'current': _verified(store/'CURRENT_LOCAL.json'),
            'revisions': revisions, 'test_receipts': [_verified(p) for p in sorted((folder/'test_receipts').glob('*.json'))],
            'scope': 'Local workflow checks; shared authority and filesystem ownership are Step5.'}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Decoder recorded engine changes and recovery; engineering validation only.')
    from ._version import __version__
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('open'); p.add_argument('store'); p.add_argument('request')
    p = commands.add_parser('capture-revision'); p.add_argument('store'); p.add_argument('change_id'); p.add_argument('candidate'); p.add_argument('--requirements'); p.add_argument('--reason', default='')
    for name in ('test', 'diff', 'prepare-activation', 'record-test-completion'):
        p = commands.add_parser(name); p.add_argument('store'); p.add_argument('change_id'); p.add_argument('revision_id')
    p = commands.add_parser('activate'); p.add_argument('store'); p.add_argument('change_id'); p.add_argument('prepared_sha256')
    p = commands.add_parser('rollback'); p.add_argument('store'); p.add_argument('expected_current'); p.add_argument('reason')
    p = commands.add_parser('status'); p.add_argument('store'); p.add_argument('change_id')
    p = commands.add_parser('confirm-save'); p.add_argument('store'); p.add_argument('object_name'); p.add_argument('readback'); p.add_argument('drive_file_id')
    args = parser.parse_args(argv); values = vars(args); command = values.pop('command')
    functions = {'open': open_change, 'capture-revision': capture_revision, 'test': test_revision,
                 'diff': diff_revision, 'prepare-activation': prepare_activation, 'activate': activate,
                 'rollback': rollback, 'status': status, 'confirm-save': confirm_object_save,
                 'record-test-completion': record_test_completion}
    values['store_root'] = values.pop('store')
    try:
        result = functions[command](**values)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 1 if result.get('status') == 'VALIDATION_FAILED' else 0
    except Exception as exc:
        result = {'status': 'PAUSED', 'operation': command, 'reason': str(exc)}
        if isinstance(exc, Error):
            result.update(code=exc.code, next_action=exc.next_action)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 2
