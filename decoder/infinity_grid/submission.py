"""Capture exact revisions and verify explicit connector save attestations.

This is workflow integrity in an editable workspace, not remote authentication
or a Python sandbox. All execution remains in the registered controller.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

from .canon import canonical_sha256, write_json_atomic
from ._version import __version__

SKIP = {'.git', '__pycache__', '.pytest_cache', 'build', 'dist', '.engineering_tmp'}
SPEC_SCHEMA = 'IG_DECODER_CAPTURE_SPEC_V1'
CAPTURE_SCHEMA = 'IG_DECODER_CAPTURE_V1'
ADMINISTRATIVE_INPUT_NAMES = frozenset({'submission_contract', 'runtime_dependency_snapshot'})


class SubmissionError(RuntimeError):
    def __init__(self, code, detail='', next_action='Inspect the saved submission and capture a corrected revision.'):
        self.code, self.detail, self.next_action = code, detail, next_action
        super().__init__(code + (':' + detail if detail else ''))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def _read(path):
    from .v05_controller_event_loop import _json_object
    return _json_object(Path(path))


def _relative(value):
    p = Path(value)
    if not value or p.is_absolute() or '..' in p.parts or '\\' in value or p.as_posix() != value:
        raise SubmissionError('CAPTURE_RELATIVE_PATH', str(value))
    return p


def _tree(root):
    root = Path(root).resolve(strict=True)
    out = {}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(x in SKIP for x in rel.parts) or path.suffix in {'.pyc', '.pyo'}:
            continue
        if path.is_symlink():
            raise SubmissionError('CAPTURE_SYMLINK', rel.as_posix())
        if path.is_file():
            out[rel.as_posix()] = path.read_bytes()
    return out


def _archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(files.items()):
            _relative(name)
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, raw)
    return stream.getvalue()


def _put_object(store, raw, suffix, role):
    digest = _sha(raw)
    path = store / 'objects' / (digest + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise SubmissionError('CONTENT_STORE_MISMATCH', digest)
    else:
        # Independent captures may submit identical bytes concurrently.
        fd, temporary = tempfile.mkstemp(prefix='.object-', dir=path.parent)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(raw); f.flush(); os.fsync(f.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != raw:
                    raise SubmissionError('CONTENT_STORE_MISMATCH', digest)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return {'role': role, 'sha256': digest, 'size_bytes': len(raw), 'object_name': path.name}


def _validate_environment(env):
    if set(env) != {'python', 'requirements', 'artifacts'} or not re.fullmatch(r'3\.\d+', env['python']):
        raise SubmissionError('CAPTURE_ENVIRONMENT_FIELDS')
    seen = set()
    for row in env['requirements']:
        if set(row) != {'distribution', 'version', 'imports'} or not row['version'] or not row['imports']:
            raise SubmissionError('CAPTURE_REQUIREMENT_FIELDS')
        if row['distribution'] in seen:
            raise SubmissionError('DUPLICATE_REQUIREMENT', row['distribution'])
        seen.add(row['distribution'])
        if not all(re.fullmatch(r'[A-Za-z_]\w*', x) for x in row['imports']):
            raise SubmissionError('CAPTURE_IMPORT_ROOT')


def _project_closure(files, env):
    """Conservative project dependency admission; engine bytes are frozen separately."""
    roots = {Path(n).parts[0].removesuffix('.py') for n in files if n.endswith('.py')}
    allowed = set(sys.stdlib_module_names) | roots | {'infinity_grid'}
    for row in env['requirements']:
        allowed.update(row['imports'])
    for name, raw in files.items():
        if not name.endswith('.py'):
            continue
        try:
            tree = ast.parse(raw, filename=name)
        except (SyntaxError, ValueError) as exc:
            raise SubmissionError('PROJECT_SYNTAX', name + ':' + str(exc)) from exc
        for node in ast.walk(tree):
            imported = []
            if isinstance(node, ast.Import):
                imported = [n.name for n in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported = [node.module]
            for mod in imported:
                if mod.split('.')[0] not in allowed:
                    raise SubmissionError('UNRESOLVED_EXECUTABLE_DEPENDENCY', name + ':' + mod,
                        'Include the project helper, or declare and save the exact external requirement, then capture again.')
            if isinstance(node, ast.Call):
                function = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
                if function in {'exec', 'eval', 'compile', '__import__', 'import_module', 'run_path', 'run_module', 'spec_from_file_location'}:
                    raise SubmissionError('GENERATED_CODE_REQUIRES_CAPTURE', name + ':' + function,
                        'Generate executable text as a saved file first; capture that file and use an ordinary registered entry/import.')


def _write_files(root, files):
    for name, raw in files.items():
        target = root / _relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)


def _failure(store, operation, exc, *, preservation=None):
    # Preparation/save failures are events, never successful run completions.
    row = {'schema_id': 'IG_DECODER_CAPTURE_EVENT_V1', 'operation': operation,
           'status': 'REFUSED', 'reason': str(exc), 'recorded_ns': time.time_ns()}
    if preservation is not None:
        row['preservation'] = preservation
    write_json_atomic(store / 'events' / (canonical_sha256(row) + '.json'), row)
    return row


def capture(store_root, specification):
    from . import v05_controller_event_loop as loop
    store = Path(store_root).resolve(); store.mkdir(parents=True, exist_ok=True)
    with loop._workspace_lock(store):
        preserved = []
        try:
            spec = _read(specification) if isinstance(specification, (str, Path)) else dict(specification)
            # Keep the original intent even if shape/dependency admission fails.
            preserved.append(_put_object(store, _json_bytes(spec), '.json', 'submitted_specification'))
            fields = {'schema_id', 'job_id', 'engine_source', 'project_source', 'question',
                      'execution', 'resources', 'inputs', 'environment', 'output_contract'}
            if set(spec) not in (fields, fields | {'repeat_id'}) or spec['schema_id'] != SPEC_SCHEMA:
                raise SubmissionError('CAPTURE_SPEC_FIELDS')
            canonical_sha256(spec)
            if not loop._JOB_ID.fullmatch(spec['job_id']):
                raise SubmissionError('JOB_IDENTIFIER')
            if not isinstance(spec['output_contract'], dict) or not spec['output_contract']:
                raise SubmissionError('OUTPUT_CONTRACT_REQUIRED')
            from .result_contracts import normalize
            effective_contract=normalize(spec['output_contract'],spec['execution'],spec['question'])
            _validate_environment(spec['environment'])
            engine = _tree(spec['engine_source'])
            if not any(n.startswith('infinity_grid/') for n in engine) or any(n.startswith('project/') for n in engine):
                raise SubmissionError('ENGINE_SOURCE_LAYOUT', 'A clean engine source tree is required; project/ is reserved.')
            project = _tree(spec['project_source']) if spec['project_source'] else {}
            objects = [_put_object(store, _archive(engine), '.zip', 'engine_source')]
            if project:
                objects.append(_put_object(store, _archive(project), '.zip', 'project_source'))
            preserved.extend(objects)
            _project_closure(project, spec['environment'])
            input_rows = []; artifact_bytes = {}
            # Project data and Decoder/runtime administration are different
            # namespaces.  This prevents a generated dependency snapshot or
            # submission contract from being mistaken for missing science data.
            for row in spec['inputs']:
                if (set(row) != {'logical_name', 'path', 'sha256'} or
                        row['logical_name'] in artifact_bytes or
                        row['logical_name'] in ADMINISTRATIVE_INPUT_NAMES):
                    raise SubmissionError('CAPTURE_INPUT_FIELDS')
                raw = Path(row['path']).read_bytes()
                preserved.append(_put_object(store, raw, '.bin', 'submitted_input:' + row['logical_name']))
                if _sha(raw) != row['sha256']:
                    raise SubmissionError('INPUT_HASH_MISMATCH', row['logical_name'])
                obj = _put_object(store, raw, '.bin', 'input:' + row['logical_name'])
                objects.append(obj); artifact_bytes[row['logical_name']] = raw
                input_rows.append({'logical_name': row['logical_name'], 'sha256': obj['sha256']})
            for row in spec['environment']['artifacts']:
                if (set(row) != {'logical_name', 'path', 'sha256'} or
                        row['logical_name'] in artifact_bytes or
                        row['logical_name'] == 'submission_contract'):
                    raise SubmissionError('CAPTURE_INPUT_FIELDS')
                raw = Path(row['path']).read_bytes()
                preserved.append(_put_object(store, raw, '.bin', 'submitted_input:' + row['logical_name']))
                if _sha(raw) != row['sha256']:
                    raise SubmissionError('INPUT_HASH_MISMATCH', row['logical_name'])
                obj = _put_object(store, raw, '.bin', 'input:' + row['logical_name'])
                objects.append(obj); artifact_bytes[row['logical_name']] = raw
                input_rows.append({'logical_name': row['logical_name'], 'sha256': obj['sha256']})
            env = dict(spec['environment'], artifacts=[{'logical_name': r['logical_name'], 'sha256': r['sha256']} for r in spec['environment']['artifacts']])
            contract = _json_bytes({'schema_id': 'IG_DECODER_SUBMISSION_CONTRACT_V1',
                                   'environment': env, 'output_contract': spec['output_contract'], 'result_contract': effective_contract})
            contract_obj = _put_object(store, contract, '.bin', 'input:submission_contract')
            objects.append(contract_obj); artifact_bytes['submission_contract'] = contract
            input_rows.append({'logical_name': 'submission_contract', 'sha256': contract_obj['sha256']})
            temp = Path(tempfile.mkdtemp(prefix='.capture-', dir=store))
            try:
                _write_files(temp / 'source', engine)
                _write_files(temp / 'source/project', project)
                sid, pid = loop._source_ids(temp / 'source')
                work = {'schema_id': loop.WORKSPACE_SCHEMA, 'source_sha256': sid, 'package_sha256': pid}
                job = {'schema_id': loop.JOB_SCHEMA, 'job_id': spec['job_id'], 'source_sha256': sid,
                       'question': spec['question'], 'input_artifacts': input_rows,
                       'execution': spec['execution'], 'resources': spec['resources']}
                job['registration_sha256'] = canonical_sha256(job)
                body = {'schema_id': CAPTURE_SCHEMA, 'decoder_version': __version__, 'workspace': work,
                        'job': job, 'objects': objects, 'environment': env,
                        'output_contract': spec['output_contract'], 'result_contract': effective_contract,
                        'execution_scope': 'REGISTERED_WORKSPACE; frozen checks enforced; descriptive legacy contracts are execution-only',
                        'save_policy': 'EXPLICIT_GOOGLE_DRIVE_CONNECTOR_READBACK_REQUIRED'}
                body['repeat_id'] = spec.get('repeat_id')
                if (store/'coordination/PROJECT.json').exists():
                    from .portable_registry import capture_context, lock as project_lock
                    with project_lock(store/'coordination'):
                        body['project_context']=capture_context(store/'coordination',objects[0]['sha256'])
                record = dict(body, capture_id=canonical_sha256(body))
                raw_record = _json_bytes(record)
                _put_object(store, raw_record, '.json', 'capture_record')
                write_json_atomic(temp / 'WORKSPACE.json', work)
                write_json_atomic(temp / 'registry' / (job['job_id'] + '.json'), job)
                (temp / 'CAPTURE.json').write_bytes(raw_record)
                for row in input_rows:
                    target = temp / 'runtime/intake/artifacts' / (row['sha256'] + '.bin')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(artifact_bytes[row['logical_name']])
                loop.validate_workspace_job(temp, job['job_id'], check_loaded=False)
                dest = store / 'captures' / record['capture_id']; dest.parent.mkdir(exist_ok=True)
                if dest.exists():
                    if (dest / 'CAPTURE.json').read_bytes() != raw_record:
                        raise SubmissionError('CAPTURE_IDENTITY_CONFLICT')
                else:
                    os.replace(temp, dest)
                for obj in required_objects(dest):
                    prior = store / 'receipts' / obj['sha256']
                    if prior.exists():
                        shutil.copytree(prior, dest / 'durability/receipts' / obj['sha256'], dirs_exist_ok=True)
                if (store/'coordination/PROJECT.json').exists():
                    from .portable_registry import adopt
                    adopt(store,dest,spec.get('repeat_id'))
                return save_status(dest)
            finally:
                if temp.exists():
                    shutil.rmtree(temp)
        except Exception as exc:
            preservation = {'status': 'CAPTURE_REFUSED_BYTES_RETAINED', 'objects': preserved,
                            'pending_save_objects': [dict(o, local_object_path=str(store/'objects'/o['object_name'])) for o in preserved],
                            'execution_authorized': False,
                            'next_action': 'Save the refusal event and listed objects through the connector; revise the captured intent. Saving a refused capture never authorizes execution.'}
            _failure(store, 'CAPTURE', exc, preservation=preservation)
            if isinstance(exc, SubmissionError):
                exc.preservation = preservation
            raise


def capture_record(workspace):
    root = Path(workspace).resolve(strict=True)
    if not (root / 'CAPTURE.json').is_file():
        raise SubmissionError('CAPTURE_REQUIRED', str(root),
            'Use Decoder capture STORE SPEC.json, then confirm each pending Drive save.')
    rec = _read(root / 'CAPTURE.json')
    body = {k: v for k, v in rec.items() if k != 'capture_id'}
    if rec.get('schema_id') != CAPTURE_SCHEMA or canonical_sha256(body) != rec.get('capture_id'):
        raise SubmissionError('CAPTURE_RECORD_MISMATCH')
    if _read(root / 'WORKSPACE.json') != rec['workspace'] or _read(root / 'registry' / (rec['job']['job_id'] + '.json')) != rec['job']:
        raise SubmissionError('CAPTURE_BINDING_MISMATCH')
    return rec


def required_objects(workspace):
    root = Path(workspace); rec = capture_record(root)
    raw = (root / 'CAPTURE.json').read_bytes()
    rows = rec['objects'] + [{'role': 'capture_record', 'sha256': _sha(raw), 'size_bytes': len(raw), 'object_name': _sha(raw) + '.json'}]
    from .portable_registry import required_objects as project_objects
    rows += project_objects(root)
    # Physical bytes are content-deduplicated, but save obligations are not.
    out=[]
    for index,row in enumerate(rows):
        logical = row.get('logical_name') or row.get('object_name') or f'obligation-{index}'
        item=dict(row, obligation_scope='CAPTURE_SAVE', logical_name=logical)
        item['obligation_id']=canonical_sha256({'capture_id':rec['capture_id'],'scope':'CAPTURE_SAVE',
            'role':item['role'],'logical_name':logical,'sha256':item['sha256'],'size_bytes':item['size_bytes']})
        out.append(item)
    return out


# Exact, audited historical producer: capture-save obligations were physical
# content identities, not role identities. This is not a version-range bypass.
_LEGACY_PHYSICAL_CAPTURE_PRODUCERS = {
    'a27badfc1a353fedac60ee7320295e02c1106c2b53254a3464ba414099f70a84':
        ('0.6.0', 3815166),
}


def _verified_legacy_physical_capture(root, rec):
    """Recognize unchanged producer bytes; never execute old code or edit receipts.

    Capsule verification independently checks the registration and completion.
    Checking the actual frozen source archive here prevents metadata/version
    labels alone from selecting weaker historical receipt semantics.
    """
    producers = [o for o in rec.get('objects', ()) if o.get('role') == 'engine_source']
    if len(producers) != 1:
        return False
    producer = producers[0]
    known = _LEGACY_PHYSICAL_CAPTURE_PRODUCERS.get(producer.get('sha256'))
    if known is None or (rec.get('decoder_version'), producer.get('size_bytes')) != known:
        return False
    try:
        files = {n:b for n,b in _tree(Path(root)/'source').items() if not n.startswith('project/')}
        actual = _archive(files)
    except (OSError, ValueError, SubmissionError):
        return False
    return len(actual) == known[1] and _sha(actual) == producer['sha256']


def _valid_receipt(path, obj, *, ambiguous=False, legacy_physical=False):
    try:
        row = _read(path)
    except (ValueError, OSError):
        return False
    from .save_transport import RECEIPT, valid_receipt
    if row.get('schema_id') == RECEIPT:
        bound=(row.get('obligation_id')==obj.get('obligation_id')
            and row.get('role')==obj.get('role')
            and row.get('logical_name')==obj.get('logical_name')
            and row.get('obligation_scope')==obj.get('obligation_scope'))
        legacy_unbound='obligation_id' not in row
        return valid_receipt(row,obj) and (bound or not ambiguous and legacy_unbound)
    if row.get('schema_id') == 'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2':
        body={k:v for k,v in row.items() if k!='receipt_sha256'}
        return (row.get('sha256')==obj['sha256'] and row.get('size_bytes')==obj['size_bytes']
            and row.get('obligation_id')==obj.get('obligation_id') and row.get('role')==obj.get('role')
            and row.get('obligation_scope')==obj.get('obligation_scope')
            and row.get('provider')=='google_drive' and row.get('raw_readback_verified') is True
            and canonical_sha256(body)==row.get('receipt_sha256'))
    body = {k: v for k, v in row.items() if k != 'receipt_sha256'}
    return ((not ambiguous or legacy_physical)
            and not any(k in row for k in ('obligation_id','obligation_scope','role','logical_name'))
            and row.get('schema_id') == 'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V1'
            and row.get('sha256') == obj['sha256'] and row.get('size_bytes') == obj['size_bytes']
            and row.get('provider') == 'google_drive' and row.get('raw_readback_verified') is True
            and re.fullmatch(r'[A-Za-z0-9_-]{10,200}', str(row.get('drive_file_id', ''))) is not None
            and canonical_sha256(body) == row.get('receipt_sha256'))


def save_status(workspace):
    root = Path(workspace).resolve(strict=True); rec = capture_record(root)
    pending = []
    required=required_objects(root); counts={}
    for obj in required: counts[obj['sha256']]=counts.get(obj['sha256'],0)+1
    legacy_physical = any(n > 1 for n in counts.values()) and _verified_legacy_physical_capture(root, rec)
    for obj in required:
        receipts = root / 'durability/receipts' / obj['sha256']
        if not any(_valid_receipt(p, obj, ambiguous=counts[obj['sha256']]>1,
                                  legacy_physical=legacy_physical) for p in receipts.glob('*.json')):
            pending.append(dict(obj, local_object_path=str(root.parent.parent / 'objects' / obj['object_name'])))
    return {'status': 'SAVE_REQUIRED' if pending else 'SAVED', 'capture_status': 'CAPTURED',
            'workspace': str(root), 'capture_id': rec['capture_id'], 'job_id': rec['job']['job_id'],
            'decoder_version': rec['decoder_version'], 'pending_objects': pending,
            'next_action': 'Upload pending objects through the Drive connector; download raw bytes and use confirm-save.' if pending else 'Invoke Decoder run with this saved workspace and job.',
            'acknowledgment_scope': 'Explicit connector/operator attestation plus local byte verification; not remote identity authentication.'}


def confirm_save(workspace, digest, readback, drive_file_id, *, role=None, logical_name=None):
    from .v05_controller_event_loop import _workspace_lock
    root = Path(workspace).resolve(strict=True)
    with _workspace_lock(root):
        try:
            matches=[x for x in required_objects(root) if x['sha256']==digest]
            if role is not None: matches=[x for x in matches if x['role']==role]
            if logical_name is not None: matches=[x for x in matches if x['logical_name']==logical_name]
            if not matches:
                raise SubmissionError('SAVE_OBJECT_NOT_REQUIRED', digest)
            if len(matches)!=1: raise SubmissionError('SAVE_OBLIGATION_AMBIGUOUS',digest,'Supply role and logical_name.')
            obj=matches[0]
            if not re.fullmatch(r'[A-Za-z0-9_-]{10,200}', drive_file_id):
                raise SubmissionError('DRIVE_FILE_ID_REQUIRED')
            raw = Path(readback).read_bytes()
            if _sha(raw) != digest or len(raw) != obj['size_bytes']:
                raise SubmissionError('SAVE_READBACK_MISMATCH', digest, 'Keep this submission paused; download the correct Drive object and confirm its bytes.')
            receipt = {'schema_id': 'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2', 'provider': 'google_drive',
                       'sha256': digest, 'size_bytes': len(raw), 'drive_file_id': drive_file_id,
                       'obligation_id':obj['obligation_id'],'obligation_scope':obj['obligation_scope'],
                       'role':obj['role'],'logical_name':obj['logical_name'],
                       'raw_readback_verified': True, 'recorded_ns': time.time_ns(),
                       'scope': 'EXPLICIT_CONNECTOR_OPERATOR_ATTESTATION_NOT_REMOTE_AUTHENTICATION'}
            receipt['receipt_sha256'] = canonical_sha256(receipt)
            name = receipt['receipt_sha256'] + '.json'
            write_json_atomic(root / 'durability/receipts' / digest / name, receipt)
            store = root.parent.parent
            if root.parent.name == 'captures' and (store / 'objects' / obj['object_name']).is_file():
                write_json_atomic(store / 'receipts' / digest / name, receipt)
            return save_status(root)
        except Exception as exc:
            _failure(root / 'durability', 'CONFIRM_SAVE', exc)
            raise


def record_save_failure(workspace, digest, reason):
    root = Path(workspace).resolve(strict=True)
    if digest not in {x['sha256'] for x in required_objects(root)}:
        raise SubmissionError('SAVE_OBJECT_NOT_REQUIRED', digest)
    _failure(root / 'durability', 'SAVE:' + digest, SubmissionError('SAVE_FAILED', reason))
    return save_status(root)


def confirm_transport(workspace, digest, manifest, parts, drive_file_id):
    from .v05_controller_event_loop import _workspace_lock
    from .save_transport import make_receipt
    root = Path(workspace).resolve(strict=True)
    with _workspace_lock(root):
        try:
            matches=[x for x in required_objects(root) if x['sha256']==digest]
            if not matches:
                raise SubmissionError('SAVE_OBJECT_NOT_REQUIRED', digest)
            if len(matches)!=1:
                raise SubmissionError('SAVE_OBLIGATION_AMBIGUOUS',digest,
                    'Multipart transport cannot satisfy multiple roles implicitly; confirm each role-aware obligation.')
            obj=matches[0]
            receipt = make_receipt(obj, manifest, parts, drive_file_id)
            name = receipt['receipt_sha256'] + '.json'
            write_json_atomic(root / 'durability/receipts' / digest / name, receipt)
            store = root.parent.parent
            if root.parent.name == 'captures' and (store / 'objects' / obj['object_name']).is_file():
                write_json_atomic(store / 'receipts' / digest / name, receipt)
            return save_status(root)
        except Exception as exc:
            _failure(root / 'durability', 'CONFIRM_TRANSPORT', exc)
            raise


def require_environment(rec):
    """Require the captured runtime only when scientific execution may occur."""
    if rec['environment']['python'] != f'{sys.version_info.major}.{sys.version_info.minor}':
        raise SubmissionError('PYTHON_ENVIRONMENT_MISMATCH', rec['environment']['python'])
    for row in rec['environment']['requirements']:
        try:
            actual = importlib.metadata.version(row['distribution'])
        except importlib.metadata.PackageNotFoundError:
            actual = 'NOT_INSTALLED'
        if actual != row['version']:
            raise SubmissionError('REQUIREMENT_MISMATCH', row['distribution'] + ':' + actual,
                'Install the captured environment requirements, or capture a deliberate environment revision.')


def require_saved(workspace, job_id, *, check_environment=True):
    """Verify save evidence, and optionally require the execution environment.

    Exact completion reuse is an evidence-verification operation and does not
    import or execute the captured scientific source.  Callers performing that
    operation may defer environment admission until they know execution is
    necessary.
    """
    root = Path(workspace); rec = capture_record(root)
    if rec['job']['job_id'] != job_id:
        raise SubmissionError('CAPTURE_JOB_MISMATCH', job_id)
    state = save_status(root)
    if state['pending_objects']:
        raise SubmissionError('SAVE_REQUIRED', ','.join(x['sha256'] for x in state['pending_objects']),
            'Use pending-saves WORKSPACE, upload/download the listed objects, then confirm-save each hash before run.')
    if check_environment:
        require_environment(rec)
    return rec


def restore_capture(capture_file, object_directory, destination):
    """Reconstruct exact pre-execution files from verified saved content objects."""
    from . import v05_controller_event_loop as loop
    raw = Path(capture_file).read_bytes(); rec = _read(capture_file)
    if rec.get('schema_id') != CAPTURE_SCHEMA or canonical_sha256({k: v for k, v in rec.items() if k != 'capture_id'}) != rec.get('capture_id'):
        raise SubmissionError('CAPTURE_RECORD_MISMATCH')
    dest = Path(destination).resolve()
    if dest.exists():
        raise SubmissionError('RESTORE_DESTINATION_EXISTS')
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.restore-capture-', dir=dest.parent))
    try:
        for obj in rec['objects']:
            name = _relative(obj['object_name'])
            data = (Path(object_directory) / name).read_bytes()
            if _sha(data) != obj['sha256'] or len(data) != obj['size_bytes']:
                raise SubmissionError('RESTORE_OBJECT_MISMATCH', obj['sha256'])
            if obj['role'] in {'engine_source', 'project_source'}:
                target = temp / 'source' if obj['role'] == 'engine_source' else temp / 'source/project'
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    if len(z.namelist()) != len(set(z.namelist())):
                        raise SubmissionError('RESTORE_DUPLICATE_PATH')
                    _write_files(target, {n: z.read(n) for n in z.namelist()})
            else:
                target = temp / 'runtime/intake/artifacts' / (obj['sha256'] + '.bin')
                target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data)
        write_json_atomic(temp / 'WORKSPACE.json', rec['workspace'])
        write_json_atomic(temp / 'registry' / (rec['job']['job_id'] + '.json'), rec['job'])
        (temp / 'CAPTURE.json').write_bytes(raw)
        if loop._source_ids(temp / 'source') != (rec['workspace']['source_sha256'], rec['workspace']['package_sha256']):
            raise SubmissionError('RESTORE_SOURCE_MISMATCH')
        os.replace(temp, dest)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    return save_status(dest)
