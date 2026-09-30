"""Native checkpoint/outbox bookkeeping; transport is an explicit capability."""
from contextlib import contextmanager, closing
from contextvars import ContextVar
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tempfile
import time
import zipfile

from .canon import canonical_bytes, canonical_sha256, write_json_atomic
from . import submission as sub

_ACTIVE = ContextVar('decoder_preservation', default=None)
SCHEMA = 'IG_DECODER_CHECKPOINT_V1'


def _zip_files(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if len(z.namelist())!=len(set(z.namelist())):raise sub.SubmissionError('CHECKPOINT_DUPLICATE_PATH')
        for n in z.namelist():sub._relative(n)
        return {n:z.read(n) for n in z.namelist()}


def _project_delta(base,full):
    old=_zip_files(base);new=_zip_files(full)
    if any(new.get(k)!=v for k,v in old.items()):raise sub.SubmissionError('PROJECT_BASELINE_NOT_RETAINED')
    return sub._archive({k:v for k,v in new.items() if k not in old})


def _restore_project(base,delta,expected):
    files=_zip_files(base)
    for name,raw in _zip_files(delta).items():
        if name in files and files[name]!=raw:raise sub.SubmissionError('PROJECT_DELTA_OVERWRITE')
        files[name]=raw
    raw=sub._archive(files)
    if sub._sha(raw)!=expected:raise sub.SubmissionError('PROJECT_DELTA_HASH')
    return raw


def _root(workspace): return Path(workspace).resolve() / 'durability/outbox'


def _put(workspace, raw):
    row = sub._put_object(_root(workspace), raw, '.bin', 'checkpoint_object')
    return {'sha256': row['sha256'], 'size_bytes': row['size_bytes']}


def _bytes(workspace, row):
    path = _root(workspace) / 'objects' / (row['sha256'] + '.bin')
    if not path.is_file():raise sub.SubmissionError('OUTBOX_OBJECT_MISSING',row['sha256'])
    raw = path.read_bytes()
    if sub._sha(raw) != row['sha256'] or len(raw) != row['size_bytes']:
        raise sub.SubmissionError('OUTBOX_OBJECT_MISMATCH', row['sha256'])
    return raw


def _receipt(workspace, row, *, ambiguous=False):
    from .save_transport import RECEIPT as TRANSPORT_RECEIPT, valid_receipt as valid_transport_receipt
    for root in (_root(workspace)/'receipts', Path(workspace)/'durability/receipts'):
        for p in (root/row['sha256']).glob('*.json'):
            try: receipt=sub._read(p)
            except (ValueError,OSError): continue
            if receipt.get('schema_id')==TRANSPORT_RECEIPT:
                bound=(receipt.get('obligation_id')==row.get('obligation_id')
                    and receipt.get('role')==row.get('role')
                    and receipt.get('logical_name')==row.get('logical_name')
                    and receipt.get('obligation_scope')=='CHECKPOINT_OUTBOX')
                legacy_unbound='obligation_id' not in receipt
                if valid_transport_receipt(receipt,row) and (bound or not ambiguous and legacy_unbound):return receipt
            if receipt.get('schema_id')=='IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2':
                body={k:v for k,v in receipt.items() if k!='receipt_sha256'}
                if (receipt.get('obligation_id')==row.get('obligation_id')
                    and receipt.get('role')==row.get('role')
                    and receipt.get('obligation_scope')=='CHECKPOINT_OUTBOX'
                    and receipt.get('sha256')==row['sha256'] and receipt.get('size_bytes')==row['size_bytes']
                    and canonical_sha256(body)==receipt.get('receipt_sha256')): return receipt
            if not ambiguous and sub._valid_receipt(p, row): return receipt
    return None

def _physical_receipt(path, row):
    """Verify physical durability independent of the logical obligation."""
    try: receipt=sub._read(path)
    except (ValueError,OSError): return False
    body={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    return (receipt.get('schema_id') in {'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V1','IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2'}
        and receipt.get('provider')=='google_drive' and receipt.get('raw_readback_verified') is True
        and receipt.get('sha256')==row['sha256'] and receipt.get('size_bytes')==row['size_bytes']
        and canonical_sha256(body)==receipt.get('receipt_sha256'))


def _checkpoint_obligations(checkpoint_sha,row,raw):
    physical={o['sha256']:o for o in row['objects']}
    specs=[]
    for obj in row.get('base_objects',[]): specs.append((obj.get('role','base_object'),obj.get('object_name',obj['sha256']),physical[obj['sha256']]))
    for role,obj in [('checkpoint_source',row['source']),('checkpoint_project_base',row['project']['base']),
                     ('checkpoint_project_delta',row['project']['delta']),('checkpoint_state',row['state'])]:
        specs.append((role,obj['sha256'],physical[obj['sha256']]))
    specs.append(('checkpoint_manifest',checkpoint_sha,{'sha256':checkpoint_sha,'size_bytes':len(raw)}))
    out=[]
    for role,logical,obj in specs:
        item=dict(obj,role=role,logical_name=logical,obligation_scope='CHECKPOINT_OUTBOX')
        item['obligation_id']=canonical_sha256({'capture_id':row['capture_id'],'checkpoint_sha256':checkpoint_sha,
            'scope':'CHECKPOINT_OUTBOX','role':role,'logical_name':logical,'sha256':item['sha256'],'size_bytes':item['size_bytes']})
        out.append(item)
    return out


def _commits(workspace):
    rows = {}
    for f in (_root(workspace)/'commits').glob('*.json'):
        raw=f.read_bytes();row=sub._read(f)
        if sub._sha(raw)!=f.stem or row.get('schema_id')!=SCHEMA:
            raise sub.SubmissionError('CHECKPOINT_MANIFEST_MISMATCH', f.name)
        rows[f.stem]=(row,raw)
    for row,_ in rows.values():
        if row['previous'] is not None and row['previous'] not in rows:
            raise sub.SubmissionError('CHECKPOINT_PARENT_MISSING', row['previous'])
    return rows


def status(workspace):
    workspace=Path(workspace).resolve();rows=_commits(workspace);objects={};pending_commits=[]
    for sha,(row,raw) in rows.items():
        required=_checkpoint_obligations(sha,row,raw); counts={}
        for obj in required: counts[obj['sha256']]=counts.get(obj['sha256'],0)+1
        unsaved=False
        for obj in required:
            _bytes(workspace,obj)
            if not _receipt(workspace,obj,ambiguous=counts[obj['sha256']]>1):
                objects.setdefault(obj['obligation_id'],dict(obj,local_object_path=str(_root(workspace)/'objects'/(obj['sha256']+'.bin'))))
                unsaved=True
        if unsaved: pending_commits.append({'checkpoint_sha256':sha,'created_unix':row['created_unix'],'reason':row['reason']})
    for obj in objects.values():
        note=_root(workspace)/'uploads'/(obj['sha256']+'.json')
        if note.is_file():obj['drive_file_id']=sub._read(note)['drive_file_id']
    oldest=min((x['created_unix'] for x in pending_commits),default=None)
    current=_root(workspace)/'CURRENT.json'
    pending=list(objects.values())
    # pending_bytes is a physical transport backlog, not a count of new logical
    # obligations. A hash already covered by a valid raw-readback receipt in
    # this capture needs a fresh checkpoint obligation receipt, but its bytes
    # are already durable and must not consume the reserve again.
    physically_saved=set()
    for x in pending:
        root=Path(workspace)/'durability/receipts'/x['sha256']
        if any(_physical_receipt(p,x) for p in root.glob('*.json')):
            physically_saved.add(x['sha256'])
    physical_pending={x['sha256']:x['size_bytes'] for x in pending if x['sha256'] not in physically_saved}
    return {'status':'SAVE_REQUIRED' if objects else 'PRESERVED' if rows else 'NO_CHECKPOINT',
            'pending_objects':pending,'pending_bytes':sum(physical_pending.values()),
            'pending_checkpoints':pending_commits,'oldest_pending_age_seconds':0 if oldest is None else max(0,time.time()-oldest),
            'latest_checkpoint':sub._read(current)['sha256'] if current.exists() else None,
            'transport':'EXPLICIT_CONNECTOR; NO_UNATTENDED_TRANSPORT_CONFIGURED'}


def note_upload(workspace, digest, drive_id):
    from .v05_controller_event_loop import _workspace_lock
    import re
    with _workspace_lock(_root(workspace)):
        if not re.fullmatch(r'[A-Za-z0-9_-]{10,200}',drive_id):raise sub.SubmissionError('DRIVE_FILE_ID_REQUIRED')
        if not any(x['sha256']==digest for x in status(workspace)['pending_objects']):raise sub.SubmissionError('OUTBOX_OBJECT_NOT_PENDING')
        path=_root(workspace)/'uploads'/(digest+'.json')
        if path.exists() and sub._read(path)['drive_file_id']!=drive_id:raise sub.SubmissionError('UPLOAD_ALREADY_RECORDED')
        write_json_atomic(path,{'sha256':digest,'drive_file_id':drive_id,'status':'UPLOADED_READBACK_REQUIRED'})
    return status(workspace)


def confirm(workspace,digest,readback,drive_id,*,role=None,logical_name=None,obligation_id=None):
    from .v05_controller_event_loop import _workspace_lock
    import re
    with _workspace_lock(_root(workspace)):
        matches=[x for x in status(workspace)['pending_objects'] if x['sha256']==digest]
        if obligation_id is not None: matches=[x for x in matches if x['obligation_id']==obligation_id]
        if role is not None: matches=[x for x in matches if x['role']==role]
        if logical_name is not None: matches=[x for x in matches if x['logical_name']==logical_name]
        if not matches:
            obj=_root(workspace)/'objects'/(digest+'.bin')
            if not obj.is_file():raise sub.SubmissionError('OUTBOX_OBJECT_NOT_REQUIRED')
            raise sub.SubmissionError('OUTBOX_OBLIGATION_NOT_PENDING')
        if len(matches)!=1: raise sub.SubmissionError('OUTBOX_OBLIGATION_AMBIGUOUS',digest,'Supply obligation_id; role and logical_name may also be used.')
        row=matches[0]
        raw=Path(readback).read_bytes()
        if sub._sha(raw)!=digest or len(raw)!=row['size_bytes']:raise sub.SubmissionError('SAVE_READBACK_MISMATCH',digest)
        if not re.fullmatch(r'[A-Za-z0-9_-]{10,200}',drive_id):raise sub.SubmissionError('DRIVE_FILE_ID_REQUIRED')
        note=_root(workspace)/'uploads'/(digest+'.json')
        if note.exists() and sub._read(note)['drive_file_id']!=drive_id:raise sub.SubmissionError('SAVE_UPLOAD_ID_MISMATCH')
        receipt={'schema_id':'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2','provider':'google_drive','sha256':digest,
                 'size_bytes':len(raw),'drive_file_id':drive_id,'raw_readback_verified':True,'recorded_ns':time.time_ns(),
                 'obligation_id':row['obligation_id'],'obligation_scope':'CHECKPOINT_OUTBOX',
                 'role':row['role'],'logical_name':row['logical_name'],
                 'scope':'EXPLICIT_CONNECTOR_OPERATOR_ATTESTATION_NOT_REMOTE_AUTHENTICATION'}
        receipt['receipt_sha256']=canonical_sha256(receipt)
        write_json_atomic(_root(workspace)/'receipts'/digest/(receipt['receipt_sha256']+'.json'),receipt)
    return status(workspace)


def _base_raw(workspace,obj):
    workspace=Path(workspace)
    candidates=[workspace/'durability/base_objects'/obj['object_name'],workspace.parent.parent/'objects'/obj['object_name'],
                workspace/'runtime/intake/artifacts'/(obj['sha256']+'.bin'),_root(workspace)/'objects'/(obj['sha256']+'.bin')]
    if obj['role']=='capture_record':candidates.insert(0,workspace/'CAPTURE.json')
    if obj['role']=='project_binding':candidates.insert(0,workspace/'PROJECT_BINDING.json')
    for p in candidates:
        if p.is_file():
            raw=p.read_bytes()
            if sub._sha(raw)==obj['sha256'] and len(raw)==obj['size_bytes']:return raw
    if obj['role'] in ('engine_source','project_source'):
        tree=sub._tree(workspace/'source')
        files={n:b for n,b in tree.items() if not n.startswith('project/')} if obj['role']=='engine_source' else {n[8:]:b for n,b in tree.items() if n.startswith('project/')}
        raw=sub._archive(files)
        if sub._sha(raw)==obj['sha256']:return raw
    raise sub.SubmissionError('RECOVERY_DEPENDENCY_MISSING',obj['sha256'],
        'Fetch this exact saved object into durability/base_objects using its original object_name; retry native snapshot/export.')


def confirm_transport(workspace,digest,manifest,parts,drive_id,*,role=None,logical_name=None,obligation_id=None):
    from .v05_controller_event_loop import _workspace_lock
    from .save_transport import make_receipt
    with _workspace_lock(_root(workspace)):
        try:
            matches=[x for x in status(workspace)['pending_objects'] if x['sha256']==digest]
            if obligation_id is not None:matches=[x for x in matches if x['obligation_id']==obligation_id]
            if role is not None:matches=[x for x in matches if x['role']==role]
            if logical_name is not None:matches=[x for x in matches if x['logical_name']==logical_name]
            if not matches:
                obj=_root(workspace)/'objects'/(digest+'.bin')
                if not obj.is_file():raise sub.SubmissionError('OUTBOX_OBJECT_NOT_REQUIRED')
                raise sub.SubmissionError('OUTBOX_OBLIGATION_NOT_PENDING')
            if len(matches)!=1:
                raise sub.SubmissionError('OUTBOX_OBLIGATION_AMBIGUOUS',digest,
                    'Supply obligation_id; role and logical_name may also be used.')
            row=matches[0]
            receipt=make_receipt(row,manifest,parts,drive_id)
            note=_root(workspace)/'uploads'/(digest+'.json')
            original=receipt['transport']['manifest'].get('original_object_drive_file_id')
            if original is not None and note.exists() and sub._read(note)['drive_file_id']!=original:
                raise sub.SubmissionError('SAVE_UPLOAD_ID_MISMATCH')
            write_json_atomic(_root(workspace)/'receipts'/digest/(receipt['receipt_sha256']+'.json'),receipt)
        except Exception as exc:
            sub._failure(_root(workspace),'CONFIRM_TRANSPORT',exc)
            raise
    return status(workspace)


def _stable_file(path, *, quiescent=False):
    # Task stores use SQLite backup, never raw-copy a live WAL database.
    with path.open('rb') as f:prefix=f.read(16)
    if prefix==b'SQLite format 3\x00' and not quiescent:
        with tempfile.TemporaryDirectory(prefix='decoder-db-checkpoint-') as temp:
            dest=Path(temp)/'state.sqlite'
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as src, closing(sqlite3.connect(dest)) as dst:
                src.backup(dst)
                if dst.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise sub.SubmissionError('CHECKPOINT_DATABASE_INVALID')
                dst.commit()
            return dest.read_bytes()
    before=path.stat();raw=path.read_bytes();after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        if path.suffix=='.log' and after.st_size>=before.st_size:return raw[:before.st_size]
        raise sub.SubmissionError('CHECKPOINT_FILE_CHANGED',str(path))
    return raw


def require_quiescent_task_databases(output_root):
    """Do not bind transient engine-owned SQLite journals into a completion.

    This is a read-only gate, not forced checkpointing or journal deletion.
    Ordinary SCRIPT artifacts are outside this engine-owned task-store scope.
    """
    output_root=Path(output_root)
    runtime_root=output_root/'chain/decoder_stage_runtime'
    for path in sorted(runtime_root.rglob('*.sqlite3')):
        if path.name not in {'partition.sqlite3','state_store.sqlite3'}:
            continue
        for suffix in ('-wal','-shm','-journal'):
            if Path(str(path)+suffix).exists():
                raise sub.SubmissionError('TASK_DATABASE_NOT_QUIESCENT',
                    str(path.relative_to(output_root)),
                    'Close external database readers, then resume this captured request. Do not remove journals.')


def _verify_terminal_state_evidence(files):
    """The exact exported state must retain every byte bound by a completion.

    Native completion authentication remains with verified_completion. This
    additional check stops snapshot filtering from producing an inconsistent
    archive even when the live workspace previously matched its completion.
    """
    runs={}
    for name,raw in sorted(files.items()):
        parts=name.split('/')
        if len(parts)>=4 and parts[:2] in (['runtime','runs'], ['runtime','sealed']):
            runs.setdefault('/'.join(parts[:3]),[]).append({
                'path':'/'.join(parts[3:]),'sha256':sub._sha(raw),'size_bytes':len(raw)})
    completion_bytes={}
    for name,raw in files.items():
        if name.startswith(('runtime/intake/completed/','runtime/intake/prepared_completions/')) and name.endswith('.json'):
            rid=Path(name).stem
            if rid in completion_bytes and completion_bytes[rid]!=raw:
                raise sub.SubmissionError('CHECKPOINT_PREPARED_COMPLETION_MISMATCH',name)
            completion_bytes[rid]=raw
            done=json.loads(raw); evidence=done.get('evidence')
            if 'evidence_protocol' in done or 'evidence_root' in done:
                from .completion_evidence import PROTOCOL
                location=done.get('evidence_root', '')
                valid=(done.get('evidence_protocol')==PROTOCOL
                       and location.startswith('runtime/sealed/intent-')
                       and len(location.split('/'))==3
                       and runs.get(location)==evidence)
            else:
                valid=sum(rows==evidence for name,rows in runs.items()
                          if name.startswith('runtime/runs/'))==1
            if not evidence or not valid:
                from .evidence_diagnostics import attach
                # Legacy records do not identify a root: retain every candidate,
                # never invent which directory the producer intended.
                observed = ({done.get('evidence_root', ''): runs.get(done.get('evidence_root', ''), [])}
                    if 'evidence_protocol' in done or 'evidence_root' in done else
                    {key: rows for key, rows in runs.items() if key.startswith('runtime/runs/')})
                raise attach(sub.SubmissionError('CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH',name,
                    'Preserve the original completion and failed bytes. Do not rewrite completion evidence to fit an archive.'),
                    phase='CHECKPOINT_STATE_INVENTORY', done=done, completion_path=name,
                    observed_roots=observed, failed_checks=['terminal_evidence_binding'],
                    observation='EXACT_IN_MEMORY_CHECKPOINT_BYTES')
    # The completion lists immutable evidence files, but replay also publishes
    # two mutable indexes. Require those indexes to agree with the published
    # terminal status before a checkpoint can carry them into another chat.
    for name, raw in files.items():
        if not name.endswith('/artifacts/REPLAY_ROOT_STATUS.json'):
            continue
        stage = name.removesuffix('/artifacts/REPLAY_ROOT_STATUS.json')
        status = json.loads(raw)
        for suffix, field, self_field in (
            ('/replay_runner/runner_state.json', 'runner_state_sha256', 'state_sha256'),
            ('/replay_reference_data/MANIFEST.json', 'reference_manifest_sha256', 'manifest_sha256'),
        ):
            current = files.get(stage + suffix)
            if current is None or json.loads(current).get(self_field) != status.get(field):
                raise sub.SubmissionError('CHECKPOINT_REPLAY_STATE_STATUS_MISMATCH', stage + suffix,
                    'Preserve the original bytes and investigate the mutable replay state.')


def _state_files(workspace):
    from .v05_controller_event_loop import _snapshot_files
    files={}
    quiescent=any(any((Path(workspace)/'runtime/intake'/name).glob('*.json'))
                  for name in ('completed','prepared_completions'))
    legacy = any('evidence_protocol' not in sub._read(p)
                 for folder in ('completed','prepared_completions')
                 for p in (Path(workspace)/'runtime/intake'/folder).glob('*.json'))
    for p,name in _snapshot_files(workspace):
        if (name.startswith(('source/','coordination/','runtime/intake/artifacts/','durability/outbox/','durability/base_objects/'))
            or name=='PROJECT_LOCATION.json' or (name.endswith(('-wal','-shm')) and not name.startswith('runtime/sealed/'))):continue
        files[name]=_stable_file(p,quiescent=legacy or name.startswith('runtime/sealed/'))
    if quiescent:_verify_terminal_state_evidence(files)
    return files


def make_checkpoint(workspace,reason,*,terminal=False):
    """Caller owns the workspace; invoked only at controller safe boundaries."""
    from .portable_registry import locate,snapshot
    from .v05_controller_event_loop import _workspace_lock,_source_ids
    workspace=Path(workspace).resolve();rec=sub.capture_record(workspace)
    if _source_ids(workspace/'source')!=(rec['workspace']['source_sha256'],rec['workspace']['package_sha256']):raise sub.SubmissionError('CHECKPOINT_SOURCE_MISMATCH')
    with _workspace_lock(_root(workspace)):
        objects={};base=[];baseline=None;baseline_raw=None
        for obj in sub.required_objects(workspace):
            row=_put(workspace,_base_raw(workspace,obj));objects[row['sha256']]=row;base.append(dict(obj))
            if obj['role']=='project_baseline':baseline=row;baseline_raw=_bytes(workspace,row)
        source=_put(workspace,sub._archive(sub._tree(workspace/'source')));objects[source['sha256']]=source
        project_root,_=locate(workspace)
        if baseline is None:raise sub.SubmissionError('PROJECT_BASELINE_REQUIRED')
        full_project=snapshot(project_root)
        delta=_put(workspace,_project_delta(baseline_raw,full_project));objects[delta['sha256']]=delta
        project={'schema_id':'IG_PROJECT_DELTA_V1','base':baseline,'delta':delta,'full_sha256':sub._sha(full_project)}
        state_files=_state_files(workspace)
        terminal_completions={Path(name).stem:json.loads(raw)['completion_sha256']
            for name,raw in state_files.items()
            if name.startswith(('runtime/intake/completed/','runtime/intake/prepared_completions/'))
            and name.endswith('.json')} if terminal else {}
        state=_put(workspace,sub._archive(state_files));objects[state['sha256']]=state
        pointer_file=_root(workspace)/'CURRENT.json'
        previous=sub._read(pointer_file)['sha256'] if pointer_file.exists() else None
        if previous:
            old=sub._read(_root(workspace)/'commits'/(previous+'.json'))
            if (old['state']==state and old['project']==project and old['source']==source
                    and (not terminal or old.get('terminal') and old.get('terminal_completions')==terminal_completions)):
                return status(workspace)
        contract=rec.get('result_contract',{})
        from .result_contracts import policy
        limits=policy(contract)
        payload_size=sum(x['size_bytes'] for x in (source,state,delta))
        if payload_size>limits['max_commit_bytes']:raise sub.SubmissionError('CHECKPOINT_SIZE_LIMIT',str(payload_size))
        row={'schema_id':SCHEMA,'capture_id':rec['capture_id'],'job_id':rec['job']['job_id'],
             'source_sha256':rec['workspace']['source_sha256'],'created_unix':time.time(),'previous':previous,
             'reason':reason,'terminal':terminal,'source':source,'project':project,'state':state,'base_objects':base,
             'objects':sorted(objects.values(),key=lambda x:x['sha256'])}
        if terminal:row['terminal_completions']=terminal_completions
        raw=sub._json_bytes(row);commit=_put(workspace,raw)
        commit_file=_root(workspace)/'commits'/(commit['sha256']+'.json')
        commit_file.parent.mkdir(parents=True,exist_ok=True)
        if not commit_file.exists():
            # Use the exact bytes whose raw SHA256 is the checkpoint identity.
            os.link(_root(workspace)/'objects'/(commit['sha256']+'.bin'),commit_file)
        write_json_atomic(pointer_file,{'sha256':commit['sha256']})
    return status(workspace)


COMPLETION_PROTOCOL = 'CHECKPOINT_BEFORE_COMPLETION_V1'


def terminal_completion_proof(workspace, done):
    """Check portable checkpoint identity/binding; absence means unpublished."""
    if done.get('publication_protocol') is None:
        return True  # Historical producers retain their original evidence rules.
    if done.get('publication_protocol') != COMPLETION_PROTOCOL:
        raise sub.SubmissionError('UNKNOWN_COMPLETION_PUBLICATION_PROTOCOL')
    workspace=Path(workspace)
    path=workspace/'runtime/terminal_checkpoints'/(done['request_id']+'.json')
    restored=workspace/'durability/RESTORED_CHECKPOINT_PROVENANCE.json'
    if path.exists():packet=sub._read(path)
    elif restored.exists():
        packet=sub._read(restored)
        if packet.get('checkpoint',{}).get('terminal') is not True:return False
    else:return False
    row=packet.get('checkpoint',{})
    rec=sub.capture_record(workspace)
    if (sub._sha(sub._json_bytes(row))!=packet.get('checkpoint_sha256')
            or row.get('schema_id')!=SCHEMA or row.get('terminal') is not True
            or row.get('capture_id')!=rec['capture_id']
            or row.get('source_sha256')!=done['source_sha256']
            or row.get('job_id')!=rec['job']['job_id']
            or row.get('terminal_completions',{}).get(done['request_id'])!=done['completion_sha256']):
        raise sub.SubmissionError('TERMINAL_COMPLETION_CHECKPOINT_MISMATCH')
    return True


def ensure_terminal_completion(workspace, done):
    """Checkpoint exact finished evidence before publishing a reusable claim."""
    workspace=Path(workspace)
    if terminal_completion_proof(workspace,done):
        return status(workspace)
    result=make_checkpoint(workspace,'TERMINAL_RESULT',terminal=True)
    digest=result['latest_checkpoint']
    row=sub._read(_root(workspace)/'commits'/(digest+'.json'))
    files=_zip_files(_bytes(workspace,row['state']))
    name='runtime/intake/prepared_completions/'+done['request_id']+'.json'
    if name not in files:name='runtime/intake/completed/'+done['request_id']+'.json'
    # Compare the existing canonical JSON representation, not Python container
    # identity: persisted arrays are lists even when the producer used tuples.
    # Byte equality also keeps bool/int and int/float distinctions that Python
    # equality would erase. No saved bytes or completion hashes are rewritten.
    if (canonical_bytes(json.loads(files.get(name,b'null')))!=canonical_bytes(done)
            or row.get('terminal_completions',{}).get(done['request_id'])!=done['completion_sha256']):
        raise sub.SubmissionError('TERMINAL_COMPLETION_CHECKPOINT_MISMATCH')
    _verify_terminal_state_evidence(files)
    path=workspace/'runtime/terminal_checkpoints'/(done['request_id']+'.json')
    packet={'checkpoint_sha256':digest,'checkpoint':row}
    if path.exists() and sub._read(path)!=packet:
        raise sub.SubmissionError('TERMINAL_COMPLETION_PROOF_COLLISION')
    write_json_atomic(path,packet)
    terminal_completion_proof(workspace,done)
    return result


def backlog(workspace,*,reserve=False):
    from .result_contracts import contract_for,policy
    limits=policy(contract_for(workspace));s=status(workspace)
    over=(s['pending_bytes']+(limits['max_commit_bytes'] if reserve else 0)>limits['max_pending_bytes']
          or len(s['pending_checkpoints'])>=limits['max_pending_commits']
          or s['oldest_pending_age_seconds']>limits['max_pending_age_seconds'])
    if over:raise sub.SubmissionError('SAVE_BACKLOG_PAUSE',str(s['pending_bytes']),
         'Use preserve status, upload or read back its exact pending objects, and preserve confirm; then resume the original capture.')
    return s


@contextmanager
def session(admission):
    token=_ACTIVE.set({'workspace':admission['workspace'],'admission':admission,'pid':os.getpid(),'last':time.monotonic(),'last_poll':0})
    try:yield
    finally:_ACTIVE.reset(token)


def safe_point(reason,*,force=False):
    active=_ACTIVE.get()
    if not active or active['pid']!=os.getpid():return
    from .v05_origin_guard import require_controller_execution_origin
    require_controller_execution_origin('preservation-safe-point')
    from .result_contracts import contract_for,policy
    limits=policy(contract_for(active['workspace']))
    backlog(active['workspace'],reserve=True)
    if force or time.monotonic()-active['last']>=limits['interval_seconds']:
        make_checkpoint(active['workspace'],reason)
        active['last']=time.monotonic()
        backlog(active['workspace'],reserve=True)


def check_budget(admission,*,extra_bytes=0):
    root=Path(admission['workspace']);budget=admission['job']['resources']['workspace_budget_bytes']
    total=sum(p.stat().st_size for base in (root/'runtime/runs',root/'runtime/sealed',_root(root)) for p in base.rglob('*') if p.is_file())+extra_bytes
    if total>budget:raise sub.SubmissionError('REGISTERED_WORKSPACE_BUDGET',str(total))


def poll(*,extra_bytes=0):
    active=_ACTIVE.get()
    if not active or active['pid']!=os.getpid():return
    now=time.monotonic()
    if not extra_bytes and now-active['last_poll']<0.25:return
    active['last_poll']=now
    request=Path(active['workspace'])/'runtime/PAUSE_REQUEST.json'
    ack=Path(active['workspace'])/'runtime/PAUSE_ACK.json'
    if request.exists():
        pending=sub._read(request)
        if not ack.exists() or sub._read(ack)['nonce']!=pending['nonce']:
            if pending['capture_id']!=sub.capture_record(active['workspace'])['capture_id']:raise sub.SubmissionError('PAUSE_CAPTURE_MISMATCH')
            write_json_atomic(ack,pending)
            raise sub.SubmissionError('REQUESTED_PAUSE',pending['reason'],'Drain pending saves and resume the same captured job.')
    check_budget(active['admission'],extra_bytes=extra_bytes)
    # Account proportional resident pages where Linux exposes them. A missing
    # measurement does not become a zero-memory guarantee.
    parents={};procs=Path('/proc')
    if procs.exists():
        for f in procs.glob('[0-9]*/stat'):
            try:parents[int(f.parent.name)]=int(f.read_text().rsplit(')',1)[1].split()[1])
            except (OSError,ValueError,IndexError):continue
        ids={os.getpid()}
        while True:
            more={pid for pid,parent in parents.items() if parent in ids}
            if more<=ids:break
            ids|=more
        measured=0
        for pid in ids:
            try:
                lines=(procs/str(pid)/'smaps_rollup').read_text().splitlines()
                measured+=next(int(x.split()[1])*1024 for x in lines if x.startswith('Pss:'))
            except (OSError,StopIteration,ValueError):
                try:measured+=int((procs/str(pid)/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                except (OSError,ValueError,IndexError):continue
        if measured>active['admission']['job']['resources']['memory_budget_bytes']:
            raise sub.SubmissionError('REGISTERED_MEMORY_BUDGET',str(measured))


def snapshot_workspace(workspace,reason='EXPLICIT_IDLE_SNAPSHOT'):
    from .v05_controller_event_loop import _workspace_lock,_workspace_ids
    root=Path(workspace).resolve()
    with _workspace_lock(root):
        _workspace_ids(root)
        return make_checkpoint(root,reason)


def export_checkpoint(workspace,output,*,slim=False):
    s=status(workspace);sha=s['latest_checkpoint']
    if not sha:raise sub.SubmissionError('CHECKPOINT_REQUIRED',next_action='Use preserve snapshot WORKSPACE before exporting idle state.')
    row,raw=_commits(workspace)[sha];target=Path(output).resolve()
    if target.is_relative_to(Path(workspace).resolve()):raise sub.SubmissionError('EXPORT_MUST_BE_OUTSIDE_WORKSPACE')
    prior=[{'sha256':key,'previous':r['previous'],'created_unix':r['created_unix'],
            'receipt':_receipt(workspace,{'sha256':key,'size_bytes':len(b)})} for key,(r,b) in _commits(workspace).items() if key!=sha]
    packet={'schema_id':'IG_DECODER_CHECKPOINT_EXPORT_V1','checkpoint_sha256':sha,'checkpoint':row,'mode':'SLIM' if slim else 'COMPLETE',
            'history_scope':'Current restart closure; earlier checkpoint references retained separately.',
            'prior_checkpoints':prior,'dependencies':[dict(o,receipts=[sub._read(p) for p in
                sorted((_root(workspace)/'receipts'/o['sha256']).glob('*.json'))]) for o in row['objects']]}
    files={'CHECKPOINT.json':sub._json_bytes(packet)}
    if not slim:
        files.update({'objects/'+o['sha256']:_bytes(workspace,o) for o in row['objects']})
    target.parent.mkdir(parents=True,exist_ok=True);raw_archive=sub._archive(files);target.write_bytes(raw_archive)
    return {'status':'EXPORTED_LOCAL','path':str(target),'sha256':sub._sha(raw_archive),'mode':packet['mode'],
            'dependencies':packet['dependencies'],'drive_save_confirmed':False}


def restore_checkpoint(archive,destination,expected_sha,objects=None):
    raw=Path(archive).read_bytes()
    if sub._sha(raw)!=expected_sha:raise sub.SubmissionError('RESTORE_ARCHIVE_HASH')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise sub.SubmissionError('RESTORE_DUPLICATE_PATH')
        packet=json.loads(z.read('CHECKPOINT.json'));row=packet['checkpoint'];raw_row=sub._json_bytes(row)
        if packet.get('schema_id')!='IG_DECODER_CHECKPOINT_EXPORT_V1' or row.get('schema_id')!=SCHEMA or sub._sha(raw_row)!=packet['checkpoint_sha256']:
            raise sub.SubmissionError('CHECKPOINT_MANIFEST_MISMATCH')
        provided={};missing=[]
        for obj in row['objects']:
            name='objects/'+obj['sha256'];content=None
            if name in names:content=z.read(name)
            elif objects:
                for suffix in ('','.bin','.readback'):
                    f=Path(objects)/(obj['sha256']+suffix)
                    if f.is_file():content=f.read_bytes();break
            if content is None:missing.append(obj);continue
            if sub._sha(content)!=obj['sha256'] or len(content)!=obj['size_bytes']:raise sub.SubmissionError('RESTORE_DEPENDENCY_MISMATCH',obj['sha256'])
            provided[obj['sha256']]=content
        if missing:raise sub.SubmissionError('RECOVERY_DEPENDENCIES_UNRESOLVED',json.dumps(missing),
             'Fetch exact objects using the export dependency receipts, then retry preserve restore with --objects DIRECTORY.')
        if set(names)-({'CHECKPOINT.json'}|{'objects/'+o['sha256'] for o in row['objects']}):raise sub.SubmissionError('RESTORE_FILE_SET')
    dest=Path(destination).resolve()
    if dest.exists():raise sub.SubmissionError('RESTORE_DESTINATION_EXISTS')
    dest.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.decoder-checkpoint-restore-',dir=dest.parent))
    def unpack(data,where):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if len(z.namelist())!=len(set(z.namelist())):raise sub.SubmissionError('RESTORE_DUPLICATE_PATH')
            for name in z.namelist():
                path=where/sub._relative(name);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
    try:
        unpack(provided[row['source']['sha256']],temp/'source')
        unpack(provided[row['state']['sha256']],temp)
        project=row['project']
        project_raw=_restore_project(provided[project['base']['sha256']],provided[project['delta']['sha256']],project['full_sha256'])
        unpack(project_raw,temp/'coordination')
        write_json_atomic(temp/'durability/RESTORED_CHECKPOINT_PROVENANCE.json',packet)
        rec=sub.capture_record(temp)
        if rec['capture_id']!=row['capture_id'] or rec['job']['job_id']!=row['job_id']:raise sub.SubmissionError('CHECKPOINT_CAPTURE_BINDING')
        for obj in row['base_objects']:
            path=temp/'durability/base_objects'/obj['object_name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(provided[obj['sha256']])
        for obj in rec['objects']:
            if obj['role'].startswith('input:'):
                path=temp/'runtime/intake/artifacts'/(obj['sha256']+'.bin');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(provided[obj['sha256']])
        from .v05_controller_event_loop import validate_workspace_job,verified_completion
        admission=validate_workspace_job(temp,row['job_id'],check_loaded=False)
        from .portable_registry import events,blob,verify_capsule
        history=temp/'coordination';es=events(history)
        for f in (history/'objects').glob('*'):blob(history,f.name)
        for e in es.values():
            capsule=e['value'].get('capsule') or e['value'].get('validation')
            if capsule:verify_capsule(history,capsule)
        # A paused snapshot may contain finished but unpublished evidence.
        # Restore it for checkpoint recovery without advertising completion.
        done=verified_completion(admission,allow_pending_checkpoint=True)
        if done is not None:terminal_completion_proof(temp,done)
        os.replace(temp,dest)
    finally:
        if temp.exists():shutil.rmtree(temp)
    return {'status':'RESTORED_NOT_RUNNING','workspace':str(dest),'source_sha256':row['source_sha256'],'checkpoint_sha256':packet['checkpoint_sha256']}


def request_pause(workspace,reason):
    if not reason.strip():raise sub.SubmissionError('PAUSE_REASON_REQUIRED')
    root=Path(workspace).resolve();rec=sub.capture_record(root)
    row={'capture_id':rec['capture_id'],'reason':reason,'nonce':str(time.time_ns())}
    write_json_atomic(root/'runtime/PAUSE_REQUEST.json',row)
    return {'status':'PAUSE_REQUESTED','capture_id':rec['capture_id'],'reason':reason,
            'scope':'Passive request; native controller acknowledges at a monitored safe boundary.'}


def main(argv):
    parser=argparse.ArgumentParser(description='Native checkpoints and explicit save outbox.')
    cmds=parser.add_subparsers(dest='command',required=True)
    for name in ('status','snapshot'):
        p=cmds.add_parser(name);p.add_argument('workspace')
    p=cmds.add_parser('note-upload');p.add_argument('workspace');p.add_argument('sha256');p.add_argument('drive_id')
    p=cmds.add_parser('pause');p.add_argument('workspace');p.add_argument('reason')
    p=cmds.add_parser('confirm');p.add_argument('workspace');p.add_argument('sha256');p.add_argument('readback');p.add_argument('drive_id');p.add_argument('--role');p.add_argument('--logical-name');p.add_argument('--obligation-id')
    p=cmds.add_parser('confirm-transport');p.add_argument('workspace');p.add_argument('sha256');p.add_argument('manifest');p.add_argument('parts');p.add_argument('drive_id');p.add_argument('--role');p.add_argument('--logical-name');p.add_argument('--obligation-id')
    p=cmds.add_parser('export');p.add_argument('workspace');p.add_argument('output');p.add_argument('--slim',action='store_true')
    p=cmds.add_parser('restore');p.add_argument('archive');p.add_argument('destination');p.add_argument('sha256');p.add_argument('--objects')
    args=parser.parse_args(argv)
    try:
        if args.command=='status':result=status(args.workspace)
        elif args.command=='snapshot':result=snapshot_workspace(args.workspace)
        elif args.command=='pause':result=request_pause(args.workspace,args.reason)
        elif args.command=='note-upload':result=note_upload(args.workspace,args.sha256,args.drive_id)
        elif args.command=='confirm':result=confirm(args.workspace,args.sha256,args.readback,args.drive_id,role=args.role,logical_name=args.logical_name,obligation_id=args.obligation_id)
        elif args.command=='confirm-transport':result=confirm_transport(args.workspace,args.sha256,args.manifest,args.parts,args.drive_id,role=args.role,logical_name=args.logical_name,obligation_id=args.obligation_id)
        elif args.command=='export':result=export_checkpoint(args.workspace,args.output,slim=args.slim)
        else:result=restore_checkpoint(args.archive,args.destination,args.sha256,args.objects)
        print(json.dumps(result,sort_keys=True,indent=2));return 0
    except Exception as exc:
        from .invocation import refusal_details
        print(json.dumps(refusal_details(exc,'preserve '+args.command,getattr(args,'workspace',None)),sort_keys=True,indent=2));return 2
