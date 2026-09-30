"""Bounded native acknowledgments for explicit, already downloaded save evidence.

Every supplied object is verified once per invocation; role receipts stay
independent. A durable prepared journal makes interrupted publication resumable.
No connector access, fabricated transport identity, or workload dispatch occurs.
"""
from pathlib import Path
import hashlib
import time

from . import submission as sub
from .canon import canonical_sha256, write_json_atomic
from . import preservation as pr
from .save_transport import _receipt_from_manifest_bytes, read_manifest, _id, _object

SCHEMA = 'IG_CHECKPOINT_ACK_BATCH_V1'
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_OBLIGATIONS = 1024
MAX_OBJECTS = 256
MAX_READBACK_BYTES = 512 * 1024 * 1024
IDENTITY = ('obligation_id','obligation_scope','role','logical_name','sha256','size_bytes')


def _fail(code):
    raise sub.SubmissionError('ACK_BATCH_' + code)


def _load(batch):
    from .storage_schema import strict_loads
    if not isinstance(batch, (str, Path)):
        _fail('FILE_REQUIRED')
    path=Path(batch)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_MANIFEST_BYTES:
        _fail('FILE')
    obj=strict_loads(path.read_bytes(),max_bytes=MAX_MANIFEST_BYTES)
    if (type(obj) is not dict or set(obj)!={'schema_id','capture_id','obligations','readbacks'}
            or obj['schema_id']!=SCHEMA or type(obj['capture_id']) is not str
            or type(obj['obligations']) is not list or not 1<=len(obj['obligations'])<=MAX_OBLIGATIONS
            or type(obj['readbacks']) is not list or not 1<=len(obj['readbacks'])<=MAX_OBJECTS):
        _fail('SCHEMA_OR_LIMIT')
    return obj


def _verify_raw(path, expected):
    path=Path(path)
    if path.is_symlink() or not path.is_file():_fail('READBACK_FILE')
    h=hashlib.sha256();size=0
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            size+=len(chunk);h.update(chunk)
            if size>expected['size_bytes']:_fail('READBACK_MISMATCH')
    if size!=expected['size_bytes'] or h.hexdigest()!=expected['sha256']:
        _fail('READBACK_MISMATCH')


def _publish(path, row):
    # The outbox lock serializes native publishers. Content-addressed filenames
    # never overwrite a different receipt or silently repair altered journals.
    if path.is_symlink():_fail('PUBLICATION_SYMLINK')
    if path.exists():
        if sub._read(path)!=row:_fail('PUBLICATION_CONFLICT')
    else:write_json_atomic(path,row)
    if sub._read(path)!=row:_fail('PUBLICATION_READBACK')


def _body_without_time(row):
    return {k:v for k,v in row.items() if k not in {'recorded_ns','receipt_sha256'}}


def confirm_batch(workspace,batch):
    from .v05_controller_event_loop import _workspace_lock
    request=_load(batch);batch_id=canonical_sha256(request)
    workspace=Path(workspace).resolve(strict=True)
    with _workspace_lock(pr._root(workspace)):
        capture=sub.capture_record(workspace)
        if capture['capture_id']!=request['capture_id']:_fail('CAPTURE')
        # Bound history before reading any large object. No old global cache.
        if len(pr._commits(workspace))>4096:_fail('CHECKPOINT_LIMIT')
        rows,by_checkpoint,physical=pr._inventory(workspace)
        if any(row['capture_id']!=capture['capture_id'] for row,raw in rows.values()):
            _fail('CHECKPOINT_CAPTURE')
        required={}
        for obligations in by_checkpoint.values():
            for row in obligations:
                ident={k:row[k] for k in IDENTITY}
                old=required.get(row['obligation_id'])
                if old is not None and old!=ident:_fail('OBLIGATION_CONFLICT')
                required[row['obligation_id']]=ident
        selected={}
        for row in request['obligations']:
            if type(row) is not dict or set(row)!=set(IDENTITY):_fail('OBLIGATION_FIELDS')
            if not _object({k:row[k] for k in ('sha256','size_bytes')}):_fail('OBLIGATION_FIELDS')
            oid=row['obligation_id']
            if type(oid) is not str or oid in selected:_fail('DUPLICATE_OBLIGATION')
            if required.get(oid)!=row:_fail('OBLIGATION_BINDING')
            selected[oid]=row
        needed={row['sha256']:physical[row['sha256']] for row in selected.values()}
        if len(needed)>MAX_OBJECTS or sum(r['size_bytes'] for r in needed.values())>MAX_READBACK_BYTES:
            _fail('READBACK_LIMIT')
        references={}
        for ref in request['readbacks']:
            if type(ref) is not dict:_fail('READBACK_FIELDS')
            fields={'sha256','kind','drive_file_id'}
            if ref.get('kind')=='RAW':fields|={'path'}
            elif ref.get('kind')=='MULTIPART':fields|={'manifest','parts'}
            else:_fail('READBACK_KIND')
            if set(ref)!=fields or any(type(v) is not str for v in ref.values()):_fail('READBACK_FIELDS')
            if ref['sha256'] in references or ref['sha256'] not in needed:_fail('READBACK_BINDING')
            if not _id(ref['drive_file_id']):_fail('DRIVE_ID')
            references[ref['sha256']]=ref
        if set(references)!=set(needed):_fail('MISSING_READBACK')
        transport_bytes=0;manifest_bytes={}
        for digest,ref in references.items():
            if ref['kind']=='RAW':transport_bytes+=needed[digest]['size_bytes']
            else:
                path=Path(ref['manifest'])
                if path.is_symlink() or not path.is_file():_fail('MANIFEST_FILE')
                raw_manifest,manifest=read_manifest(path)
                manifest_bytes[digest]=raw_manifest
                if manifest['object']!={k:needed[digest][k] for k in ('sha256','size_bytes')}:
                    _fail('TRANSPORT_BINDING')
                transport_bytes+=len(raw_manifest)+sum(x['size_bytes'] for x in manifest['parts'])
        if transport_bytes>MAX_READBACK_BYTES:_fail('TRANSPORT_LIMIT')
        # Verify only the batch payloads once; unrelated pending payloads remain
        # unverified here. Full preserve status still verifies the complete set.
        for row in needed.values():pr._bytes(workspace,row)
        # Preflight every readback BEFORE publishing any journal/role receipt.
        proofs={}
        for digest,ref in references.items():
            obj={k:needed[digest][k] for k in ('sha256','size_bytes')}
            note=pr._root(workspace)/'uploads'/(digest+'.json')
            if ref['kind']=='RAW':
                _verify_raw(ref['path'],obj)
                original=ref['drive_file_id']
                proof={'schema_id':'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2','provider':'google_drive',
                       **obj,'drive_file_id':original,'raw_readback_verified':True,
                       'recorded_ns':time.time_ns(),
                       'scope':'EXPLICIT_CONNECTOR_OPERATOR_ATTESTATION_NOT_REMOTE_AUTHENTICATION'}
            else:
                proof=_receipt_from_manifest_bytes(obj,manifest_bytes[digest],ref['parts'],ref['drive_file_id'])
                original=proof['transport']['manifest'].get('original_object_drive_file_id')
            if note.exists() and (original is None or sub._read(note)['drive_file_id']!=original):
                _fail('UPLOAD_ID_MISMATCH')
            proofs[digest]=proof
        prepared=[]
        for oid,row in sorted(selected.items()):
            receipt={k:v for k,v in proofs[row['sha256']].items() if k!='receipt_sha256'}
            receipt.update(row,batch_id=batch_id)
            receipt['receipt_sha256']=canonical_sha256(receipt)
            prepared.append(receipt)
        directory=pr._root(workspace)/'batches'/batch_id
        plan_path=directory/'PREPARED.json'
        plan={'schema_id':'IG_CHECKPOINT_ACK_PREPARED_V1','batch_id':batch_id,
              'capture_id':capture['capture_id'],'request':request,'receipts':prepared}
        if plan_path.exists():
            old=sub._read(plan_path)
            if (set(old)!=set(plan) or any(old.get(k)!=plan[k] for k in ('schema_id','batch_id','capture_id','request'))
                    or not isinstance(old.get('receipts'),list) or len(old['receipts'])!=len(prepared)):
                _fail('JOURNAL_BINDING')
            for saved,fresh in zip(old['receipts'],prepared):
                if (type(saved) is not dict or _body_without_time(saved)!=_body_without_time(fresh)
                        or type(saved.get('recorded_ns')) is not int
                        or canonical_sha256({k:v for k,v in saved.items() if k!='receipt_sha256'})!=saved.get('receipt_sha256')):
                    _fail('JOURNAL_RECEIPT')
            plan=old
        _publish(plan_path,plan)
        # Crash here or mid-loop leaves a verified, resumable prepared batch.
        for receipt in plan['receipts']:
            target=pr._root(workspace)/'receipts'/receipt['sha256']/(receipt['receipt_sha256']+'.json')
            _publish(target,receipt)
        outcome=pr._status_from_inventory(workspace,rows,by_checkpoint,physical,verified_count=len(needed))
        if set(selected)&{r['obligation_id'] for r in outcome['pending_objects']}:
            _fail('ACKNOWLEDGMENT_NOT_RECOGNIZED')
        complete={'schema_id':'IG_CHECKPOINT_ACK_COMPLETE_V1','batch_id':batch_id,
                  'capture_id':capture['capture_id'],'prepared_sha256':canonical_sha256(plan),
                  'obligations_acknowledged':len(selected),'readbacks_verified':len(proofs)}
        _publish(directory/'COMPLETE.json',complete)
        return {**complete,'outbox':outcome,'verification_scope':'CALL_LOCAL_NO_PERSISTENT_HASH_CACHE',
                'remote_identity_authenticated':False}
