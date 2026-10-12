from __future__ import annotations
"""Decoder-owned passive science-cancel lifecycle control."""
import hashlib, json, os, re
from pathlib import Path
from typing import Any, Mapping
CANCEL_SCHEMA='IG_DECODER_SCIENCE_CANCEL_REQUEST_V1'
CANCEL_RECEIPT_SCHEMA='IG_DECODER_SCIENCE_CANCEL_RECEIPT_V1'
CANCEL_COMPLETION_SCHEMA='IG_DECODER_SCIENCE_CANCEL_COMPLETION_V1'
CANCEL_OPERATION='CANCEL_SCIENCE_REQUEST'
SCIENCE_JOB='DECODER.G6.SCIENCE'
_ID_RE=re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$')
_SHA_RE=re.compile(r'^[0-9a-f]{64}$')
class CancelControlError(ValueError): pass
def _canon(obj:Any)->bytes:return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
def _sha_obj(obj:Any)->str:return hashlib.sha256(_canon(obj)).hexdigest()
def _sha_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def _id(v:Any,name:str)->str:
    if type(v) is not str or _ID_RE.fullmatch(v) is None:raise CancelControlError('CANCEL_IDENTIFIER:'+name)
    return v
def validate_cancel_request(req:Mapping[str,Any])->dict[str,Any]:
    if type(req) is not dict:raise CancelControlError('CANCEL_REQUEST_TYPE')
    allowed={'schema_id','request_id','operation','target_request_id','parent_source_sha256','human_note'}
    if set(req)-allowed:raise CancelControlError('CANCEL_REQUEST_FIELD_FORBIDDEN')
    required={'schema_id','request_id','operation','target_request_id','parent_source_sha256'}
    if not required<=set(req):raise CancelControlError('CANCEL_REQUEST_FIELD_MISSING')
    if req['schema_id']!=CANCEL_SCHEMA or req['operation']!=CANCEL_OPERATION:raise CancelControlError('CANCEL_REQUEST_SCHEMA_OR_OPERATION')
    src=req['parent_source_sha256']
    if type(src) is not str or _SHA_RE.fullmatch(src) is None:raise CancelControlError('CANCEL_PARENT_SHA')
    out={'schema_id':CANCEL_SCHEMA,'request_id':_id(req['request_id'],'request_id'),'operation':CANCEL_OPERATION,'target_request_id':_id(req['target_request_id'],'target_request_id'),'parent_source_sha256':src}
    note=req.get('human_note')
    if note is not None:
        if type(note) is not str or len(note)>2048:raise CancelControlError('CANCEL_HUMAN_NOTE')
        out['human_note']=note
    return out
def _atomic_create(path:Path,obj:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    except BaseException:
        try:path.unlink()
        except OSError:pass
        raise
def submit_science_cancel_request(intake_root:str|Path,request:Mapping[str,Any])->dict[str,str]:
    obj=validate_cancel_request(request);root=Path(intake_root).resolve();path=root/'control_pending'/f"{obj['request_id']}.json"
    try:_atomic_create(path,obj)
    except FileExistsError as exc:raise CancelControlError('CANCEL_REQUEST_DUPLICATE:'+obj['request_id']) from exc
    return {'schema_id':CANCEL_RECEIPT_SCHEMA,'request_id':obj['request_id'],'request_sha256':_sha_obj(obj),'state':'PENDING_PASSIVE_CONTROL'}
def _load_json(path:Path)->dict[str,Any]:
    obj=json.loads(path.read_text(encoding='utf-8'))
    if type(obj) is not dict:raise CancelControlError('CANCEL_JSON_TYPE')
    return obj
def process_next_cancel_request(runtime_root:str|Path,*,expected_source_sha256:str)->dict[str,Any]|None:
    root=Path(runtime_root).resolve();cp=root/'intake'/'control_pending';cp.mkdir(parents=True,exist_ok=True);rows=sorted(cp.glob('*.json'))
    if not rows:return None
    path=rows[0];req=validate_cancel_request(_load_json(path))
    if req['parent_source_sha256']!=expected_source_sha256:raise CancelControlError('CANCEL_PARENT_SOURCE_MISMATCH')
    target=req['target_request_id'];pending=root/'intake'/'pending'/f'{target}.json';completed=root/'intake'/'completed'/f'{target}.json';cancelled=root/'intake'/'cancelled_operator'/f'{target}.json';cancelled.parent.mkdir(parents=True,exist_ok=True)
    control_done=root/'intake'/'control_completed'/f"{req['request_id']}.json";control_done.parent.mkdir(parents=True,exist_ok=True)
    active_path=root/'status'/'ACTIVE_REQUEST.json';active={}
    if active_path.is_file():
        try:active=_load_json(active_path)
        except Exception:active={}
    target_active=(active.get('state')=='ACTIVE' and active.get('request_id')==target and active.get('registered_job_id')==SCIENCE_JOB)
    target_sha=None;action=None
    if completed.is_file():action='ALREADY_COMPLETED'
    elif cancelled.is_file():action='ALREADY_CANCELLED';target_sha=_sha_file(cancelled)
    elif pending.is_file():
        from .v05_passive_intake import validate_passive_request
        pobj=validate_passive_request(_load_json(pending))
        if pobj['registered_job_id']!=SCIENCE_JOB:raise CancelControlError('CANCEL_TARGET_NOT_SCIENCE')
        target_sha=_sha_file(pending);os.replace(pending,cancelled)
        if _sha_file(cancelled)!=target_sha:raise CancelControlError('CANCEL_ARCHIVE_HASH_MISMATCH')
        action='ARCHIVED_ACTIVE_SCIENCE' if target_active else 'ARCHIVED_PENDING_SCIENCE'
    else:raise CancelControlError('CANCEL_TARGET_NOT_FOUND')
    rec={'schema_id':CANCEL_COMPLETION_SCHEMA,'status':'PASS','request_id':req['request_id'],'operation':CANCEL_OPERATION,'target_request_id':target,'target_request_sha256':target_sha,'action':action,'terminate_controller_child':bool(target_active),'accepted_source_sha256':expected_source_sha256,'scientific_effect':'NONE'}
    _atomic_create(control_done,rec);path.unlink();return rec
__all__=['CANCEL_SCHEMA','CANCEL_OPERATION','CancelControlError','validate_cancel_request','submit_science_cancel_request','process_next_cancel_request']
