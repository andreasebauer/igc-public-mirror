from __future__ import annotations
import json
from .canon import write_json_atomic
from .firewall import enforce_claim
from .protocols import ProtocolRegistry
from .resources import schema
from .schema import validate
from .safety import validate_identifier
from .verification import verify_run

def _claim_path(paths,cid,ver):
    validate_identifier(cid,field='claim_id'); d=paths.store/'claims'/cid; d.mkdir(parents=True,exist_ok=True); return d/f'v{int(ver):04d}.json'

def write_claim(paths,claim:dict,run_record:dict):
    errs=validate(schema('CLAIM_RECORD_SCHEMA_v0.17.json'),claim,raise_on_error=False)
    if errs: raise ValueError(f'invalid claim: {errs}')
    validate_identifier(claim['claim_id'],field='claim_id')
    rid=run_record.get('run_id'); vr=verify_run(paths,rid)
    if vr.get('status')!='PASS': raise RuntimeError(f'claim evidence run failed verification: {vr}')
    desc=ProtocolRegistry().get(run_record['protocol']['protocol_id']); enforce_claim(desc,run_record,claim)
    # All evidence refs must resolve to verified runs. Earned claims require this run among evidence.
    refs=claim.get('evidence_refs',[])
    if claim.get('status')=='EARNED_SCOPED' and rid not in refs: raise RuntimeError('earned claim must cite creating run')
    for ref in refs:
        if not isinstance(ref,str): raise RuntimeError('evidence run ref must be a run_id string')
        if verify_run(paths,ref).get('status')!='PASS': raise RuntimeError(f'unresolved/unverified evidence run {ref}')
    for dep in claim.get('dependencies',[]):
        validate_identifier(dep,field='dependency claim_id'); d=paths.store/'claims'/dep
        if not d.is_dir() or not list(d.glob('*.json')): raise RuntimeError(f'unresolved claim dependency {dep}')
    p=_claim_path(paths,claim['claim_id'],claim['version'])
    if p.exists():
        old=json.loads(p.read_text())
        if old!=claim: raise RuntimeError('claim version immutable')
    else: write_json_atomic(p,claim)
    # Reciprocal immutable run linkage: append only the exact claim version.
    rp=paths.runs/rid/'run.json'; rr=json.loads(rp.read_text()); ref={'claim_id':claim['claim_id'],'version':int(claim['version'])}
    if ref not in rr.get('claim_refs',[]): rr.setdefault('claim_refs',[]).append(ref); write_json_atomic(rp,rr)
    return p

def list_claims(paths):
    out=[]
    for p in sorted((paths.store/'claims').glob('*/*.json')):
        try: out.append(json.loads(p.read_text()))
        except Exception: pass
    return out
