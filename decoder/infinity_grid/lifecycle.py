from __future__ import annotations
import json
import re
from pathlib import Path
from .canon import canonical_sha256, write_json_atomic
from .safety import contained_path, validate_identifier

HOLD_MODE={
  'schema_id':'IG_EXECUTION_CAPABILITY_V0_17_1',
  'mode':'PHASE1_HOLD',
  'new_science_enabled':False,
  'reason':'Unified runtime has not yet consumed a successful Phase-1.1 graduation record.'
}

class GraduationError(RuntimeError): pass

class ExecutionModeManager:
    def __init__(self, paths): self.paths=paths
    @property
    def cap_dir(self):
        p=self.paths.store/'capabilities'; p.mkdir(parents=True,exist_ok=True); return p
    @property
    def grad_dir(self):
        p=self.paths.store/'graduation'; p.mkdir(parents=True,exist_ok=True); return p
    def hold_record(self):
        rec=dict(HOLD_MODE); rec['capability_sha256']=canonical_sha256(HOLD_MODE); return rec
    def active(self):
        pointer=self.grad_dir/'active.json'
        if not pointer.exists(): return self.hold_record()
        ptr=json.loads(pointer.read_text())
        sha=ptr.get('graduation_sha256')
        if not isinstance(sha,str) or len(sha)!=64: raise GraduationError('invalid graduation pointer')
        gp=self.grad_dir/f'{sha}.json'
        if not gp.is_file(): raise GraduationError('graduation record missing')
        rec=json.loads(gp.read_text())
        base={k:v for k,v in rec.items() if k!='graduation_sha256'}
        if canonical_sha256(base)!=sha: raise GraduationError('graduation record hash mismatch')
        self._validate_graduation_record(rec)
        return {
          'schema_id':'IG_EXECUTION_CAPABILITY_V0_17_1','mode':'GRADUATED_SCIENCE',
          'new_science_enabled':True,'graduation_sha256':sha,'audit_result_sha256':rec.get('audit_result_sha256')
        }

    def _validate_graduation_record(self, record:dict):
        if record.get('schema_id') != 'IG_GRADUATION_CERTIFICATE_V1':
            raise GraduationError('graduation requires IG_GRADUATION_CERTIFICATE_V1')
        if record.get('status')!='GRADUATED' or record.get('all_required_gates_passed') is not True:
            raise GraduationError('graduation requires all required gates PASS')
        audit_sha=record.get('audit_result_sha256')
        if not isinstance(audit_sha,str) or not re.fullmatch(r'[0-9a-f]{64}',audit_sha):
            raise GraduationError('graduation requires a bound audit_result_sha256')
        gates=record.get('required_gate_evidence')
        if not isinstance(gates,list) or not gates:
            raise GraduationError('graduation requires nonempty required_gate_evidence')
        for g in gates:
            if not isinstance(g,dict) or g.get('status')!='PASS' or not isinstance(g.get('evidence_sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',g['evidence_sha256']):
                raise GraduationError('graduation gate evidence is invalid/unbound')
        return True
    def install_graduation(self, record:dict):
        self._validate_graduation_record(record)
        base={k:v for k,v in record.items() if k!='graduation_sha256'}; sha=canonical_sha256(base)
        out=dict(base,graduation_sha256=sha); p=self.grad_dir/f'{sha}.json'
        if p.exists() and json.loads(p.read_text())!=out: raise GraduationError('graduation identity collision')
        if not p.exists(): write_json_atomic(p,out)
        write_json_atomic(self.grad_dir/'active.json',{'graduation_sha256':sha})
        return out
    def check_plan(self, descriptor:dict, plan:dict, mode:dict|None=None):
        # Caller-supplied mode dictionaries are never scientific authority.
        # The active installed graduation certificate is authoritative.
        mode=self.active(); sub=plan.get('subject',{}); pid=descriptor.get('protocol_id')
        # Frozen Phase-1 hold only blocks actual new-science boundaries, not historical/fixture replay.
        if not mode.get('new_science_enabled'):
            if pid=='SCOUT' and int(sub.get('level',0) or 0)>=16:
                raise GraduationError('L16 deferred until runtime graduation')
            if pid=='OSCOUT' and int(sub.get('level',0) or 0)>=7 and sub.get('live_generation',False):
                raise GraduationError('live O7 deferred until runtime graduation')
        return mode
