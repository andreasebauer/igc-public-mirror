from __future__ import annotations
import json
from importlib.resources import files
from pathlib import Path
from .canon import canonical_sha256, write_json_atomic
from .schema import validate
from .resources import schema
from .firewall import enforce_plan

class ProtocolRegistry:
    def __init__(self):
        # Multiple frozen protocol versions may coexist.  The canonical filename
        # <PROTOCOL_ID>.json remains the default used by create_plan(), while
        # immutable historical/future plans are resolved by their exact
        # descriptor SHA-256 and version.  This lets new science extend a
        # protocol without invalidating old replay plans.
        self._desc={}
        self._by_sha={}
        self._by_pid_version={}
        root=files('infinity_grid').joinpath('resources/protocols')
        for p in sorted(root.iterdir(),key=lambda q:q.name):
            if not p.name.endswith('.json'):
                continue
            o=json.loads(p.read_text(encoding='utf-8'))
            errs=validate(schema('PROTOCOL_DESCRIPTOR_SCHEMA_v0.17.json'),o,raise_on_error=False)
            if errs: raise ValueError(f'protocol descriptor {p.name} invalid: {errs}')
            o=dict(o); base={k:v for k,v in o.items() if k!='descriptor_sha256'}; observed=canonical_sha256(base)
            declared=o.get('descriptor_sha256')
            if declared is not None and declared!=observed:
                raise ValueError(f'protocol descriptor {p.name} declared hash mismatch')
            o['descriptor_sha256']=observed
            sha=o['descriptor_sha256']; pid=o['protocol_id']; ver=str(o.get('version'))
            if sha in self._by_sha and self._by_sha[sha]!=o:
                raise ValueError(f'duplicate protocol descriptor SHA conflict {sha}')
            self._by_sha[sha]=o
            key=(pid,ver)
            self._by_pid_version.setdefault(key,[]).append(o)
            # Preserve backwards-compatible planning default from PID.json.
            if p.name==f'{pid}.json':
                if pid in self._desc and self._desc[pid]!=o:
                    raise ValueError(f'duplicate default protocol {pid}')
                self._desc[pid]=o
        # For protocol families with no canonical default filename, choose the
        # lexicographically smallest stable descriptor solely as a fallback.
        for o in self._by_sha.values():
            self._desc.setdefault(o['protocol_id'],o)
    def list(self):
        return sorted(self._by_sha.values(),key=lambda d:(d['protocol_id'],str(d.get('version')),d['descriptor_sha256']))
    def get(self,pid,version=None,descriptor_sha=None):
        if descriptor_sha is not None:
            if descriptor_sha not in self._by_sha: raise KeyError(descriptor_sha)
            d=self._by_sha[descriptor_sha]
            if d['protocol_id']!=pid: raise KeyError((pid,descriptor_sha))
            if version is not None and str(d.get('version'))!=str(version): raise KeyError((pid,version,descriptor_sha))
            return d
        if version is not None:
            ds=self._by_pid_version.get((pid,str(version)),[])
            if len(ds)==1:return ds[0]
            if not ds:raise KeyError((pid,version))
            raise KeyError(f'ambiguous protocol version {(pid,version)}; descriptor_sha required')
        if pid not in self._desc: raise KeyError(pid)
        return self._desc[pid]
    def _match_profile(self,d,plan):
        profiles=d.get('plan_profiles',[])
        if not profiles: raise ValueError(f'protocol {d["protocol_id"]} has no exact plan profile')
        stages=plan.get('stages',[])
        for prof in profiles:
            if prof.get('allowed_protocol_labels') and plan.get('protocol_label') not in prof.get('allowed_protocol_labels',[]): continue
            sub=plan.get('subject',{})
            if any(sub.get(k)!=v for k,v in prof.get('subject_equals',{}).items()): continue
            if prof.get('subject_level_min') is not None and int(sub.get('level',-10**9)) < int(prof['subject_level_min']): continue
            if prof.get('subject_level_max') is not None and int(sub.get('level',10**9)) > int(prof['subject_level_max']): continue
            ps=prof.get('stages',[])
            if len(ps)!=len(stages): continue
            ok=True
            for a,b in zip(stages,ps):
                if a.get('stage_id')!=b.get('stage_id') or a.get('depends_on',[])!=b.get('depends_on',[]) or a.get('runner')!=b.get('runner'): ok=False; break
                allowed=set(b.get('allowed_param_keys',[])); actual=set(a.get('params',{}))
                if actual-allowed: ok=False; break
                required=set(b.get('required_param_keys',[]))
                if not required.issubset(actual): ok=False; break
            if ok:
                allowed_flags=set(prof.get('allowed_input_flags',[])); actual_flags=set(plan.get('input_flags',{}))
                if actual_flags-allowed_flags: continue
                return prof
        raise ValueError('plan stages/DAG/runners/flags do not match any frozen protocol plan profile')
    def verify_plan(self,plan,*,paths=None,phase1=None):
        d=self.get(plan['protocol_id'],version=plan.get('protocol_version'),descriptor_sha=plan.get('descriptor_sha256'))
        enforce_plan(d,plan)
        if plan.get('protocol_version')!=d['version']: raise ValueError('protocol version mismatch')
        if plan.get('descriptor_sha256')!=d['descriptor_sha256']: raise ValueError('protocol descriptor hash mismatch')
        self._match_profile(d,plan)
        q=plan.get('question',{})
        if q.get('protocol_id')!=d['protocol_id']: raise ValueError('question protocol mismatch')
        if q.get('stopping_rules')!=d.get('stopping_rules',{}): raise ValueError('stopping-rule contract mismatch')
        if q.get('protocol_label')!=plan.get('protocol_label'): raise ValueError('question protocol-label mismatch')
        if q.get('subject')!=plan.get('subject'): raise ValueError('question subject mismatch')
        if q.get('input_dataset_sha256') not in {x.get('dataset_sha256') for x in plan.get('input_datasets',[])}: raise ValueError('question input dataset mismatch')
        if canonical_sha256(q)!=plan.get('question_sha256'): raise ValueError('question hash mismatch')
        oc=d.get('observer_contract') or {}; flags=plan.get('input_flags',{})
        if oc.get('must_be_explicit') and not plan.get('observer'): raise ValueError('explicit observer required')
        if oc.get('observer_freeze_before_generation') and plan.get('subject',{}).get('live_generation') and flags.get('observer_frozen') is not True: raise ValueError('live generation requires observer_frozen=true')
        ic=d.get('input_contract') or {}
        if ic.get('parent_certificate_required_for_live') and plan.get('subject',{}).get('live_generation') and flags.get('parent_certified') is not True: raise ValueError('live generation requires parent_certified=true')
        if ic.get('bounded_preregistered') and flags.get('bounded_preregistered') is not True: raise ValueError('bounded reconnaissance requires bounded_preregistered=true')
        if ic.get('separate_lineage_required') and flags.get('separate_lineage') is not True: raise ValueError('separate_lineage=true required')
        if paths is not None: self.verify_installed(paths,d['descriptor_sha256'])
        return True
    def install_into_store(self,paths):
        pdir=paths.store/'protocols'; pdir.mkdir(parents=True,exist_ok=True)
        for d in self.list():
            out=pdir/f"{d['descriptor_sha256']}.json"
            if out.exists():
                observed=json.loads(out.read_text())
                if observed!=d: raise RuntimeError(f'protocol store corruption/conflict: {out}')
                base={k:v for k,v in observed.items() if k!='descriptor_sha256'}
                if canonical_sha256(base)!=d['descriptor_sha256']: raise RuntimeError(f'protocol store hash mismatch: {out}')
            else: write_json_atomic(out,d)
        return len(self._desc)
    def verify_installed(self,paths,sha):
        p=paths.store/'protocols'/f'{sha}.json'
        if not p.is_file(): raise FileNotFoundError(p)
        o=json.loads(p.read_text()); base={k:v for k,v in o.items() if k!='descriptor_sha256'}
        if o.get('descriptor_sha256')!=sha or canonical_sha256(base)!=sha: raise RuntimeError('stored protocol identity mismatch')
        d=self.get(o['protocol_id'],version=o.get('version'),descriptor_sha=sha)
        if d!=o: raise RuntimeError('stored protocol bytes differ from executable descriptor')
        return True
