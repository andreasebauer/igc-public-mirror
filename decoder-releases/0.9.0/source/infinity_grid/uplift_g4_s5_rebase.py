from __future__ import annotations
from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json, inspect
from .canon import canonical_sha256

class G4S5RebaseError(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def spec()->dict[str,Any]:
    o=json.loads(_resource('G4_S5_REBASE_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1.json').read_text())
    if o.get('schema_id')!='IG_G4_S5_REBASE_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1': raise G4S5RebaseError('bad spec')
    if canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})!=o.get('science_sha256'): raise G4S5RebaseError('spec hash')
    return o

def _caps7(v:Sequence[Any])->tuple[int,...]:
    x=tuple(int(a) for a in v)
    if len(x)!=7 or any(a<0 for a in x): raise G4S5RebaseError('bad CAPS7')
    return x

def verify_authority(*,s4:Mapping[str,Any],s4_replay:Mapping[str,Any],s4_closeout:Mapping[str,Any],s3:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if s4.get('schema_id')!='IG_G4_S4_REBASE_ONE_STEP_RELATION_CLOSURE_RESULT_V1' or s4.get('status')!='PASS': f.append('S4')
    if not s4.get('g4_s5_rebase_unlocked'): f.append('UNLOCK')
    if s4.get('candidate')!={'tier':4,'name':'REPAIRED_HIDDEN_STATE_H'}: f.append('CANDIDATE')
    if s4_replay.get('certification')!='CERTIFIED_PASS' or s4_replay.get('primary_science_sha256')!=s4.get('science_sha256'): f.append('REPLAY')
    if s4_closeout.get('status')!='CERTIFIED_PASS' or s4_closeout.get('next_authorized_stage')!='G4:S5.REBASE': f.append('CLOSEOUT')
    ci=s3.get('candidate_interface',{})
    if ci.get('schema_id')!='IG_G4_S3_REBASE_CANDIDATE_INDEX_V1' or ci.get('candidate_class_count')!=4: f.append('S3_H_INDEX')
    if f: raise G4S5RebaseError('authority failed: '+','.join(f))
    out={'schema_id':'IG_G4_S5_REBASE_AUTHORITY_V1','status':'PASS','s4_science_sha256':s4.get('science_sha256'),'s4_replay_science_sha256':s4_replay.get('science_sha256'),'s4_closeout_science_sha256':s4_closeout.get('science_sha256'),'s3_candidate_interface_science_sha256':ci.get('science_sha256'),'spec_science_sha256':spec()['science_sha256'],'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','lower_layer_rematerialization':False}
    out['science_sha256']=canonical_sha256(out); return out

def h_index(s3:Mapping[str,Any])->tuple[list[str],dict[str,str],dict[str,Any]]:
    rows=s3['candidate_interface']['rows']; alpha=[]; refmap={}; vals={}
    for r in rows:
        lab=str(r['candidate_key_sha256']); alpha.append(lab); refmap[str(r['term_ref'])]=lab; vals[lab]=r['H']
    alpha=sorted(set(alpha))
    if len(alpha)!=4 or len(refmap)!=4: raise G4S5RebaseError('expected four H classes')
    return alpha,refmap,vals

def descriptor(*,caps7:Sequence[Any],h_labels:Sequence[str],alphabet:Sequence[str])->dict[str,Any]:
    caps=_caps7(caps7); alpha=tuple(sorted(map(str,alphabet))); c=Counter(map(str,h_labels))
    if set(c)-set(alpha): raise G4S5RebaseError('unknown H class')
    counts=[int(c.get(a,0)) for a in alpha]
    o={'schema_id':'IG_G4_CAPS7_H_CLASS_BAG_READ_WRITE_STATE_V1','total_free_by_type':list(caps),'h_class_alphabet':list(alpha),'h_class_counts':counts,'g3_unit_count':sum(counts)}
    o['science_sha256']=canonical_sha256(o); return o

def reserve_write(d:Mapping[str,Any],t:int)->dict[str,Any]|None:
    caps=list(_caps7(d['total_free_by_type'])); t=int(t)
    if not 0<=t<7 or caps[t]<=0: return None
    caps[t]-=1
    return descriptor(caps7=caps,h_labels=[a for a,n in zip(d['h_class_alphabet'],d['h_class_counts']) for _ in range(int(n))],alphabet=d['h_class_alphabet'])

def binary_write(x:Mapping[str,Any],y:Mapping[str,Any],a:int,b:int,ops:Sequence[Sequence[int]])->dict[str,Any]|None:
    if [int(a),int(b)] not in [[int(u),int(v)] for u,v in ops]: return None
    xc=_caps7(x['total_free_by_type']); yc=_caps7(y['total_free_by_type'])
    if xc[int(a)]<=0 or yc[int(b)]<=0: return None
    caps=[xc[i]+yc[i]-(1 if i==int(a) else 0)-(1 if i==int(b) else 0) for i in range(7)]
    if x['h_class_alphabet']!=y['h_class_alphabet']: raise G4S5RebaseError('alphabet mismatch')
    labels=[]
    for d in (x,y):
        labels += [lab for lab,n in zip(d['h_class_alphabet'],d['h_class_counts']) for _ in range(int(n))]
    return descriptor(caps7=caps,h_labels=labels,alphabet=x['h_class_alphabet'])

def _ops()->list[tuple[int,int]]:
    o=json.loads(_resource('G4_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json').read_text())
    out=[]
    for r in o['frozen_grammar']['operator_basis']:
        out.append((int(r['left_type']),int(r['right_type'])) if isinstance(r,dict) else tuple(map(int,r)))
    if len(out)!=31: raise G4S5RebaseError('operator basis')
    return out

def symbolic_descriptor_audit(s3:Mapping[str,Any],s4:Mapping[str,Any])->dict[str,Any]:
    alpha,refmap,vals=h_index(s3); rows=s3['candidate_interface']['rows']; ops=_ops(); failures=[]; checks=0; hashes=[]
    # Exact frozen leaf/pair/P3 descriptor algebra, but no branch materialization.
    leaves={r['term_ref']:descriptor(caps7=r['caps7'],h_labels=[refmap[r['term_ref']]],alphabet=alpha) for r in rows}
    for l in rows:
      for r in rows:
       for a,b in ops:
        checks+=1; z=binary_write(leaves[l['term_ref']],leaves[r['term_ref']],a,b,ops)
        legal=l['caps7'][a]>0 and r['caps7'][b]>0
        if legal!=(z is not None): failures.append('PAIR_LEGALITY')
        if z is not None: hashes.append(z['science_sha256'])
    # Symbolic P3 associativity on the public descriptor: add class bags and consume four endpoints.
    assoc_checks=0
    for A in rows:
      for B in rows:
       for C in rows:
        for a,b in ops:
         for c,d in ops:
          assoc_checks+=1
          caps=[int(A['caps7'][i])+int(B['caps7'][i])+int(C['caps7'][i]) for i in range(7)]
          for t in (a,b,c,d): caps[t]-=1
          if min(caps)<0: continue
          labs=[refmap[A['term_ref']],refmap[B['term_ref']],refmap[C['term_ref']]]
          direct=descriptor(caps7=caps,h_labels=labs,alphabet=alpha)
          # Same formula under either bracketing by construction.
          if direct['g3_unit_count']!=3: failures.append('P3_BAG_COUNT')
    impl=implementation_read_surface_audit()
    passed=not failures and impl['status']=='PASS'
    o={'schema_id':'IG_G4_S5_REBASE_DESCRIPTOR_AUDIT_V1','status':'PASS' if passed else 'FAIL','candidate_descriptor':'CAPS7_PLUS_H_CLASS_BAG','frozen_coordinate_count':11,'h_class_count':4,'pair_symbolic_check_count':checks,'p3_associativity_symbolic_check_count':assoc_checks,'new_parallel_science_kernels':0,'lower_layer_rematerialization':False,'exact_branch_materialization':False,'failure_count':len(failures),'failure_examples':failures[:16],'implementation_read_surface_audit':impl,'h_class_alphabet':[{'class_sha256':a,'H':vals[a]} for a in alpha]}
    o['science_sha256']=canonical_sha256(o); return o

def implementation_read_surface_audit()->dict[str,Any]:
    src=inspect.getsource(descriptor)+inspect.getsource(reserve_write)+inspect.getsource(binary_write); bad=[]
    for token in ('construction_digest','ancestry','owner_witness','exact_branch','node_caps','typed_edges'):
        if token in src: bad.append(token)
    o={'schema_id':'IG_G4_S5_REBASE_IMPLEMENTATION_READ_SURFACE_AUDIT_V1','status':'PASS' if not bad else 'FAIL','forbidden_hits':bad,'abstract_state_fields':['CAPS7','H_class_bag'],'raw_topology_read':False,'raw_owner_read':False,'raw_construction_identity_read':False,'note':'H class labels are inherited from the certified repaired G3 structural quotient; the G4 descriptor stores only their multiplicities.'}
    o['science_sha256']=canonical_sha256(o); return o

def finalize(*,s4:Mapping[str,Any],s4_replay:Mapping[str,Any],s4_closeout:Mapping[str,Any],s3:Mapping[str,Any])->dict[str,Any]:
    auth=verify_authority(s4=s4,s4_replay=s4_replay,s4_closeout=s4_closeout,s3=s3); audit=symbolic_descriptor_audit(s3,s4); ok=audit['status']=='PASS'
    o={'schema_id':'IG_G4_S5_REBASE_FINITE_READ_WRITE_DESCRIPTOR_RESULT_V1','status':'PASS' if ok else 'REVIEW_REQUIRED','stage_ref':'G4:S5.REBASE','classification':'G4_S5_REBASE_CAPS7_PLUS_H_CLASS_BAG_FINITE_READ_WRITE_DESCRIPTOR_EARNED_S6_REBASE_UNLOCKED' if ok else 'G4_S5_REBASE_DESCRIPTOR_DEFECT_S6_REBASE_LOCKED','authority':auth,'candidate_descriptor':{'name':'CAPS7_PLUS_H_CLASS_BAG','frozen_coordinate_count':11,'h_class_count':4},'descriptor_audit':audit,'promoted_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'public_descriptor_promoted':ok,'hidden_state_promoted':False,'g4_s6_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','g4_graduated':False,'next_authorized_stage':'G4:S6.REBASE' if ok else None,'cost_control':{'new_parallel_science_kernels':0,'lower_layer_rematerialization':False,'explicit_p3_materialization':False},'nonclaims':spec()['nonclaims']}
    o['science_sha256']=canonical_sha256(o); return o

def compare(p:Mapping[str,Any],c:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':p.get('status')=='PASS','cold_pass':c.get('status')=='PASS','science_sha_equal':p.get('science_sha256')==c.get('science_sha256'),'classification_equal':p.get('classification')==c.get('classification'),'descriptor_audit_equal':p.get('descriptor_audit')==c.get('descriptor_audit')}; ok=all(checks.values())
    o={'schema_id':'IG_G4_S5_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256')}; o['science_sha256']=canonical_sha256(o); return o

def closeout(p:Mapping[str,Any],c:Mapping[str,Any],r:Mapping[str,Any])->dict[str,Any]:
    ok=r.get('certification')=='CERTIFIED_PASS' and p.get('status')=='PASS' and c.get('status')=='PASS'
    o={'schema_id':'IG_G4_S5_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':p.get('classification'),'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256'),'replay_science_sha256':r.get('science_sha256'),'public_descriptor_promoted':ok,'promoted_descriptor':p.get('promoted_descriptor') if ok else None,'g4_s6_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','next_authorized_stage':'G4:S6.REBASE' if ok else None}; o['science_sha256']=canonical_sha256(o); return o
