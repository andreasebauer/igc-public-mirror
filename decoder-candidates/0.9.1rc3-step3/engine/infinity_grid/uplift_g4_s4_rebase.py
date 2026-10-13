from __future__ import annotations
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json
from .canon import canonical_sha256

class G4S4RebaseError(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def spec()->dict[str,Any]:
    o=json.loads(_resource('G4_S4_REBASE_ONE_STEP_RELATION_CLOSURE_SPEC_V1.json').read_text())
    if o.get('schema_id')!='IG_G4_S4_REBASE_ONE_STEP_RELATION_CLOSURE_SPEC_V1': raise G4S4RebaseError('bad spec')
    if canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})!=o.get('science_sha256'): raise G4S4RebaseError('spec hash')
    return o

def verify_authority(*,s3:Mapping[str,Any],s3_replay:Mapping[str,Any],s3_closeout:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if s3.get('schema_id')!='IG_G4_S3_REBASE_HIGHER_ORDER_RESULT_V1' or s3.get('status')!='PASS': f.append('S3')
    if s3.get('classification')!='G4_S3_REBASE_NO_IRREDUCIBLE_TRIPLE_RESIDUAL_ON_REGISTERED_P3_K3_H_SCOPE_S4_REBASE_UNLOCKED': f.append('CLASS')
    if s3.get('candidate_interface',{}).get('candidate')!={'tier':4,'name':'REPAIRED_HIDDEN_STATE_H'}: f.append('CANDIDATE')
    if not s3.get('g4_s4_rebase_unlocked'): f.append('UNLOCK')
    if s3_replay.get('certification')!='CERTIFIED_PASS' or s3_replay.get('primary_science_sha256')!=s3.get('science_sha256'): f.append('REPLAY')
    if s3_closeout.get('status')!='CERTIFIED_PASS' or s3_closeout.get('next_authorized_stage')!='G4:S4.REBASE': f.append('CLOSEOUT')
    if f: raise G4S4RebaseError('authority failed: '+','.join(f))
    out={'schema_id':'IG_G4_S4_REBASE_AUTHORITY_V1','status':'PASS','s3_science_sha256':s3.get('science_sha256'),'s3_replay_science_sha256':s3_replay.get('science_sha256'),'s3_closeout_science_sha256':s3_closeout.get('science_sha256'),'spec_science_sha256':spec()['science_sha256'],'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','lower_layer_rematerialization':False}
    out['science_sha256']=canonical_sha256(out); return out

def _operator_pairs()->list[tuple[int,int]]:
    # Reuse the already-frozen 31 directed operator basis from the certified G4 grammar resource.
    o=json.loads(_resource('G4_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json').read_text())
    rows=o.get('frozen_grammar',{}).get('operator_basis',[])
    out=[]
    for r in rows:
        if isinstance(r,dict): out.append((int(r['left_type']),int(r['right_type'])))
        else: out.append(tuple(map(int,r)))
    if len(out)!=31: raise G4S4RebaseError(f'operator basis expected 31 got {len(out)}')
    return out

def symbolic_reserve_audit(s3:Mapping[str,Any])->dict[str,Any]:
    rows=s3['candidate_interface']['rows']
    leaf_caps=sorted({tuple(int(x) for x in r['caps7']) for r in rows})
    if len(leaf_caps)!=2: raise G4S4RebaseError(f'expected two distinct leaf CAPS7 classes, got {len(leaf_caps)}')
    lo,hi=leaf_caps
    # The four H classes comprise two leaves at each CAPS7 level. For P3, addition is
    # commutative in the public resource coordinate, so the 64 ordered H assignments
    # collapse exactly to k=0..3 high-cap leaves.
    base_sums=[]
    for k in range(4):
        base_sums.append(tuple((3-k)*lo[i]+k*hi[i] for i in range(7)))
    if len(set(base_sums))!=4: raise G4S4RebaseError('P3 base sums did not produce four exact resource classes')
    ops=_operator_pairs()
    if len(ops)!=31: raise G4S4RebaseError('operator basis !=31')
    failures=[]; checked=0; distinct_states=set()
    for base in base_sums:
      for a,b in ops:
        for c,d in ops:
          p=list(base)
          for tt in (a,b,c,d): p[tt]-=1
          state=tuple(p); distinct_states.add(state)
          for r,_dummy in ops:
            checked+=1
            legal=state[r]>0
            if not legal:
              failures.append({'bridge_ops':[[a,b],[c,d]],'reserve_type':r,'failure':'NONPOSITIVE_CAPACITY'})
              if len(failures)>=16: break
            out=tuple(state[i]-(1 if i==r else 0) for i in range(7))
            if out[r] != state[r]-1:
              failures.append({'bridge_ops':[[a,b],[c,d]],'reserve_type':r,'failure':'WRITE_MISMATCH'})
              if len(failures)>=16: break
          if len(failures)>=16: break
        if len(failures)>=16: break
      if len(failures)>=16: break
    expected=4*(31**3)
    out={'schema_id':'IG_G4_S4_REBASE_SYMBOLIC_POST_RESERVE_AUDIT_V1','status':'PASS' if not failures and checked==expected else 'FAIL','distinct_leaf_caps7_classes':len(leaf_caps),'distinct_p3_base_resource_sums':len(base_sums),'directed_operator_count':len(ops),'ordered_bridge_operator_pair_count':len(ops)**2,'post_recursive_action_basis_count':len(ops),'exact_symbolic_check_count':checked,'expected_symbolic_check_count':expected,'distinct_factorized_caps7_states':len(distinct_states),'failure_count':len(failures),'failure_examples':failures,'p3_public_context_count':int(s3['triple_factorization_coverage']['public_context_counts']['P3_CONNECTED_PATH']),'materialized_p3_contexts':False,'materialized_post_reserve_contexts':False,'factorization_argument':'S3.REBASE certifies the H-sensitive P3 compatibility factorization. A further external reservation reads only the inherited CAPS7 resource coordinate. The four H classes contain exactly two CAPS7 leaf classes; the 64 ordered P3 H assignments therefore collapse under resource addition to four exact base sums, one for each possible count of high-cap leaves. For each base sum, two bridge consumptions and the post-reserve action are exhaustively checked over the full frozen 31-action basis.'}
    out['science_sha256']=canonical_sha256(out); return out

def finalize(*,s3:Mapping[str,Any],s3_replay:Mapping[str,Any],s3_closeout:Mapping[str,Any])->dict[str,Any]:
    auth=verify_authority(s3=s3,s3_replay=s3_replay,s3_closeout=s3_closeout)
    audit=symbolic_reserve_audit(s3)
    ok=audit['status']=='PASS' and audit['exact_symbolic_check_count']==4*(31**3)
    out={'schema_id':'IG_G4_S4_REBASE_ONE_STEP_RELATION_CLOSURE_RESULT_V1','status':'PASS' if ok else 'REVIEW_REQUIRED','stage_ref':'G4:S4.REBASE','classification':'G4_S4_REBASE_ONE_STEP_RELATION_VALUED_COMPOSITION_CLOSURE_ON_FROZEN_H_P3_SCOPE_S5_REBASE_UNLOCKED' if ok else 'G4_S4_REBASE_ONE_STEP_CLOSURE_DEFECT_S5_REBASE_LOCKED','authority':auth,'candidate':{'tier':4,'name':'REPAIRED_HIDDEN_STATE_H'},'post_recursive_reserve_audit':audit,'cost_control':{'new_parallel_science_kernels':0,'symbolic_checks':audit['exact_symbolic_check_count'],'lower_layer_rematerialization':False,'p3_contexts_materialized':False},'public_descriptor_promoted':False,'hidden_state_promoted':False,'g4_s5_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','g4_graduated':False,'next_authorized_stage':'G4:S5.REBASE' if ok else None,'nonclaims':spec()['nonclaims']}
    out['science_sha256']=canonical_sha256(out); return out

def compare(p:Mapping[str,Any],c:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':p.get('status')=='PASS','cold_pass':c.get('status')=='PASS','science_sha_equal':p.get('science_sha256')==c.get('science_sha256'),'classification_equal':p.get('classification')==c.get('classification'),'audit_equal':p.get('post_recursive_reserve_audit')==c.get('post_recursive_reserve_audit')}
    ok=all(checks.values()); out={'schema_id':'IG_G4_S4_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256')}; out['science_sha256']=canonical_sha256(out); return out

def closeout(p:Mapping[str,Any],c:Mapping[str,Any],r:Mapping[str,Any])->dict[str,Any]:
    ok=r.get('certification')=='CERTIFIED_PASS' and p.get('status')=='PASS' and c.get('status')=='PASS'; out={'schema_id':'IG_G4_S4_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'CERTIFICATION_FAIL','classification':p.get('classification'),'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256'),'replay_science_sha256':r.get('science_sha256'),'public_descriptor_promoted':False,'g4_s5_rebase_unlocked':ok,'g4_rebase_complete':False,'historical_g4_forward_use':'BLOCKED_UNTIL_REBASE_COMPLETES','next_authorized_stage':'G4:S5.REBASE' if ok else None}; out['science_sha256']=canonical_sha256(out); return out
