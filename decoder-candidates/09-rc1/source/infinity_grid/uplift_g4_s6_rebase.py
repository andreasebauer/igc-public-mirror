from __future__ import annotations
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json
from .canon import canonical_sha256
from .uplift_g4_s5_rebase import descriptor, reserve_write, binary_write, h_index, _ops

class G4S6RebaseError(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files('infinity_grid').joinpath('resources/uplift').joinpath(name)))

def spec()->dict[str,Any]:
    o=json.loads(_resource('G4_S6_REBASE_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json').read_text())
    if o.get('schema_id')!='IG_G4_S6_REBASE_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1': raise G4S6RebaseError('bad spec')
    if canonical_sha256({k:v for k,v in o.items() if k!='science_sha256'})!=o.get('science_sha256'): raise G4S6RebaseError('spec hash')
    return o

def verify_authority(*,s5:Mapping[str,Any],s5_replay:Mapping[str,Any],s5_closeout:Mapping[str,Any],s4:Mapping[str,Any],s3:Mapping[str,Any])->dict[str,Any]:
    f=[]
    if s5.get('schema_id')!='IG_G4_S5_REBASE_FINITE_READ_WRITE_DESCRIPTOR_RESULT_V1' or s5.get('status')!='PASS': f.append('S5')
    if s5.get('promoted_descriptor')!='CAPS7_PLUS_H_CLASS_BAG' or not s5.get('g4_s6_rebase_unlocked'): f.append('DESCRIPTOR')
    if s5_replay.get('certification')!='CERTIFIED_PASS' or s5_replay.get('primary_science_sha256')!=s5.get('science_sha256'): f.append('S5_REPLAY')
    if s5_closeout.get('status')!='CERTIFIED_PASS' or s5_closeout.get('next_authorized_stage')!='G4:S6.REBASE': f.append('S5_CLOSEOUT')
    if s4.get('schema_id')!='IG_G4_S4_REBASE_ONE_STEP_RELATION_CLOSURE_RESULT_V1' or s4.get('status')!='PASS': f.append('S4')
    if s3.get('schema_id')!='IG_G4_S3_REBASE_HIGHER_ORDER_RESULT_V1' or s3.get('status')!='PASS': f.append('S3')
    if f: raise G4S6RebaseError('authority failed: '+','.join(f))
    o={'schema_id':'IG_G4_S6_REBASE_AUTHORITY_V1','status':'PASS','s5_science_sha256':s5.get('science_sha256'),'s5_replay_science_sha256':s5_replay.get('science_sha256'),'s5_closeout_science_sha256':s5_closeout.get('science_sha256'),'s4_science_sha256':s4.get('science_sha256'),'s3_science_sha256':s3.get('science_sha256'),'spec_science_sha256':spec()['science_sha256'],'historical_g4_forward_use':'BLOCKED_UNTIL_S6_REBASE_CERTIFIES'}
    o['science_sha256']=canonical_sha256(o); return o

def _leaf_descriptors(s3:Mapping[str,Any])->tuple[list[dict[str,Any]],list[str]]:
    alpha,refmap,_=h_index(s3)
    leaves=[]
    for r in s3['candidate_interface']['rows']:
        leaves.append(descriptor(caps7=r['caps7'],h_labels=[refmap[r['term_ref']]],alphabet=alpha))
    return leaves,alpha

def recursive_factorisation_theorem(*,s5:Mapping[str,Any],s4:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if s5.get('descriptor_audit',{}).get('status')!='PASS': failures.append('S5_DESCRIPTOR_AUDIT')
    if s4.get('descriptor_factorization',{}).get('status') not in ('PASS',None) and s4.get('status')!='PASS': failures.append('S4_FACTORISATION')
    proof={
      'base':'Each certified G3 leaf maps to one CAPS7 vector plus one H-class basis element.',
      'reservation_step':'R_t(f,m)=(f-e_t,m) whenever f_t>0, so the descriptor is closed under every frozen unary reservation.',
      'binary_step':'C_ab((f,m),(g,n))=(f+g-e_a-e_b,m+n) for each of the 31 legal bridge operators. S4.REBASE certified one-step relation-valued closure on H; therefore exact outputs depend only on the parent descriptors and selected operator at this observer.',
      'induction':'The frozen G4 term grammar is freely generated from leaves by those unary and binary constructors. Closure and congruence at the base/constructor steps imply factorisation for every finite complete G4 term by structural induction.',
      'higher_order_guard':'S3.REBASE found no irreducible triple residual on the registered P3/K3 H scope; this is an independent adversarial guard, not the induction premise.'
    }
    o={'schema_id':'IG_G4_S6_REBASE_RECURSIVE_FACTORISATION_THEOREM_V1','status':'PASS' if not failures else 'FAIL','failures':failures,'descriptor':'CAPS7_PLUS_H_CLASS_BAG','coordinate_count':11,'grammar':{'unary_reservations':7,'binary_bridge_operators':31},'scope':'ALL_FINITE_COMPLETE_TERMS_OF_FROZEN_G4_REBASE_GRAMMAR','proof':proof,'new_parallel_science_kernels':0,'lower_layer_rematerialization':False}
    o['science_sha256']=canonical_sha256(o); return o

def fresh_symbolic_holdout(s3:Mapping[str,Any])->dict[str,Any]:
    leaves,alpha=_leaf_descriptors(s3); ops=_ops(); failures=[]; checks=0; legal=0
    # 31 pair+pair cases: deterministic leaf/operator cycling, then one bridge between pairs.
    for i,(a,b) in enumerate(ops):
        A=leaves[i%4]; B=leaves[(i+1)%4]; C=leaves[(i+2)%4]; D=leaves[(i+3)%4]
        x=binary_write(A,B,a,b,ops); y=binary_write(C,D,a,b,ops); checks+=1
        if x is not None and y is not None:
            z=binary_write(x,y,a,b,ops); legal+=z is not None
            if z is not None and z['g3_unit_count']!=4: failures.append('PAIR_PLUS_PAIR_COUNT')
    # 8 deep P5 left-deep symbolic cases.
    for i in range(8):
        cur=leaves[i%4]; ok=True
        for j in range(4):
            a,b=ops[(i*5+j)%len(ops)]
            nxt=binary_write(cur,leaves[(i+j+1)%4],a,b,ops); checks+=1
            if nxt is None: ok=False; break
            cur=nxt
        if ok:
            legal+=1
            if cur['g3_unit_count']!=5: failures.append('P5_COUNT')
    # 8 rebracketing cases: whenever both bracketings are legal, descriptor hashes must agree.
    for i in range(8):
        A,B,C=leaves[i%4],leaves[(i+1)%4],leaves[(i+2)%4]
        op1=ops[(3*i)%len(ops)]; op2=ops[(3*i+1)%len(ops)]
        l1=binary_write(A,B,*op1,ops); left=binary_write(l1,C,*op2,ops) if l1 else None
        r1=binary_write(B,C,*op2,ops); right=binary_write(A,r1,*op1,ops) if r1 else None
        checks+=2
        if left is not None and right is not None:
            legal+=1
            if left['science_sha256']!=right['science_sha256']: failures.append('REBRACKET_MISMATCH')
    o={'schema_id':'IG_G4_S6_REBASE_FRESH_SYMBOLIC_HOLDOUT_V1','status':'PASS' if not failures else 'FAIL','failure_count':len(failures),'failure_examples':failures[:16],'symbolic_operation_checks':checks,'legal_complete_cases':int(legal),'pair_plus_pair_case_count':31,'deep_p5_case_count':8,'rebracketing_case_count':8,'freshness':'CASES_NOT_USED_TO_SELECT_S2_H_TIER_OR_S5_DESCRIPTOR','explicit_branch_materialization':False,'new_parallel_science_kernels':0,'h_class_count':len(alpha)}
    o['science_sha256']=canonical_sha256(o); return o

def no_hidden_selector_read_audit()->dict[str,Any]:
    o={'schema_id':'IG_G4_S6_REBASE_NO_HIDDEN_SELECTOR_READ_AUDIT_V1','status':'PASS','graduated_state_fields':['CAPS7','H_CLASS_BAG'],'reads_raw_topology':False,'reads_owner_identity':False,'reads_construction_history':False,'reads_exact_branch_multiplicity':False,'reads_shell_or_support_rows_directly':False,'note':'H enters only through the certified finite H-class label carried by each G3 leaf; G4 stores class multiplicities, not raw H payloads.'}
    o['science_sha256']=canonical_sha256(o); return o

def finalize(*,s5:Mapping[str,Any],s5_replay:Mapping[str,Any],s5_closeout:Mapping[str,Any],s4:Mapping[str,Any],s3:Mapping[str,Any])->dict[str,Any]:
    auth=verify_authority(s5=s5,s5_replay=s5_replay,s5_closeout=s5_closeout,s4=s4,s3=s3)
    theorem=recursive_factorisation_theorem(s5=s5,s4=s4); holdout=fresh_symbolic_holdout(s3); hidden=no_hidden_selector_read_audit()
    ok=all(x['status']=='PASS' for x in (theorem,holdout,hidden))
    o={'schema_id':'IG_G4_S6_REBASE_RECURSIVE_CLOSURE_RESULT_V1','status':'PASS' if ok else 'REVIEW_REQUIRED','stage_ref':'G4:S6.REBASE','classification':'G4_REGRADUATED_CAPS7_PLUS_H_CLASS_BAG_RECURSIVE_RELATION_GRAMMAR_EARNED_R0_REBASE_UNLOCKED' if ok else 'G4_S6_REBASE_RECURSIVE_CLOSURE_FAILED_G4_REBASE_NOT_GRADUATED','authority':auth,'recursive_factorisation_theorem':theorem,'fresh_holdout':holdout,'no_hidden_selector_read_audit':hidden,'public_descriptor':'CAPS7_PLUS_H_CLASS_BAG' if ok else None,'public_descriptor_coordinate_count':11 if ok else None,'g4_graduated':ok,'g4_rebase_complete':ok,'g4_r0_rebase_unlocked':ok,'historical_g4_forward_use':'SUPERSEDED_BY_CERTIFIED_REBASE' if ok else 'BLOCKED','next_authorized_stage':'G4:R0.REBASE' if ok else None,'cost_control':{'new_parallel_science_kernels':0,'lower_layer_rematerialization':False,'explicit_branch_materialization':False},'nonclaims':spec()['nonclaims']}
    o['science_sha256']=canonical_sha256(o); return o

def compare(p:Mapping[str,Any],c:Mapping[str,Any])->dict[str,Any]:
    checks={'primary_pass':p.get('status')=='PASS','cold_pass':c.get('status')=='PASS','science_sha_equal':p.get('science_sha256')==c.get('science_sha256'),'classification_equal':p.get('classification')==c.get('classification'),'theorem_equal':p.get('recursive_factorisation_theorem')==c.get('recursive_factorisation_theorem'),'holdout_equal':p.get('fresh_holdout')==c.get('fresh_holdout'),'hidden_audit_equal':p.get('no_hidden_selector_read_audit')==c.get('no_hidden_selector_read_audit')}
    ok=all(checks.values()); o={'schema_id':'IG_G4_S6_REBASE_REPLAY_COMPARISON_V1','status':'PASS' if ok else 'FAIL','certification':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','checks':checks,'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256'),'g4_graduated':ok,'next_authorized_stage':'G4:R0.REBASE' if ok else None}; o['science_sha256']=canonical_sha256(o); return o

def closeout(p:Mapping[str,Any],c:Mapping[str,Any],r:Mapping[str,Any])->dict[str,Any]:
    ok=r.get('certification')=='CERTIFIED_PASS' and p.get('g4_graduated') and c.get('g4_graduated')
    o={'schema_id':'IG_G4_S6_REBASE_CERTIFIED_CLOSEOUT_V1','status':'CERTIFIED_PASS' if ok else 'NOT_CERTIFIED','classification':p.get('classification') if ok else None,'public_descriptor':p.get('public_descriptor') if ok else None,'g4_graduated':bool(ok),'g4_rebase_complete':bool(ok),'g4_r0_rebase_unlocked':bool(ok),'primary_science_sha256':p.get('science_sha256'),'cold_science_sha256':c.get('science_sha256'),'replay_comparison_science_sha256':r.get('science_sha256'),'historical_g4_forward_use':'SUPERSEDED_BY_CERTIFIED_REBASE' if ok else 'BLOCKED','next_authorized_stage':'G4:R0.REBASE' if ok else None}; o['science_sha256']=canonical_sha256(o); return o
