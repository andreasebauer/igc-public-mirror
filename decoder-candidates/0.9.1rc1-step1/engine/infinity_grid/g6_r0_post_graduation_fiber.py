from __future__ import annotations

"""G6:R0 post-graduation non-public fiber reconnaissance.

R0 is non-promoting.  It asks whether any exact carrier multiplicity remains
above one graduated Q_D_MARKER_RELATION public state.  The universal authority
is the already-earned global q_D injectivity/unique-marker decoder theorem,
not a new bounded enumeration.
"""
import hashlib, json, zipfile
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .v05_origin_guard import require_controller_execution_origin

STAGE_ID='G6:R0'
PLAN_SCHEMA='IG_G6_R0_POST_GRADUATION_FIBER_PREREGISTRATION_V1'
RESULT_SCHEMA='IG_G6_R0_POST_GRADUATION_FIBER_RESULT_V1'
PASS_CLASS='PASS_TRIVIAL_SINGLETON_FIBER_NO_ADDITIONAL_NONPUBLIC_CARRIER'

class G6R0Error(RuntimeError): pass

def _sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _member(path:Path,suffix:str)->dict[str,Any]:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1: raise G6R0Error('MEMBER_'+suffix)
        return json.loads(z.read(names[0]))

def derive_singleton_fiber(*,graduation:Mapping[str,Any],s5:Mapping[str,Any],s6:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if graduation.get('status')!='PASS' or graduation.get('decision')!='G6_GRADUATED' or graduation.get('g6_graduated') is not True:
        failures.append('G6_GRADUATION')
    if graduation.get('authorizes')!='G6:R0_POST_GRADUATION_STRUCTURAL_RECONNAISSANCE': failures.append('R0_AUTHORIZATION')
    pub=graduation.get('public_descriptor') or {}
    if pub.get('id')!='Q_D_MARKER_RELATION' or pub.get('decoder')!='ANY_CHILD_UNIQUE_D_DELETION': failures.append('GRADUATED_DESCRIPTOR')
    if s5.get('status')!='PASS' or s5.get('classification')!='PASS_S5_MARKER_OBSERVER_STATE_AND_GLOBAL_WRITE_LAW_EARNED': failures.append('S5_MARKER')
    if s5.get('global_marker_observer_injectivity_earned') is not True or s5.get('q_D_state_earned') is not True: failures.append('S5_GLOBAL_INJECTIVITY')
    cand=s5.get('candidate') or {}
    if cand.get('read_state')!='Q_D_MARKER_RELATION' or cand.get('observer_decoder')!='ANY_CHILD_UNIQUE_D_DELETION': failures.append('S5_DECODER')
    if s6.get('status')!='PASS' or s6.get('classification')!='RECURSIVE_CLOSURE_CANDIDATE_MARKER_STATE' or s6.get('recursive_closure_candidate_earned') is not True: failures.append('S6_RECURSIVE_CLOSURE')
    ob={x.get('id'):x.get('status') for x in s6.get('proof_obligations',[])}
    for oid in ('R3_QD_GLOBAL_INJECTIVITY','R4_WRITE_FACTORISATION','R6_STRUCTURAL_INDUCTION_ALL_FINITE_TERMS'):
        if ob.get(oid)!='PASS': failures.append('S6_'+oid)
    theorem={
      'schema_id':'IG_G6_R0_SINGLETON_FIBER_THEOREM_V1',
      'status':'PASS' if not failures else 'FAIL',
      'failures':failures,
      'proof_method':'DIRECT_COROLLARY_OF_GLOBAL_QD_INJECTIVITY_AND_UNIQUE_MARKER_DECODER',
      'fiber_definition':'Fiber(q)={ exact G6 carrier isomorphism classes [X] : q_D(X)=q } for realized graduated q values.',
      'injectivity_premise':'q_D(X)=q_D(Y) implies X is isomorphic to Y on the graduated domain.',
      'existence_premise':'Every realized q_D value is q_D(X) for at least one exact generated X.',
      'conclusion':'Every realized graduated public fiber has exactly one exact carrier isomorphism class.',
      'fiber_cardinality':1 if not failures else None,
      'composition_consequence':'No additional exact hidden state inside one graduated q_D fiber can alter exact future composition because the S6 write law factors exact composition through q_D globally.',
    }
    theorem['science_sha256']=canonical_sha256(theorem)
    return theorem

def run_g6_r0(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,accepted_source_sha256:str,internal_execution_id:str)->dict[str,Any]:
    require_controller_execution_origin('g6-r0-post-graduation-fiber')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID: raise G6R0Error('PLAN')
    if canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})!=plan.get('question_sha256'): raise G6R0Error('PLAN_HASH')
    auth=plan['authority']
    bindings={
      'master_prereg':'g6_master_file_sha256',
      'graduation_certificate':'g6_graduation_certificate_file_sha256',
      's5_marker_closeout':'s5_marker_certified_closeout_file_sha256',
      's6_marker_closeout':'s6_marker_certified_closeout_file_sha256',
    }
    for logical,key in bindings.items():
        if _sha(paths[logical])!=auth[key]: raise G6R0Error('AUTH_SHA_'+logical)
    if auth.get('executor_source_binding')!=accepted_source_sha256: raise G6R0Error('EXECUTOR_SOURCE_BINDING')
    graduation=json.loads(paths['graduation_certificate'].read_text(encoding='utf-8'))
    if graduation.get('graduation_science_sha256')!=auth['g6_graduation_science_sha256']: raise G6R0Error('GRADUATION_SCIENCE')
    s5=_member(paths['s5_marker_closeout'],'G6_S5R_MARKER_RESULT.json')
    s6=_member(paths['s6_marker_closeout'],'G6_S6R_MARKER_RESULT.json')
    if _sha_json_member(paths['s5_marker_closeout'],'G6_S5R_MARKER_RESULT.json')!=auth['s5_marker_result_file_sha256']: raise G6R0Error('S5_RESULT_FILE_SHA')
    if _sha_json_member(paths['s6_marker_closeout'],'G6_S6R_MARKER_RESULT.json')!=auth['s6_marker_result_file_sha256']: raise G6R0Error('S6_RESULT_FILE_SHA')
    if graduation.get('graduated_scope',{}).get('term_domain')!=plan['graduated_public_state']['scope']: raise G6R0Error('SCOPE')
    theorem=derive_singleton_fiber(graduation=graduation,s5=s5,s6=s6)
    passed=theorem['status']=='PASS'
    cls=PASS_CLASS if passed else 'R0_AUTHORITY_OR_PROOF_FAILURE'
    result={
      'schema_id':RESULT_SCHEMA,'status':'PASS' if passed else 'REVIEW_REQUIRED','stage_id':STAGE_ID,'classification':cls,
      'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],
      'promotion':False,'g6_graduation_preserved':True,'graduated_public_descriptor':'Q_D_MARKER_RELATION','public_descriptor_changed':False,
      'fiber_theorem':theorem,'fiber_status':'TRIVIAL_SINGLETON_UP_TO_EXACT_ISOMORPHISM' if passed else 'UNRESOLVED',
      'fiber_cardinality_per_realized_public_state':1 if passed else None,
      'intrinsic_finite_combinatorial_carrier':'ONE_POINT_TRIVIAL_CARRIER' if passed else None,
      'nonpublic_fiber_remaining':False if passed else None,
      'r0_complete':passed,'r1_earned':False if passed else None,'r1_started':False,
      'next_authorized':plan['next_on_trivial_fiber'] if passed else plan['next_on_unclassified'],
      'nonclaims':plan['nonclaims'],
    }
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out/'G6_R0_RESULT.json',result)
    write_json_atomic(out/'G6_R0_SINGLETON_FIBER_THEOREM.json',theorem)
    (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:R0 POST-GRADUATION FIBER\n\n'+cls+'\n\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8')
    return result

def _sha_json_member(path:Path,suffix:str)->str:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1: raise G6R0Error('MEMBER_'+suffix)
        return hashlib.sha256(z.read(names[0])).hexdigest()
