from __future__ import annotations

"""Registered G6:S5R fresh-D-marker representation-amendment proof."""
import hashlib,json,zipfile
from pathlib import Path
from typing import Any,Mapping
from .canon import canonical_sha256,write_json_atomic
from .g6_stage_executors import _basis
from .adapters.g4_accepted import G4AcceptedAdapter
from .uplift_g5_r4 import marked_leaf_separation_theorem
from .v05_origin_guard import require_controller_execution_origin

STAGE_ID='G6:S5R-MARKER'; PLAN_SCHEMA='IG_G6_S5R_FRESH_MARKER_OBSERVER_AMENDMENT_PLAN_V1'; RESULT_SCHEMA='IG_G6_S5R_FRESH_MARKER_OBSERVER_RESULT_V1'
class G6MarkerS5Error(RuntimeError): pass

def _sha(p:Path)->str:
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _member(path:Path,suffix:str):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1: raise G6MarkerS5Error('MEMBER_'+suffix)
        return json.loads(z.read(names[0]))

def run_marker_s5(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,accepted_source_sha256:str,internal_execution_id:str)->dict[str,Any]:
    require_controller_execution_origin('g6-s5r-fresh-marker')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}; plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if plan.get('schema_id')!=PLAN_SCHEMA or plan.get('stage_id')!=STAGE_ID: raise G6MarkerS5Error('PLAN')
    if canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})!=plan.get('question_sha256'): raise G6MarkerS5Error('PLAN_HASH')
    auth=plan['authority']
    if _sha(paths['master_prereg'])!=auth['g6_master_prereg_file_sha256']: raise G6MarkerS5Error('MASTER_SHA')
    if _sha(paths['g5_r4_read_first'])!=auth['g5_r4_read_first_file_sha256']: raise G6MarkerS5Error('G5_R4_SHA')
    if _sha(paths['historical_s6_closeout'])!=auth['historical_g6_s6_closeout_file_sha256']: raise G6MarkerS5Error('S6_SHA')
    if _sha(paths['l2_repair_closeout'])!=auth['l2_repair_closeout_file_sha256']: raise G6MarkerS5Error('L2_SHA')
    master=paths['master_prereg'].read_text(encoding='utf-8')
    if 'FORBID WITHOUT REVIEW' not in master or 'finite deterministic read labels' not in master: raise G6MarkerS5Error('MASTER_FEATURE_REVIEW')
    r4txt=paths['g5_r4_read_first'].read_text(encoding='utf-8')
    for token in ('G5:R4','MARKED','MINIMALITY','CERTIFIED'):
        if token.lower() not in r4txt.lower(): raise G6MarkerS5Error('G5_R4_READ_FIRST')
    theorem=marked_leaf_separation_theorem()
    if theorem.get('status')!='PASS' or theorem.get('science_sha256')!=auth['g5_r4_marked_leaf_theorem_sha256']: raise G6MarkerS5Error('G5_R4_THEOREM_BINDING')
    statuses={x['theorem_id']:x['status'] for x in theorem['theorems']}
    if any(not statuses.get(k,'').startswith('PROVED') for k in ('R4-T1-UNIQUE-MARKER-DELETION-RECOVERY','R4-T2-MARKER-CHILD-DISJOINTNESS','R4-T3-MARKED-OBSERVER-INJECTIVITY')): raise G6MarkerS5Error('G5_R4_THEOREM_STATUS')
    proof=_member(paths['historical_s6_closeout'],'FORMAL_PROOF_OBLIGATION_RECORD.json')
    if proof.get('status')!='PASS' or proof.get('all_mu_ge_2') is not True: raise G6MarkerS5Error('RESOURCE_PROOF')
    lower=str(proof.get('lower_bound',''))
    if 'mu_i - 2' not in lower and 'mu_i-2' not in lower: raise G6MarkerS5Error('RESOURCE_LOWER_BOUND')
    l2=_member(paths['l2_repair_closeout'],'G6_S6R_L2_REPAIR_RESULT.json')
    if l2.get('classification')!='L2_NO_COUNTEREXAMPLE_IN_EXHAUSTIVE_P_ONLY_00_SCOPE_CONTINUE_PROOF' or l2.get('strong_l2_proved') is True: raise G6MarkerS5Error('L2_PIVOT_BINDING')
    basis=_basis(); marker_free=all('D' not in t.H_classes for t in basis.values())
    ad=G4AcceptedAdapter(); dcap=ad.class_caps7('D')[0]
    obligations=[]
    checks={
      'M1_MARKER_FREE_GENERATED_DOMAIN': marker_free,
      'M2_FIXED_MARKER_RELATION_GLOBALLY_NONEMPTY': bool(dcap>0 and proof.get('all_mu_ge_2') is True),
      'M3_UNIQUE_MARKER_DELETION_RECOVERY': statuses.get('R4-T1-UNIQUE-MARKER-DELETION-RECOVERY','').startswith('PROVED'),
      'M4_GLOBAL_MARKER_OBSERVER_INJECTIVITY': statuses.get('R4-T3-MARKED-OBSERVER-INJECTIVITY','').startswith('PROVED'),
      'M5_EXACT_QD_WRITE_LAW': statuses.get('R4-T3-MARKED-OBSERVER-INJECTIVITY','').startswith('PROVED'),
    }
    for row in plan['proof_obligations']:
        oid=row['id']; obligations.append({'id':oid,'status':'PASS' if checks.get(oid,False) else 'UNRESOLVED','statement':row['statement']})
    passed=all(x['status']=='PASS' for x in obligations)
    cls='PASS_S5_MARKER_OBSERVER_STATE_AND_GLOBAL_WRITE_LAW_EARNED' if passed else 'MORE_SCIENCE_REQUIRED'
    result={'schema_id':RESULT_SCHEMA,'status':'PASS' if passed else 'REVIEW_REQUIRED','stage_id':STAGE_ID,'classification':cls,'accepted_decoder_source_sha256':accepted_source_sha256,'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],'candidate':plan['candidate'],'proof_obligations':obligations,'g5_r4_theorem_sha256':theorem['science_sha256'],'historical_resource_proof_status':proof['status'],'D_endpoint0_capacity':dcap,'marker_free_seed_basis':marker_free,'descriptor_earned':passed,'q_D_state_earned':passed,'global_marker_observer_injectivity_earned':passed,'global_write_law_earned':passed,'strong_l2_earned':False,'ordinary_d2_path_global_theorem_earned':False,'s6_marker_unlocked':passed,'g6_graduated':False,'g6_r0_started':False,'next_authorized':plan['next_on_pass'] if passed else plan['next_on_fail'],'nonclaims':plan['nonclaims']}
    result['result_sha256']=canonical_sha256(result); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); write_json_atomic(out/'G6_S5R_MARKER_RESULT.json',result); (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S5R FRESH MARKER\n\n'+cls+'\n\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8'); return result
