from __future__ import annotations

"""G6:S5R observer-continuity repair evaluator.

This module does not choose or execute arbitrary code.  The controller invokes a
fixed registered operation and supplies content-addressed evidence artifacts.
The operation binds the frozen observer, rejects known-insufficient candidate
families, and records the next admissible scientific search.  It does not
promote a descriptor, reopen hidden topology, graduate G6, or start R0.
"""

import hashlib, json, zipfile
from pathlib import Path
from typing import Any, Mapping
from .canon import canonical_sha256, write_json_atomic
from .v05_origin_guard import require_controller_execution_origin

PLAN_SCHEMA='IG_G6_S5R_OBSERVER_CONTINUITY_REPAIR_REGISTRATION_V1'
RESULT_SCHEMA='IG_G6_S5R_OBSERVER_CONTINUITY_REPAIR_RESULT_V1'
EXPECTED_LOGICAL_NAMES={
    'science_plan','master_prereg','g5_parent_review','s0_s1_closeout','s1r_closeout',
    's2r_closeout','s3r_closeout','s4r_closeout','s5_closeout','e3_recovered_fixture','audit'
}

class G6S5RRepairError(RuntimeError): pass

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _json_member(zp:Path,name:str)->dict[str,Any]:
    with zipfile.ZipFile(zp) as z:
        if name not in z.namelist(): raise G6S5RRepairError('EVIDENCE_MEMBER_MISSING:'+name)
        obj=json.loads(z.read(name))
    if type(obj) is not dict: raise G6S5RRepairError('EVIDENCE_MEMBER_SCHEMA:'+name)
    return obj

def _text_member(zp:Path,name:str)->str:
    with zipfile.ZipFile(zp) as z:
        if name not in z.namelist(): raise G6S5RRepairError('EVIDENCE_MEMBER_MISSING:'+name)
        return z.read(name).decode('utf-8')

def _verify_plan(plan:dict[str,Any],artifacts:Mapping[str,Path])->None:
    required={'schema_id','repair_id','leaf_universe','frozen_observer','candidate_acceptance_requirement','evidence_bindings','required_dispositions','next_on_pass','nonclaims'}
    if set(plan)!=required or plan.get('schema_id')!=PLAN_SCHEMA: raise G6S5RRepairError('S5R_PLAN_SCHEMA')
    if set(artifacts)!=EXPECTED_LOGICAL_NAMES: raise G6S5RRepairError('S5R_ARTIFACT_SET')
    binds=plan['evidence_bindings']
    if type(binds) is not dict or set(binds)!=(EXPECTED_LOGICAL_NAMES-{'science_plan'}): raise G6S5RRepairError('S5R_EVIDENCE_BINDINGS')
    for logical,sha in binds.items():
        if type(sha) is not str or len(sha)!=64 or _sha_file(artifacts[logical])!=sha: raise G6S5RRepairError('S5R_EVIDENCE_SHA:'+logical)
    leaf=plan['leaf_universe']
    if leaf!={'kind':'FOUR_FROZEN_S0_SEEDS','seed_refs':['D2_PATH','D2_BROOM','D4_PATH','D4_BROOM']}: raise G6S5RRepairError('S5R_LEAF_UNIVERSE')
    obs=plan['frozen_observer']
    expected={'probe_ref':'D2_PATH','operator':[0,0],'position':'LEFT','semantics':'complete exact relation signature under structural unrooted canon'}
    if obs!=expected: raise G6S5RRepairError('S5R_OBSERVER')
    req=plan['candidate_acceptance_requirement']
    if req.get('factorization')!='q(x)=q(y) => O(x)=O(y)' or req.get('decoder_form')!='exists d with O(x)=d(q(x))': raise G6S5RRepairError('S5R_ACCEPTANCE_REQUIREMENT')

def run_observer_continuity_repair(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,accepted_source_sha256:str,internal_execution_id:str)->dict[str,Any]:
    require_controller_execution_origin('g6-s5r-observer-continuity')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    _verify_plan(plan,paths)

    master=paths['master_prereg'].read_text(encoding='utf-8')
    g5=paths['g5_parent_review'].read_text(encoding='utf-8')
    audit=paths['audit'].read_text(encoding='utf-8')
    master_ok=('finite bags/counts' in master and 'finite deterministic read labels' in master and 'raw hidden tree topology/canon' in master)
    g5_ok=('D_out' in g5 and 'f + g' in g5 and 'e_a' in g5 and 'e_b' in g5)
    audit_ok=('main scientific discontinuity is at S5' in audit and 'q(X)=q(Y)' in audit and '31,114' in audit)

    s1=_json_member(paths['s0_s1_closeout'],'chain/stage_commits/G6__S1.json')
    rows=s1['result']['rows']
    def row(left,right):
        found=[r for r in rows if r.get('left')==left and r.get('right')==right and r.get('operator')==[0,0]]
        if len(found)!=1: raise G6S5RRepairError('S1_WITNESS_ROW')
        return found[0]
    pp=row('D2_PATH','D2_PATH'); pb=row('D2_PATH','D2_BROOM')
    s1_witness_ok=(pp['exact_outcome_count']==6 and pb['exact_outcome_count']==12 and pp['input_public_left']==pb['input_public_left'] and pp['input_public_right']==pb['input_public_right'] and pp['public_outcomes']==pb['public_outcomes'])

    s1r=_json_member(paths['s1r_closeout'],'g6_s1_repair_closeout/G6_S1_REPAIR_CERTIFIED_CLOSEOUT_2026-09-07.json')
    c1=s1r['candidate_results']['C1_RHO01']['first_conflict']; c2=s1r['candidate_results']['C2_TYPED_OWNER_ORBITS']['first_conflict']
    counts_only_witness_ok=(c1['outcome_count_a']==c1['outcome_count_b']==16 and c1['representative_a_sha256']!=c1['representative_b_sha256'] and c2['outcome_count_a']==c2['outcome_count_b']==40 and c2['representative_a_sha256']!=c2['representative_b_sha256'])

    s2r=_json_member(paths['s2r_closeout'],'g6_s2r_closeout/G6_S2_REPAIRED_CERTIFIED_CLOSEOUT_2026-09-07.json')
    s2_ok=(s2r['result']['universe_count']==4520 and s2r['result']['behavior_class_count']==4520 and s2r['result']['max_class_size']==1 and s2r['result']['frozen_separating_context']=={'operator':[0,0],'probe_ref':'D2_PATH','tested_parent_position':'LEFT'})

    s3r=_json_member(paths['s3r_closeout'],'G6_S3R_SCIENTIFIC_RECOVERY_CLOSEOUT_2026-09-07/G6_S3R_SCIENTIFIC_RECOVERY_CLOSEOUT.json')
    s3_ok=(s3r['fresh_higher_fixture']['state_count']==26594 and s3r['fresh_higher_fixture']['overlap_with_s1_count']==0 and s3r['primary_l1']['class_count']==26594 and s3r['primary_l1']['max_class_size']==1 and s3r['mandatory_independent_gate']['context']=={'operator':[0,0],'position':'LEFT','probe_ref':'D2_PATH'})

    s4r=_json_member(paths['s4r_closeout'],'G6_S4R_CERTIFIED_CLOSEOUT_2026-09-07/G6_S4R_RESULT.json')
    s4_ok=(s4r['certified_scope']['union_exact_state_count']==31114 and s4r['gates']['full_certified_union_L1_injective'] is True and s4r['finite_write_law_earned'] is False and s4r['registered_observer']==plan['frozen_observer'])
    s4_not_transition_closed=(21 in s4r['proof']['cross_image_argument']['higher_parent_n_set'] and s4r['finite_write_law_earned'] is False)

    s5=_json_member(paths['s5_closeout'],'G6_S5_CERTIFIED_CLOSEOUT_2026-09-07/G6_S5_RESULT.json')
    unit=_json_member(paths['s5_closeout'],'G6_S5_CERTIFIED_CLOSEOUT_2026-09-07/CANDIDATE_UNIT_STATE_LAW.json')
    old_s5_ok=(unit['state_cardinality_on_frozen_scope']==1 and unit['state_name']=='UNIT' and s5['feature_search']['family']=='2^11 coordinate projections of D=(f,m)')

    e3_read=_text_member(paths['e3_recovered_fixture'],'e3_fixture_checkpoint/READ_FIRST.txt')
    with zipfile.ZipFile(paths['e3_recovered_fixture']) as z: e3_names=set(z.namelist())
    e3_ok=('e3_fixture_checkpoint/fixture/higher_states.jsonl.gz' in e3_names and '"higher_distinct_states": 26594' in e3_read and 'a47855103369a8e329976094db4a21ac1ccdcc49798e4fbbd7adad49d6133306' in e3_read)

    checks={
      'master_feature_grammar_bound':master_ok,'g5_parent_law_bound':g5_ok,'audit_discontinuity_bound':audit_ok,
      's1_six_vs_twelve_exact_relation_witness':s1_witness_ok,'s1r_equal_count_unequal_set_witnesses':counts_only_witness_ok,
      's2r_4520_injective_observer':s2_ok,'s3r_26594_injective_higher_observer':s3_ok,'s4r_union_31114_injective':s4_ok,
      's4r_no_finite_write_law_earned':s4_not_transition_closed,'old_s5_unit_and_d_projection_family_bound':old_s5_ok,
      'recovered_higher_fixture_available_for_next_registered_search':e3_ok,
    }
    if not all(checks.values()): raise G6S5RRepairError('S5R_EVIDENCE_CHECK:'+','.join(k for k,v in checks.items() if not v))

    result={
      'schema_id':RESULT_SCHEMA,
      'status':'PASS',
      'stage_id':'G6:S5R-OBSERVER-CONTINUITY',
      'repair_id':plan['repair_id'],
      'accepted_decoder_source_sha256':accepted_source_sha256,
      'internal_execution_id':internal_execution_id,
      'leaf_universe':plan['leaf_universe'],
      'frozen_observer':plan['frozen_observer'],
      'candidate_acceptance_requirement':plan['candidate_acceptance_requirement'],
      'evidence_checks':checks,
      'candidate_dispositions':{
        'UNIT':{'status':'NON_ADMITTED','reason':'same q value cannot reconstruct the recorded 6-class and 12-class exact relations under the same operator'},
        'D_ONLY_COORDINATE_PROJECTIONS':{'status':'NON_ADMITTED_FAMILY','reason':'D2_PATH and D2_BROOM share the full inherited public D while the frozen exact observer distinguishes them; every function of D alone merges this witness'},
        'OUTCOME_COUNT_ONLY':{'status':'NON_ADMITTED_FAMILY','reason':'recorded S1R conflicts have equal counts 16=16 and 40=40 but unequal exact outcome sets'},
      },
      'information_lower_bound_on_certified_U':{'exact_state_count':31114,'minimum_distinct_q_values':31114,'basis':'O is injective on U and candidate acceptance requires q(x)=q(y) => O(x)=O(y)'},
      's4_scope_disposition':{'representative_independence_on_U':'RETAIN','transition_closed_carrier':'NOT_EARNED','finite_write_law':'NOT_EARNED'},
      'feature_grammar_disposition':{'d_coordinate_family':'EXHAUSTED_BY_LOGICAL_WITNESS_NO_2048_RERUN','counts_only':'INSUFFICIENT','certified_observer_derived_materialized_labels':'AUTHORIZED_NEXT_FAMILY','raw_hidden_topology':'NOT_AUTHORIZED'},
      'historical_s5_unit_disposition':'RETAIN_ONLY_AS_WEAKER_AVAILABILITY_OBSERVER_RESULT',
      'classification':'OBSERVER_CONTINUITY_REPAIR_PASS_WIDER_FEATURE_SEARCH_REQUIRED',
      'next_authorized':plan['next_on_pass'],
      'descriptor_earned':False,
      's6_exact_observer_closure_unlocked':False,
      'g6_graduated':False,
      'g6_r0_started':False,
      'authority_effect':'CONTROLLER_REGISTERED_REPAIR_EVIDENCE_ONLY_NO_STAGE_PROMOTION',
      'nonclaims':plan['nonclaims'],
    }
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    jp=out/'G6_S5R_OBSERVER_CONTINUITY_RESULT.json'; write_json_atomic(jp,result); jp.chmod(0o444)
    read_first="""INFINITY GRID — G6:S5R OBSERVER-CONTINUITY REPAIR\n\nSTATUS: PASS / WIDER FEATURE SEARCH REQUIRED / NO S5 DESCRIPTOR PROMOTED\n\nThe frozen exact observer is retained. UNIT is non-admitted by the stored six-versus-twelve witness. D-only projections are non-admitted as a family because the witness already agrees on the full inherited D. Counts-only candidates are non-admitted by the stored 16=16 and 40=40 unequal-set witnesses.\n\nOn certified U, O is injective on 31,114 exact states, so every O-preserving q must have at least 31,114 distinct values there. This is an information lower bound, not a claim that no finite recursive description exists. S4 representative independence is retained only on U; transition closure and a finite write law remain unearned.\n\nNEXT: registered wider-feature search using certified observer-derived materialized labels and the recovered fixtures. Raw hidden topology remains outside the authorized grammar. G6 is not graduated; S6 exact-observer closure is not unlocked; R0 is not started.\n\nRESULT SHA-256: """+result['result_sha256']+'\n'
    tp=out/'READ_FIRST.txt'; tp.write_text(read_first,encoding='utf-8'); tp.chmod(0o444)
    return result
