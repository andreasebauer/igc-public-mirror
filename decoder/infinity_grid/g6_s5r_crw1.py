from __future__ import annotations

"""Registered G6:S5R CRW1 observer-state representation and exact write-law test."""

import json, zipfile, hashlib
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .g6_stage_executors import _basis
from .g6_s5r_compositional_read_write import _load_s1, _load_higher, _tree_record
from .adapters.g4_accepted import G4AcceptedAdapter
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

STAGE_ID='G6:S5R-CRW1'
PLAN_SCHEMA='IG_G6_S5R_CRW1_OBSERVER_STATE_WRITE_LAW_PLAN_V1'
RESULT_SCHEMA='IG_G6_S5R_CRW1_OBSERVER_STATE_WRITE_LAW_RESULT_V1'
EVALUATOR_REF='infinity_grid.g6_s5r_crw_evaluators:observer_state_write_law_evaluator'

class G6CRW1Error(RuntimeError):
    pass

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def _member(path:Path,suffix:str)->dict[str,Any]:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1:
            raise G6CRW1Error('CRW1_MEMBER_'+suffix)
        return json.loads(z.read(names[0]))

def _task(tid:str,left:dict[str,Any],right:dict[str,Any],op:tuple[int,int],*,expected_child_n=None)->TaskSpec:
    payload={
        'left_tree':left,'right_tree':right,'operator':list(op),
        'observer_probe_ref':'D2_PATH','observer_operator':[0,0],
    }
    if expected_child_n is not None:
        payload['expected_child_n']=int(expected_child_n)
    return TaskSpec(
        task_id=tid,task_kind='G6_S5R_CRW1_OBSERVER_STATE_WRITE',
        binding_sha256=canonical_sha256({
            'id':tid,'operator':list(op),'left':left,'right':right,
            'expected_child_n':expected_child_n,
        }),
        payload=payload,
        cost_weight=max(1.0,float(left.get('n',1)+right.get('n',1))),
    )

def run_crw1(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,
             accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s5r-crw1')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    required={
        'schema_id','stage_id','candidate','certified_domain','validation',
        'negative_regressions','dependency_sha256','registered_outcomes',
        'next_on_pass','next_on_fail','nonclaims','question_sha256',
    }
    if type(plan) is not dict or set(plan)!=required or plan['schema_id']!=PLAN_SCHEMA or plan['stage_id']!=STAGE_ID:
        raise G6CRW1Error('CRW1_PLAN')
    base={k:v for k,v in plan.items() if k!='question_sha256'}
    if canonical_sha256(base)!=plan['question_sha256']:
        raise G6CRW1Error('CRW1_PLAN_HASH')
    for logical,expected in plan['dependency_sha256'].items():
        if logical not in paths or _sha_file(paths[logical])!=expected:
            raise G6CRW1Error('CRW1_DEP_'+logical)

    crw0=json.loads(paths['crw0_result'].read_text(encoding='utf-8'))
    if crw0.get('classification')!='PASS_OBSERVER_INVERSION_ON_DECLARED_SCOPE_REPRESENTATION_AMENDMENT_REVIEW_NEXT' or crw0.get('decode_failure_count')!=0:
        raise G6CRW1Error('CRW1_CRW0_BINDING')
    ph=crw0.get('phases',[])
    if (len(ph)!=2 or ph[0]['science'].get('task_count')!=4524 or
        ph[0]['science'].get('class_count')!=4524 or ph[0]['science'].get('max_class_size')!=1 or
        ph[1]['science'].get('task_count')!=26594 or ph[1]['science'].get('class_count')!=26594 or
        ph[1]['science'].get('max_class_size')!=1):
        raise G6CRW1Error('CRW1_CRW0_SCOPE')

    reg=json.loads(paths['amendment_registered_input'].read_text(encoding='utf-8'))
    if reg.get('schema_id')!='IG_G6_S5R_COMPOSITIONAL_READ_WRITE_AMENDMENT_REGISTERED_INPUT_V1' or reg.get('promotion') is not False:
        raise G6CRW1Error('CRW1_AMENDMENT')
    if 'raw hidden topology' not in reg.get('feature_rule',''):
        raise G6CRW1Error('CRW1_FEATURE_RULE')

    s4=_member(paths['s4r_closeout'],'G6_S4R_RESULT.json')
    if (s4.get('gates',{}).get('full_certified_union_L1_injective') is not True or
        s4.get('finite_write_law_earned') is not False or
        s4.get('certified_scope',{}).get('union_exact_state_count')!=31114):
        raise G6CRW1Error('CRW1_S4')

    g5=paths['g5_parent_review'].read_text(encoding='utf-8')
    g5_lower=g5.lower()
    if 'exact hidden relation-valued grafting law' not in g5_lower or 'global single-c/(0,0)-probe injectivity' not in g5_lower:
        raise G6CRW1Error('CRW1_G5_PARENT')

    audit=paths['audit'].read_text(encoding='utf-8')
    for token in ('6 distinct exact child classes','12 distinct exact child classes','16 versus 16','40 versus 40'):
        if token not in audit:
            raise G6CRW1Error('CRW1_NEGATIVE_'+token.replace(' ','_'))
    s1r=_member(paths['s1r_closeout'],'G6_S1_REPAIR_CERTIFIED_CLOSEOUT_2026-09-07.json')
    c3=s1r['candidate_results']['C3_ROOTED_OWNER_RESPONSE_BAG']
    if s1r.get('promotion') is not False or c3.get('expanded_status')!='PASS':
        raise G6CRW1Error('CRW1_NEG_C3')

    basis=_basis()
    s1=sorted(_load_s1(paths['s2r_closeout']),key=lambda r:r['id'])
    higher=sorted(_load_higher(paths['e3_recovered_fixture']),key=lambda r:r['id'])
    ops=tuple(G4AcceptedAdapter().operator_basis())
    contexts=[(p,tuple(op),pos) for p in sorted(basis) for op in ops for pos in ('LEFT','RIGHT')]
    if len(contexts)!=248:
        raise G6CRW1Error('CRW1_CONTEXT_COUNT')

    tasks=[]
    for label,rows in (('S1',s1[:248]),('H',higher[:248])):
        for i,(row,ctx) in enumerate(zip(rows,contexts)):
            pref,op,pos=ctx
            st=row['tree']; bt=_tree_record(basis[pref])
            left,right=(st,bt) if pos=='LEFT' else (bt,st)
            tasks.append(_task(f'{label}-CTX-{i:03d}-{row["id"]}',left,right,op))
    p0=runtime.run_structural_partition(
        phase_id='CRW1_CERTIFIED_WRITE_PANEL',tasks=tasks,evaluator_ref=EVALUATOR_REF,
        requested_workers=4,max_tasks=496,
    )
    m0=p0.execution_metadata.get('worker_metric_totals',{})

    n21=[r for r in higher if int(r['tree'].get('n',0))==21]
    if len(n21)<128:
        raise G6CRW1Error('CRW1_N21_PANEL')
    probe=_tree_record(basis['D2_PATH'])
    growth=[
        _task(f'GROW-{i:03d}-{r["id"]}',r['tree'],probe,(0,0),expected_child_n=26)
        for i,r in enumerate(n21[:128])
    ]
    p1=runtime.run_structural_partition(
        phase_id='CRW1_OUT_OF_U_WRITE_PANEL',tasks=growth,evaluator_ref=EVALUATOR_REF,
        requested_workers=4,max_tasks=128,
    )
    m1=p1.execution_metadata.get('worker_metric_totals',{})

    write_mismatches=int(m0.get('write_mismatch',0))+int(m1.get('write_mismatch',0))
    decode_failures=int(m0.get('parent_decode_failure',0))+int(m1.get('parent_decode_failure',0))
    size_checks=int(m1.get('out_of_u_size_check',0))
    passed=(write_mismatches==0 and decode_failures==0 and size_checks==128)
    classification=('PASS_S5_OBSERVER_STATE_AND_WRITE_LAW_EARNED_ON_CERTIFIED_SCOPE' if passed else 'WRITE_CONFLICT')

    result={
        'schema_id':RESULT_SCHEMA,
        'status':'PASS' if passed else 'REVIEW_REQUIRED',
        'stage_id':STAGE_ID,
        'classification':classification,
        'accepted_decoder_source_sha256':accepted_source_sha256,
        'internal_execution_id':internal_execution_id,
        'question_sha256':plan['question_sha256'],
        'candidate':plan['candidate'],
        'crw0_result_sha256':crw0['result_sha256'],
        'coarsest_observer_preserving_state_earned':bool(passed),
        'descriptor_earned':bool(passed),
        'finite_read_write_description_earned':bool(passed),
        'write_law_earned':bool(passed),
        'proof_scope':(
            'All S0-S4 certified contexts whose parents are in S0 union U. CRW0 exact inversion on all 31,118 parent states plus deterministic exact grafting proves W equals raw composition projected through q throughout that domain; the 496-task panel validates the q-only implementation on every registered context type.'
        ),
        'validation_phases':[
            {'id':'CRW1_CERTIFIED_WRITE_PANEL','science':p0.summary,'execution':p0.execution_metadata},
            {'id':'CRW1_OUT_OF_U_WRITE_PANEL','science':p1.summary,'execution':p1.execution_metadata},
        ],
        'write_mismatch_count':write_mismatches,
        'parent_decode_failure_count':decode_failures,
        'out_of_U_growth_checks_passed':size_checks,
        'negative_regressions_retained':True,
        's6_unlocked_for_preregistration':bool(passed),
        'recursive_closure_all_finite_terms_earned':False,
        'global_single_probe_theorem_earned':False,
        'g6_graduated':False,
        'g6_r0_started':False,
        'next_authorized':plan['next_on_pass'] if passed else plan['next_on_fail'],
        'nonclaims':plan['nonclaims'],
        'scientific_interpretation':(
            'The already-certified exact observer itself is now an earned coarsest read state on the S0-S4 domain, with an exact relation-valued write algorithm that consumes only observer states. Global recursive closure remains the S6 question.'
            if passed else
            'The observer state did not sustain the registered q-only write comparison; no S5 descriptor is promoted.'
        ),
    }
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    write_json_atomic(out/'G6_S5R_CRW1_RESULT.json',result)
    (out/'READ_FIRST.txt').write_text(
        'INFINITY GRID — G6:S5R CRW1\n\n'+classification+'\n\nResult SHA-256: '+result['result_sha256']+'\n',
        encoding='utf-8',
    )
    return result
