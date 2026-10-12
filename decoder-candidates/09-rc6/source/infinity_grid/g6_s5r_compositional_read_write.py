from __future__ import annotations

"""Registered G6:S5R compositional read/write investigation, CRW0.

CRW0 does not promote topology or an S5 descriptor.  It tests a prerequisite for
an observer-derived exact write law: whether the frozen exact L1 observer can be
mechanically inverted on the preregistered certified scope using only its child
relation and the fixed probe.
"""
import gzip, io, json, zipfile
from pathlib import Path
from typing import Any, Mapping

from .canon import canonical_sha256, write_json_atomic
from .g6_stage_executors import _basis
from .v05_origin_guard import require_controller_execution_origin
from .v05_stage_runtime import TaskSpec

STAGE_ID='G6:S5R-CRW0'
PLAN_SCHEMA='IG_G6_S5R_CRW0_OBSERVER_INVERSION_PLAN_V1'
RESULT_SCHEMA='IG_G6_S5R_CRW0_OBSERVER_INVERSION_RESULT_V1'
EVALUATOR_REF='infinity_grid.g6_s5r_crw_evaluators:observer_inversion_descriptor_evaluator'

class G6CRWError(RuntimeError): pass

def _sha_file(p:Path)->str:
    import hashlib
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def _member(path:Path,suffix:str)->dict[str,Any]:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith(suffix)]
        if len(names)!=1: raise G6CRWError('CRW_MEMBER_'+suffix)
        return json.loads(z.read(names[0]))

def _s1_identifier(r:Mapping[str,Any])->str:
    for key in ('state_id','exact_canon_sha256','id'):
        value=r.get(key)
        if value:
            return str(value)
    exact_repr=r.get('exact_canon_repr')
    if type(exact_repr) is str and exact_repr:
        return canonical_sha256({'exact_canon_repr':exact_repr})
    raise G6CRWError('CRW_S1_IDENTIFIER')

def _load_s1(path:Path)->list[dict[str,Any]]:
    d=_member(path,'S1_CHILD_UNIVERSE_INDEX.json'); rows=d['states'] if isinstance(d,dict) else d
    if len(rows)!=4520: raise G6CRWError('CRW_S1_COUNT')
    out=[]; seen=set()
    for r in rows:
        tree=r.get('tree') or r.get('tree_record') or r.get('state_tree')
        sid=_s1_identifier(r)
        if type(tree) is not dict: raise G6CRWError('CRW_S1_TREE')
        task_id='S1-'+sid
        if task_id in seen: raise G6CRWError('CRW_S1_DUPLICATE_IDENTIFIER')
        seen.add(task_id); out.append({'id':task_id,'tree':tree})
    return out

def _load_higher(path:Path)->list[dict[str,Any]]:
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('higher_states.jsonl.gz')]
        if len(names)!=1: raise G6CRWError('CRW_HIGHER_MEMBER')
        raw=z.read(names[0])
    out=[]
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as g:
        for line in g:
            r=json.loads(line); tree=r.get('tree'); sid=str(r.get('state_id',''))
            if type(tree) is not dict: raise G6CRWError('CRW_HIGHER_TREE')
            out.append({'id':'H-'+sid,'tree':tree})
    if len(out)!=26594: raise G6CRWError('CRW_HIGHER_COUNT')
    return out

def _tree_record(t):
    return {'n':int(t.n),'edges':[list(e) for e in t.edges],'H_classes':list(t.H_classes),'edge_operators':[list(o) for o in t.edge_operators]}

def _task(row:dict[str,Any],observer:dict[str,Any])->TaskSpec:
    payload={'state_tree':row['tree'],'probe_ref':observer['probe_ref'],'operator':observer['operator']}
    return TaskSpec(task_id=row['id'],task_kind='G6_S5R_CRW0_OBSERVER_INVERSION',binding_sha256=canonical_sha256({'id':row['id'],'observer':observer}),payload=payload,cost_weight=max(1.0,float(row['tree'].get('n',1))))

def run_compositional_read_write_crw0(*,plan_path:str|Path,artifacts:Mapping[str,str|Path],output_dir:str|Path,
                                      accepted_source_sha256:str,internal_execution_id:str,runtime)->dict[str,Any]:
    require_controller_execution_origin('g6-s5r-crw0')
    paths={k:Path(v).resolve(strict=True) for k,v in artifacts.items()}
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    required={'schema_id','stage_id','question_sha256','scope','frozen_observer','dependency_sha256','acceptance','nonclaims','next_on_pass','next_on_fail'}
    if type(plan) is not dict or set(plan)!=required or plan['schema_id']!=PLAN_SCHEMA or plan['stage_id']!=STAGE_ID: raise G6CRWError('CRW_PLAN')
    q=canonical_sha256({k:v for k,v in plan.items() if k!='question_sha256'})
    if q!=plan['question_sha256']: raise G6CRWError('CRW_PLAN_HASH')
    for logical,expected in plan['dependency_sha256'].items():
        if logical not in paths or _sha_file(paths[logical])!=expected: raise G6CRWError('CRW_DEP_'+logical)
    master=paths['master_prereg'].read_text(encoding='utf-8')
    if not all(x in master for x in ('finite deterministic read labels','raw hidden tree topology/canon','owner identity')): raise G6CRWError('CRW_GRAMMAR')
    s1r=_member(paths['s1r_closeout'],'G6_S1_REPAIR_CERTIFIED_CLOSEOUT_2026-09-07.json')
    c3=s1r['candidate_results']['C3_ROOTED_OWNER_RESPONSE_BAG']
    if s1r.get('promotion') is not False or c3['expanded_status']!='PASS' or c3['refined_classes']!=4520: raise G6CRWError('CRW_S1R_BINDING')
    s4=_member(paths['s4r_closeout'],'G6_S4R_RESULT.json')
    if s4['certified_scope']['union_exact_state_count']!=31114 or s4['finite_write_law_earned'] is not False or s4['gates']['full_certified_union_L1_injective'] is not True: raise G6CRWError('CRW_S4_BINDING')
    wider=json.loads(paths['wider_result'].read_text(encoding='utf-8'))
    if wider['classification']!='PASS_NO_COMPLETE_S5_READ_WRITE_DESCRIPTOR_IN_CURRENT_MATERIALIZED_FEATURE_SET' or wider['descriptor_earned'] is not False: raise G6CRWError('CRW_WIDER_BINDING')
    amendment=json.loads(paths['amendment_spec'].read_text(encoding='utf-8'))
    if amendment.get('stage_id')!='G6:S5R' or amendment.get('promotion') is not False: raise G6CRWError('CRW_AMENDMENT_BINDING')

    obs=plan['frozen_observer']; basis=_basis()
    seeds=[{'id':'S0-'+k,'tree':_tree_record(v)} for k,v in sorted(basis.items())]
    s1=_load_s1(paths['s2r_closeout'])
    phases=[]
    p0=runtime.run_structural_partition(phase_id='CRW0_S0_S1_INVERSION',tasks=[_task(x,obs) for x in seeds+s1],evaluator_ref=EVALUATOR_REF,requested_workers=4,max_tasks=4524)
    failures=int(p0.execution_metadata.get('worker_metric_totals',{}).get('decode_failure',0))
    phases.append({'scope':'S0+S1','science':p0.summary,'execution':p0.execution_metadata,'decode_failures':failures})
    if failures==0 and plan['scope']=='FULL_CERTIFIED_U':
        higher=_load_higher(paths['e3_recovered_fixture'])
        p1=runtime.run_structural_partition(phase_id='CRW0_HIGHER_INVERSION',tasks=[_task(x,obs) for x in higher],evaluator_ref=EVALUATOR_REF,requested_workers=4,max_tasks=26594)
        f1=int(p1.execution_metadata.get('worker_metric_totals',{}).get('decode_failure',0))
        phases.append({'scope':'HIGHER','science':p1.summary,'execution':p1.execution_metadata,'decode_failures':f1}); failures+=f1
    passed=failures==0
    if passed:
        classification='PASS_OBSERVER_INVERSION_ON_DECLARED_SCOPE_REPRESENTATION_AMENDMENT_REVIEW_NEXT'
        next_auth=plan['next_on_pass']
    else:
        classification='OBSERVER_INVERSION_FALSIFIED_ON_DECLARED_SCOPE'
        next_auth=plan['next_on_fail']
    result={
      'schema_id':RESULT_SCHEMA,'status':'PASS','stage_id':STAGE_ID,'accepted_decoder_source_sha256':accepted_source_sha256,
      'internal_execution_id':internal_execution_id,'question_sha256':plan['question_sha256'],'scope':plan['scope'],'frozen_observer':obs,
      'phases':phases,'decode_failure_count':failures,'classification':classification,
      'descriptor_earned':False,'representation_amendment_earned':False,'s6_unlocked':False,'g6_graduated':False,'g6_r0_started':False,
      'scientific_interpretation':('The frozen observer is mechanically invertible on the tested scope using only its complete child relation and the fixed probe. This supports, but does not yet prove globally, an observer-derived exact recursive representation. No topology-like representation is promoted in CRW0.' if passed else 'The proposed observer inversion is not sufficient on the tested scope; no representation is promoted.'),
      'next_authorized':next_auth,'nonclaims':plan['nonclaims'],
    }
    result['result_sha256']=canonical_sha256(result)
    out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); write_json_atomic(out/'G6_S5R_CRW0_RESULT.json',result)
    (out/'READ_FIRST.txt').write_text('INFINITY GRID — G6:S5R CRW0\n\n'+classification+'\n\nResult SHA-256: '+result['result_sha256']+'\n',encoding='utf-8')
    return result
