"""Native STAGE qualification for saved marker services and theorem eligibility."""
import hashlib
import json
from pathlib import Path
import time

from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from infinity_grid.theorem_registry import verify_all_theorems
from infinity_grid.semantic_sentinel import verify_native_semantics

REFS = {
    'q':'infinity_grid.g6_saved_marker_service_evaluators:saved_marker_q_evaluator',
    'cold':'infinity_grid.g6_saved_marker_service_evaluators:saved_marker_cold_decode_evaluator',
    'write':'infinity_grid.g6_saved_marker_service_evaluators:saved_marker_write_evaluator',
}

def write(root, name, data):
    (root/name).write_text(json.dumps(data,sort_keys=True,indent=2)+'\n')

def handler(stage, runtime):
    started = time.monotonic()
    root = runtime.runtime_root.parent.parent
    paths = stage['input_artifacts'];params = stage['execution']['parameters']
    for name,sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(paths[name]).read_bytes()).hexdigest() == sha
    cases = json.loads(Path(paths['cases']).read_text())
    prior = json.loads(Path(paths['marker_r2_result']).read_text())
    consumer = json.loads(Path(paths['v2_consumer_result']).read_text())
    assert prior['status']=='PASS' and prior['independent_marker_records']==20
    assert consumer['scientific_outcome']=='FINITE_SAVED_V2_CONSUMER_QUALIFIED_NO_PRODUCTION_AUTHORITY'
    assert consumer['packet_matches']==10 and consumer['loaded_root_occurrences']==747
    baseline = verify_native_semantics(raise_on_change=False)
    theorems = verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    write(root,'THEOREM_ELIGIBILITY_RESULT.json',{
        'status':'REVIEW_RETAINED','theorem_verifications':theorems,
        'finite_V2_observer_evidence_available':True,'production_admission':False,
        'baseline_adopted':False,'theorems_adopted':False,
        'remaining_premises':{
            'generic_factorisation':'Intended-domain parent congruence, leaf/reservation transport and complete Counter/action closure.',
            'organization_service':'Finite V2 recipe/panel/consumer evidence is available; it supplies no universal future-tree or transition-Counter conclusion.',
            'exact_read':'Current R100 population, intended-domain Counter correspondence and exact-read obligations remain separate.'}})
    rows=cases['cases'];assert len(rows)==20 and len({r['occurrence'] for r in rows})==20
    phases=[];all_signatures=[]
    for mode in ('q','cold','write'):
        payloads=rows if mode!='write' else [cases['write_case']]
        tasks=[TaskSpec(task_id=mode+'-'+str(i).zfill(3),task_kind='SAVED_MARKER_SERVICE_'+mode.upper(),
                       binding_sha256=canonical_sha256(payload),payload=payload,
                       cost_weight=float(payload.get('tree',payload.get('left',{'n':1}))['n']))
               for i,payload in enumerate(payloads)]
        phase='SAVED_MARKER_'+mode.upper()
        result=runtime.run_structural_partition(phase_id=phase,tasks=tasks,evaluator_ref=REFS[mode],requested_workers=params['workers'])
        signatures=list(runtime.iter_partition_signatures(phase))
        assert len(signatures)==len(tasks) and all(r['signature'][0]=='PASS' for r in signatures)
        metrics=result.execution_metadata['worker_metric_totals']
        if mode=='q':assert metrics['marker_q_matches']==metrics['warm_decode_matches']==20
        if mode=='cold':assert metrics['cold_decode_matches']==20
        if mode=='write':assert metrics['marker_write_matches']==1 and metrics['written_child_decode_matches']==18
        assert metrics['forbidden_service_refusals']==2*len(tasks)
        phases.append({'mode':mode,'summary':result.summary,'execution':result.execution_metadata})
        all_signatures.append({'mode':mode,'rows':signatures})
        write(root,'SERVICE_PROGRESS.json',{'completed_modes':[p['mode'] for p in phases]})
    assert verify_native_semantics(raise_on_change=False)==baseline
    assert verify_all_theorems(raise_on_stale=False)==theorems
    write(root,'SERVICE_SIGNATURES.json',{'status':'PASS','phases':all_signatures,'task_count':41})
    write(root,'SERVICE_EXECUTION.json',{'status':'PASS','phases':phases,'runtime_owner':'NATIVE_STAGE_SCIENCE_RUNTIME','private_pools':False})
    result={'schema_id':'IG_SAVED_MARKER_KERNELVIEW_QUALIFICATION_V1','status':'PASS',
        'outcome':'SAVED_MARKER_RESTRICTED_SERVICE_ROUTE_QUALIFIED_FOR_DECLARED_SCOPE',
        'marker_q_cases':20,'warm_decode_cases':20,'cold_decode_cases':20,
        'full_write_cases':1,'write_retained_children':18,'written_child_decodes':18,
        'forbidden_service_refusals':82,'native_tasks':41,'independent_reference_records_reused':20,
        'independent_oracle_replays':0,'old_marker_job_replayed':False,
        'production_generation_allowed':False,'current_R100_admitted':False,
        'semantic_status':'REOPEN_REQUIRED','stale_theorems':4,'baseline_adopted':False,
        'theorems_adopted':False,'wall_seconds':time.monotonic()-started,
        'scope':'Saved R2 finite contexts only; actual registered KernelView services. No global marker or master graduation.'}
    write(root,'QUALIFICATION_RESULT.json',result)
    return ChainExecutionResult(result)
