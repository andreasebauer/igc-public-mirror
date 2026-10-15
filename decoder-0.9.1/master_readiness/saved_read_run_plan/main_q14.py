"""Native qualification of installed candidate RunPlan/session saved-read admission."""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile
from infinity_grid import saved_read_run_plan as sr, run_plan_v2 as rp
from infinity_grid.materialized_discovery import MaterializedDiscoverySession
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import canonical_sha256, write_json_atomic


def main():
    started=time.monotonic()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest()==sha
    binding=json.loads(Path(inputs['saved_binding']).read_text())
    calibration=json.loads(Path(inputs['calibration']).read_text())
    master=json.loads(Path(inputs['master_contract']).read_text())
    plan=json.loads(Path(inputs['plan']).read_text())
    assert calibration==sr.make_calibration(binding,inputs['q11_evidence'],master)
    compiled=rp.compile_run_plan_v2(plan,saved_read_calibration=calibration)
    assert len(compiled['requests'])==27 and compiled['required_theorems']==[]
    # Compare all pre-existing functions/methods, not only named sentinel helpers.
    compatibility={}
    for module in ('materialized_discovery','run_plan_v2'):
        old=Path(inputs['baseline_'+module]).read_text()
        new=Path(sr.__file__).with_name(module+'.py').read_text()
        def definitions(text):
            tree=ast.parse(text);out={}
            for node in tree.body:
                if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    out[node.name]=ast.dump(node,include_attributes=False)
                elif isinstance(node,ast.ClassDef):
                    for child in node.body:
                        if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):
                            out[node.name+'.'+child.name]=ast.dump(child,include_attributes=False)
            return out
        if module=='run_plan_v2':
            expected=old.replace("__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()",
                'datetime.datetime.now(datetime.timezone.utc).isoformat()')
            old_defs=definitions(old);expected_defs=definitions(expected);new_defs=definitions(new)
            clock_changed=[n for n in old_defs if old_defs[n]!=expected_defs[n]]
            assert len(clock_changed)==1 and new_defs[clock_changed[0]]==expected_defs[clock_changed[0]]
        else:clock_changed=[]
        before,after=definitions(old),definitions(new)
        changed=[n for n in before if n not in after or before[n]!=after[n]]
        assert set(changed)==(set(['compile_run_plan_v2']+clock_changed) if module=='run_plan_v2' else set())
        compatibility[module]={'unchanged_existing_definitions':len(before)-len(changed),
            'changed_existing_definitions':changed,'new_definitions':sorted(set(after)-set(before))}
    failures=[]
    calls=[]
    def trace(frame,event,arg):
        if event=='call':calls.append((str(frame.f_globals.get('__name__','')),frame.f_code.co_name))
    def refuses(name,fn):
        count=len(calls)
        try:fn()
        except Exception as exc:
            assert type(exc).__name__ in ('SavedReadPlanError','RunPlanV2Error','MaterializedDiscoveryError','SavedInputError')
            assert not any(n in ('_states_from_dag','_build_and_store','_scan_level') for m,n in calls[count:])
            failures.append({'case':name,'error':type(exc).__name__,'message':str(exc),'metadata_first':True})
        else:raise AssertionError('accepted '+name)
    assert sys.getprofile() is None
    sys.setprofile(trace)
    try:
        for field,value in [('production_authority','PRODUCTION'),('generation_allowed',True),
            ('stop_policy','ADVANCE_TO_100'),('calibration_sha256','0'*64),('route','LEGACY')]:
            p=deepcopy(plan);p[field]=value
            refuses(field,lambda p=p:rp.compile_run_plan_v2(p,saved_read_calibration=calibration))
        requests=[{'kind':'saved_packet','domain':'selected','level':100},
            {'kind':'saved_packet','domain':'selected','level':14},
            {'kind':'saved_packet','domain':'full_pool','level':7},
            {'kind':'saved_packet','domain':'selected','level':True},
            {'kind':'saved_packet','domain':'selected','level':'7'},
            {'kind':'complete_read','domain':calibration['counter_anchor_ids'][0],'level':100},
            {'kind':'complete_read','domain':'unbound','level':7},
            {'kind':'matched_findings','domain':'selected','level':8},
            {'kind':'matched_findings','domain':'FROZEN_MOTIFS','level':100},
            {'kind':'temporal_window','domain':'FROZEN_MOTIFS','level':13},
            {'kind':'temporal_window','domain':'FROZEN_MOTIFS','level':8},
            {'kind':'r100_descriptors','domain':'HISTORICAL_V1','level':100},
            {'kind':'r100_descriptors','domain':'CURRENT_Q07','level':99},
            {'kind':'exact_incidence_counter','domain':'CURRENT_Q07','level':100},
            {'kind':'V2_CAPACITY_PRODUCTS_AS_SITE_COUNTER','domain':'selected','level':7}]
        for i,row in enumerate(requests):
            p=deepcopy(plan);p['requests']=[row]
            refuses('population_'+str(i),lambda p=p:rp.compile_run_plan_v2(p,saved_read_calibration=calibration))
        p=deepcopy(plan);p['requests'].append(p['requests'][0])
        refuses('duplicate_request',lambda:rp.compile_run_plan_v2(p,saved_read_calibration=calibration))
        p=deepcopy(plan);p['theorems_adopted']=True
        refuses('extra_adoption_field',lambda:rp.compile_run_plan_v2(p,saved_read_calibration=calibration))
        refuses('frontier_alias',lambda:rp.compile_run_plan_v2(plan,frontier={},saved_read_calibration=calibration))
        refuses('legacy_constructor_V1_calibration',lambda:MaterializedDiscoverySession())
        for value in (7.0, True, '7'):
            refuses('session_request_type_'+repr(value),lambda value=value:sr.read_request(None,
                {'kind':'saved_packet','domain':'selected','level':value}))
        with tempfile.TemporaryDirectory(prefix='ig_saved_route_q14_') as directory:
            root=Path(directory)
            with zipfile.ZipFile(inputs['saved_inputs']) as z:
                assert len(z.namelist())==len(set(z.namelist()))
                assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
                z.extractall(root)
            for field,value in [('counter_unit','CAPACITY_SLOT'),('counter_anchor_ids',['unbound']*6),
                ('r100_domain','ALL_O100_STATES'),('production_authority','PRODUCTION')]:
                c=deepcopy(calibration);c[field]=value;c=sr.seal({k:v for k,v in c.items() if k!='science_sha256'})
                p=deepcopy(plan);p['calibration_sha256']=c['science_sha256']
                refuses('calibration_'+field,lambda c=c,p=p:MaterializedDiscoverySession.from_candidate_read_plan(
                    root,binding,c,inputs['q11_evidence'],master,p))
            altered=deepcopy(binding);altered['production_authority']='PRODUCTION'
            refuses('input_adoption',lambda:MaterializedDiscoverySession.from_candidate_read_plan(
                root,altered,calibration,inputs['q11_evidence'],master,plan))
            m=deepcopy(master);m['full_master_generation_allowed']=True
            refuses('master_authority',lambda:MaterializedDiscoverySession.from_candidate_read_plan(
                root,binding,calibration,inputs['q11_evidence'],m,plan))
            # Actual installed RunPlan executor enters the actual session API.
            result=rp.execute_saved_read_run_plan(plan,root=root,binding=binding,calibration=calibration,
                evidence_path=inputs['q11_evidence'],master_contract=master)
            assert len(result['observations'])==27
    finally:sys.setprofile(None)
    forbidden=[list(c) for c in sorted(set(calls)) if c[1] in ('_scan_level','_read_probe',
        '_resource_future_signature','_build_next_panel','_farthest_select','run_fixed_grammar_transport',
        'load_earned_regime_laws','construct_backbone','consume_complete','run_leaf_test')]
    assert not forbidden
    assert ('infinity_grid.materialized_discovery','from_candidate_read_plan') in calls
    assert ('infinity_grid.run_plan_v2','execute_saved_read_run_plan') in calls
    assert sum(c==('infinity_grid.maturation_parallel','_states_from_dag') for c in calls)==2
    rows=result['observations'];kinds={}
    for row in rows:
        kinds.setdefault(row['request']['kind'],[]).append(row)
        assert row['scientific_reexecution'] is False and row['production_authority']=='NONE'
    r100=kinds['r100_descriptors'][0]['payload']
    assert len(r100['descriptors'])==193
    assert r100['producer_binding']['producer_experiment_id']=='MASTER.STEP3.CURRENT.R100.Q07'
    assert r100['producer_binding']['source_sha256']=='76f3bfee70c13d34ac2eafc91060d96b199ee7d9fa794dd20d938417179ce2a2'
    assert sum(len(r['payload']['original_dispatch']['executions']) for r in kinds['saved_packet'])==100
    assert sum(len(r['payload']['evidence_payload']['rows']) for r in kinds['complete_read'])==42
    baseline=ss.verify_native_semantics(raise_on_change=False)
    theorem=tr.verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorem['stale_theorems'])==4
    write_json_atomic(Path('INSTALLED_RUN_PLAN_RESULT.json'),result)
    write_json_atomic(Path('REFUSALS.json'),{'status':'PASS','refusal_count':len(failures),'checks':failures})
    write_json_atomic(Path('SOURCE_COMPATIBILITY.json'),{'status':'PASS','definitions':compatibility,
        'forbidden_calls':forbidden,'source_bound_new_route':True,'legacy_default_guard':'REOPEN_REQUIRED',
        'stale_theorems':theorem['stale_theorems']})
    write_json_atomic(Path('MASTER_SCOPE_RECONCILIATION.json'),{'status':'PASS',
        'master_contract_sha256':canonical_sha256(master),'master_core':master['core'],
        'read_route_coverage':['TEN_SAVED_V2_PACKET_DOMAINS','SIX_O7_COMPLETE_READ_ANCHORS',
            'SIX_SAVED_FROZEN_MOTIF_FINDING_DEPTHS','FOUR_CLOSED_SAVED_TEMPORAL_WINDOWS','CURRENT_Q07_193_DESCRIPTORS'],
        'full_master_producer_compatible':False,'production_eligible':False,
        'unprovided_requirements':['COMPLETE_TYPED_INCIDENCE_ACTION_COUNTERS_WHERE_CONSUMED',
            'FULL_REQUIRED_ALTERNATIVES_AND_WITNESS_OCCURRENCE_EXPORT_CORRESPONDENCE',
            'FULL_REQUIRED_FAMILY_PRODUCER_DAG_AND_CURRENT_ADMISSIONS'],
        'R100_scanner_packet':'NOT_ADMITTED','R100_complete_counter':'NOT_ADMITTED',
        'old_theorem_guards':'UNCHANGED_FAIL_CLOSED','generation_allowed':False})
    write_json_atomic(Path('QUALIFICATION_RESULT.json'),{'status':'PASS',
        'scientific_outcome':'INSTALLED_SOURCE_BOUND_SAVED_READ_RUN_PLAN_QUALIFIED',
        'request_count':27,'saved_Test_executions_reused':100,'saved_complete_counters_reused':42,
        'R100_descriptors_read':193,'saved_temporal_windows_reused':4,'refusal_count':len(failures),
        'scientific_reexecutions':0,'stale_theorems':4,'production_authority':'NONE',
        'generation_allowed':False,'package2_complete':False,'production_cutover_complete':False,
        'body_elapsed_seconds':time.monotonic()-started})

if __name__=='__main__':main()
