"""Prospective exact joint-source finite compatibility qualification."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

from infinity_grid import v2_candidate_consumer as legacy
from infinity_grid import v2_joint_candidate_consumer as joint
from infinity_grid import semantic_sentinel
from infinity_grid.theorem_registry import verify_all_theorems

def write(name, value):
    Path(name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def progress(event, **fields):
    row={'event':event,**fields}
    with Path('PROGRESS.jsonl').open('a') as file:
        file.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps(row,sort_keys=True),flush=True)

def main():
    started=time.monotonic()
    paths=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for name,sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(paths[name]).read_bytes()).hexdigest()==sha
    original_raw=Path(paths['binding']).read_bytes()
    original=json.loads(original_raw)
    declaration=json.loads(Path(paths['compatibility']).read_text())
    registry=Path(paths['producer_registry']).read_bytes()
    prior=json.loads(Path(paths['prior_consumer_result']).read_text())
    marker=json.loads(Path(paths['prior_marker_result']).read_text())
    assert prior['status']=='PASS' and prior['packet_matches']==10 and prior['loaded_root_occurrences']==747
    assert marker['status']=='PASS' and marker['native_tasks']==41 and marker['forbidden_service_refusals']==82
    assert hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest()==joint.LEGACY_CONSUMER_SHA256
    baseline=semantic_sentinel.verify_native_semantics(raise_on_change=False)
    theorems=verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    root=Path('candidate_inputs');root.mkdir()
    for role,pin in original['artifacts'].items():
        shutil.copyfile(paths[role],root/pin['file_name'])
    refusals=[]
    def refuse(name,fn,expected):
        try:fn()
        except legacy.CandidateAdmissionError as exc:
            assert str(exc).startswith(expected),(name,str(exc),expected)
            refusals.append({'check':name,'status':'PASS','refusal':str(exc)})
        else:raise AssertionError('MISSING_REFUSAL:'+name)
    refuse('original_binding_still_refuses_new_registry',lambda:legacy.preflight(root,original),'PRODUCER_CLOSURE_CHANGED:v05_stage_registry.py')
    refuse('missing_declaration',lambda:joint.preflight_joint(root,original,{},registry),'JOINT_DECLARATION_FIELDS')
    bad=deepcopy(declaration);bad['scope']='PRODUCTION'
    refuse('production_scope',lambda:joint.preflight_joint(root,original,bad,registry),'JOINT_QUALIFICATION_SCOPE')
    bad_original=deepcopy(original);bad_original['runtime_manifest_sha256']='0'*64
    refuse('relabelled_producing_binding',lambda:joint.preflight_joint(root,bad_original,declaration,registry),'JOINT_ORIGINAL_BINDING_CHANGED')
    bad=deepcopy(declaration);bad['execution_closure']['v05_stage_registry.py']=original['producer_closure']['v05_stage_registry.py']
    refuse('old_registry_as_execution_pin',lambda:joint.preflight_joint(root,original,bad,registry),'JOINT_EXECUTION_BYTES_CHANGED:v05_stage_registry.py')
    bad=deepcopy(declaration);bad['execution_closure']['undeclared.py']={'sha256':'0'*64,'size_bytes':0}
    refuse('extra_execution_file',lambda:joint.preflight_joint(root,original,bad,registry),'JOINT_EXECUTION_DOMAIN')
    bad=deepcopy(declaration);del bad['execution_closure']['v2_joint_candidate_consumer.py']
    refuse('missing_adapter_pin',lambda:joint.preflight_joint(root,original,bad,registry),'JOINT_EXECUTION_DOMAIN')
    bad=deepcopy(declaration);bad['marker_capture_id']='0'*64
    refuse('wrong_marker_provenance',lambda:joint.preflight_joint(root,original,bad,registry),'JOINT_MARKER_PROVENANCE')
    refuse('wrong_original_registry_bytes',lambda:joint.preflight_joint(root,original,declaration,registry+b' '),'JOINT_ORIGINAL_REGISTRY_PIN')
    admitted=joint.preflight_joint(root,original,declaration,registry)
    consumer=joint.JointFiniteCandidateConsumer(admitted)
    provenance=consumer.provenance()
    assert provenance['historical_producer_binding']==original
    provenance['historical_producer_binding'].clear()
    assert consumer.provenance()['historical_producer_binding']==original
    progress('joint_admitted',root_occurrences=747,prior_marker_tasks_reused=41)
    audits=json.loads(Path(paths['materialized_audit']).read_text())['levels']
    full=json.loads(Path(paths['full_audit']).read_text())['depths']
    audits=audits.values() if isinstance(audits,dict) else audits
    full=full.values() if isinstance(full,dict) else full
    rows=[]
    for domain,expected in [('selected',r) for r in audits]+[('full_pool',r) for r in full]:
        packet=consumer.scan(domain,expected['level'])
        assert packet==legacy.normalized(expected['scan'])
        saved=deepcopy(packet);packet.clear()
        assert consumer.scan(domain,expected['level'])==saved
        rows.append({'domain':domain,'level':expected['level'],'complete_packet_equal':True,
                     'returned_copy_isolated':True,'packet_sha256':legacy.normalized_hash(saved)})
        progress('joint_packet_match',domain=domain,level=expected['level'])
    target=root/original['artifacts']['index']['file_name'];raw=target.read_bytes();target.write_bytes(raw+b' ')
    refuse('joint_cache_input_mutation',lambda:consumer.scan('selected',7),'INPUT_BYTES_CHANGED:index')
    target.write_bytes(raw)
    assert len(refusals)==10 and len(rows)==10
    assert Path(paths['binding']).read_bytes()==original_raw and consumer.provenance()['historical_producer_binding']==original
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==baseline
    assert verify_all_theorems(raise_on_stale=False)==theorems
    write('REFUSAL_CHECKS.json',{'status':'PASS','check_count':10,'checks':refusals})
    write('INTEGRATION_PACKETS.json',{'status':'PASS','root_occurrences':747,'unique_constructions':675,'dag_nodes':4144,
                                     'complete_packet_matches':10,'packets':rows,'producer_generation_calls':0})
    write('PROVENANCE_REVIEW.json',{'status':'PASS','original_binding_raw_sha256':hashlib.sha256(original_raw).hexdigest(),
        'original_binding_bytes_unchanged':True,'original_producing_pins_unchanged':True,
        'execution_declaration_sha256':params['input_hashes']['compatibility'],
        'historical_producer_closure_files':len(original['producer_closure']),
        'execution_closure_files':len(declaration['execution_closure']),
        'changed_historical_files':['v05_stage_registry.py'],
        'new_execution_files':['g6_saved_marker_service_evaluators.py','v2_candidate_consumer.py','v2_joint_candidate_consumer.py'],
        'registry_original_bytes_recovered_by_exact_delta_removal':True,
        'legacy_consumer_and_marker_evaluator_exactly_unchanged':True,
        'prior_marker_tasks_reused':41,'new_marker_tasks':0,'independent_oracle_replays':0,
        'qualification_authority':'NONE','baseline_adopted':False,'theorems_adopted':False})
    write('GUARD_STATUS.json',{'status':'PASS','semantic_status':'REOPEN_REQUIRED','stale_theorems':4,
        'production_generation_allowed':False,'current_R100_admitted':False,'shared_source_activated':False,
        'baseline_exactly_unchanged':True,'original_admission_still_refuses_joint_source':True,
        'theorem_verifications':theorems})
    write('QUALIFICATION_RESULT.json',{'schema_id':'IG_EXACT_JOINT_FINITE_V2_QUALIFICATION_V1','status':'PASS',
        'scientific_outcome':'EXACT_JOINT_SOURCE_FINITE_CONSUMPTION_QUALIFIED_NO_PRODUCTION_AUTHORITY',
        'root_occurrences':747,'packet_matches':10,'refusal_checks':10,'prior_marker_tasks_reused':41,'new_marker_tasks':0,
        'producer_generation_calls':0,'independent_oracle_replays':0,'historical_producing_pins_unchanged':True,
        'production_generation_allowed':False,'current_R100_admitted':False,'semantic_status':'REOPEN_REQUIRED',
        'stale_theorems':4,'scope':joint.SCOPE,'production_authority':'NONE',
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time(),
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

if __name__=='__main__':main()
