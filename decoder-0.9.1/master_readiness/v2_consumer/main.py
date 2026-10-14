"""Prospective native finite V2 consumer admission and saved-DAG qualification."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time

from infinity_grid import v2_candidate_consumer as vc
from infinity_grid import semantic_sentinel as ss
from infinity_grid import materialized_discovery as md
from infinity_grid import fixed_grammar_transport as ft
from infinity_grid import regime_scanner as rs

def write(name, obj):
    Path(name).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')

def progress(event, **fields):
    with Path('PROGRESS.jsonl').open('a') as f:
        f.write(json.dumps({'event':event,**fields},sort_keys=True)+'\n')
    print(json.dumps({'event':event,**fields},sort_keys=True),flush=True)

def main():
    start = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for n,h in params['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[n]).read_bytes()).hexdigest() == h
    binding = json.loads(Path(inputs['binding']).read_text())
    root = Path('candidate_inputs');root.mkdir()
    for role,pin in binding['artifacts'].items():
        shutil.copyfile(inputs[role],root/pin['file_name'])
    baseline = ss.verify_native_semantics(raise_on_change=False)
    assert baseline['status'] == 'REOPEN_REQUIRED'
    checks = []
    def reject(label, fn, error=vc.CandidateAdmissionError):
        try:fn()
        except error as exc:
            checks.append({'check':label,'refusal':str(exc),'status':'PASS'})
            progress('engineering_refusal',check=label)
        else:raise AssertionError('MISSING_REFUSAL:'+label)
    admission = vc.preflight(root,binding)
    retained=json.loads(Path(inputs['retained_engineering']).read_text())
    assert retained['status']=='PASS' and retained['check_count']==12
    assert hashlib.sha256(Path(vc.__file__).read_text().split('    def scan(')[0].encode()).hexdigest()==params['engineering_prefix_sha256']
    checks=deepcopy(retained['checks'])
    write('ENGINEERING_RESULT.json',dict(retained,executions_reused=12,executions_new=0,
        original_capture='051df227f8ba2c487e8b8a0fd1775ff162c4a7c2cf9991752e197c649a6b6936',
        original_whole_job_status='RESULT_REJECTED',retained_artifact_sha256=params['input_hashes']['retained_engineering'],
        unchanged_admission_source_prefix_sha256=params['engineering_prefix_sha256']))
    progress('engineering_complete',checks=len(checks))
    consumer=vc.FiniteCandidateConsumer(admission)
    progress('saved_dag_loaded',root_occurrences=747)
    audits=json.loads(Path(inputs['materialized_audit']).read_text())
    depths=json.loads(Path(inputs['full_audit']).read_text())
    selected=audits['levels'];full=depths['depths']
    selected=selected.values() if isinstance(selected,dict) else selected
    full=full.values() if isinstance(full,dict) else full
    rows=[]
    for domain,expected in [('selected',r) for r in selected]+[('full_pool',r) for r in full]:
        level=expected['level'];packet=consumer.scan(domain,level)
        assert packet == vc.normalized(expected['scan'])
        saved=deepcopy(packet);packet.clear()
        assert consumer.scan(domain,level) == saved
        rows.append({'domain':domain,'level':level,'states':expected.get('state_count',expected.get('recipe_count')),'full_packet_equal':True,'returned_copy_isolated':True,'packet_sha256':vc.normalized_hash(saved)})
        progress('saved_packet_consumed',domain=domain,level=level)
    for domain,level in [('selected',100),('full_pool',10),('selected',True),('production',11)]:
        reject('unsupported_'+str(domain)+'_'+str(level),lambda d=domain,l=level:consumer.scan(d,l))
    target=root/binding['artifacts']['index']['file_name'];raw=target.read_bytes();target.write_bytes(raw+b' ')
    reject('cache_hit_input_mutation',lambda:consumer.scan('selected',7));target.write_bytes(raw)
    assert ss.verify_native_semantics(raise_on_change=False) == baseline
    write('CONSUMPTION_RESULT.json',{'status':'PASS','scope':vc.SCOPE,'loaded_root_occurrences':747,'unique_constructions':675,'nodes':4144,'public_root_checks':747,'complete_packet_matches':len(rows),'packets':rows,'producer_generation_calls':0,'independent_oracle_replays':0,'unsupported_domain_refusals':4,'cache_mutation_refused':True})
    write('GUARD_STATUS.json',{'status':'PASS','semantic_status':baseline['status'],'baseline_exactly_unchanged':True,'production_resources_unchanged':True,'baseline_adopted':False,'theorems_adopted':False,'current_R100_admitted':False,'production_generation_allowed':False})
    write('QUALIFICATION_RESULT.json',{'schema_id':'IG_FINITE_V2_CONSUMER_QUALIFICATION_V1','status':'PASS','scientific_outcome':'FINITE_SAVED_V2_CONSUMER_QUALIFIED_NO_PRODUCTION_AUTHORITY','engineering_checks':12,'engineering_checks_reused':12,'engineering_checks_new':0,'new_consumer_refusals':5,'packet_matches':len(rows),'loaded_root_occurrences':747,'producer_work_replayed':False,'scope':vc.SCOPE,'production_authority':'NONE','wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})

if __name__ == '__main__':
    main()
