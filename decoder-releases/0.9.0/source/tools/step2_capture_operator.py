#!/usr/bin/env python3
"""Explicit operator for new Step 2 engineering captures. No synthetic save proofs."""
from pathlib import Path
from contextlib import closing
import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
RUNTIME=Path('/tmp/ig_runtime_v55_fresh_20260930/python-fixed-host')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(output):
    from infinity_grid import submission as sub,portable_registry
    from infinity_grid.sqlite_attestation import require_wal_fix
    from infinity_grid.v05_engineering_worker import engineering_source_tree_digest
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'CAPTURES.json').exists():raise RuntimeError('PREPARATION_ALREADY_EXISTS')
    binding=output/'SQLITE_BINDING.json'
    write(binding,{'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'REQUIRE_REVIEWED_WAL_BUILD','observation':require_wal_fix()})
    admin=Path('/tmp/ig_decoder_dev147_20261001/decoder-admin')
    artifact_paths=[('sqlite_runtime_binding',binding),('step2_protocol',SOURCE/'STEP2_PROTOCOL.txt'),
                    ('host_runtime_manifest',admin/'RUNTIME_MANIFEST.json'),('runtime_mode_policy',admin/'RUNTIME_POLICY.json')]
    artifacts=[{'logical_name':name,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in artifact_paths]
    profile=json.loads((SOURCE/'qualification/PROFILE.json').read_text())
    cases={}
    for name,workers,interval,pending in [('ordinary',2,30,8),('pause',1,30,8),('worker_loss',2,30,8),('backpressure',2,1,4)]:
        spec={'schema_id':sub.SPEC_SCHEMA,'job_id':'ENGINEERING.STEP2.DEV149.'+name.upper(),
            'engine_source':str(SOURCE),'project_source':None,
            'question':{'stage_id':'ENGINEERING:STEP2:BOUNDED','description':'Step 2 bounded generation/preservation fault qualification',
                        'outcomes':['PASS'],'stopping_rule':(SOURCE/'STEP2_PROTOCOL.txt').read_text()},
            'execution':{'kind':'STAGE','handler_ref':'infinity_grid.step2_engineering_fixture:handler',
                'evaluator_refs':['infinity_grid.step2_engineering_fixture:evaluate'],
                'parameters':{'tasks':24,'rounds':3000000,'shard_records':100000,'max_work_seconds':30}},
            'resources':{'workers':workers,'start_method':'spawn','memory_budget_bytes':1073741824,
                'workspace_budget_bytes':1073741824,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},
            'inputs':[],'environment':{'python':'3.13','requirements':[{k:r[k] for k in ('distribution','version','imports')} for r in profile['environment']],
                                       'artifacts':artifacts},
            'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'EXECUTION_ONLY',
                'required_artifacts':[],'result_checks':[{'pointer':'/outcome','equals':'PASS'},
                    {'pointer':'/generation/task_count','equals':24},
                    {'pointer':'/generation/raw_generated_occurrence_count','equals':48},
                    {'pointer':'/generation/distinct_exact_state_count','equals':36}],
                'prerequisites':[],'preservation':{'interval_seconds':interval,'max_pending_commits':pending,
                    'max_pending_bytes':268435456,'max_commit_bytes':67108864,'max_retained_commits':32,
                    'max_state_bytes':16777216,'max_state_file_bytes':8388608,'max_state_files':4096,
                    'max_state_shards':8}}}
        case=output/name;case.mkdir()
        write(case/'SPEC.json',spec)
        portable_registry.initialize(case/'store','DEV149 Step2 '+name,SOURCE)
        result=sub.capture(case/'store',spec);write(case/'CAPTURE_SAVE_STATUS.json',result)
        cases[name]={'workspace':result['workspace'],'capture_id':result['capture_id'],'job_id':spec['job_id']}
    write(output/'CAPTURES.json',{'source_sha256':engineering_source_tree_digest(SOURCE),'cases':cases})
    return {'status':'CAPTURED_SAVE_REQUIRED','cases':cases}


def native_run(workspace,result_file):
    sys.path.insert(0,str(workspace/'source'))
    from infinity_grid import submission as sub,v05_controller_event_loop as loop
    started=time.monotonic()
    try:
        result=loop.run_workspace_job(workspace,sub.capture_record(workspace)['job']['job_id'])
        write(result_file,{'status':result['status'],'wall_seconds':time.monotonic()-started,'native_result':result})
    except BaseException as exc:
        write(result_file,{'status':'INTERRUPTED','wall_seconds':time.monotonic()-started,
                           'error_type':type(exc).__name__,'error':str(exc),'code':getattr(exc,'code',None)})
        raise


def committed(workspace):
    paths=list((workspace/'runtime/runs').rglob('state_store.sqlite3'))
    if not paths:return None
    with closing(sqlite3.connect(paths[0].as_uri()+'?mode=ro',uri=True,timeout=1)) as conn:
        try:
            tasks=list(conn.execute('SELECT task_id,payload_sha256,state_occurrence_count FROM generation_tasks ORDER BY task_id'))
            occurrences=list(conn.execute('SELECT occurrence_id,task_id,state_token FROM occurrences ORDER BY occurrence_id'))
            identities=sorted({(m['engineering_worker_pid'],m['engineering_controller_pid'])
                for (raw,) in conn.execute('SELECT metrics_json FROM generation_tasks')
                for m in [json.loads(raw)]
                if 'engineering_worker_pid' in m and 'engineering_controller_pid' in m})
        except sqlite3.OperationalError:return None
    return {'tasks':tasks,'occurrences':occurrences,'worker_identities':identities}


def monitor(workspace,output,action):
    output.mkdir(parents=True,exist_ok=True);result_file=output/'NATIVE_RESULT.json';events=[]
    with (output/'NATIVE_RUN.log').open('w') as log:
        proc=subprocess.Popen([str(RUNTIME),'-B',str(Path(__file__).resolve()),'native-run','--workspace',str(workspace),'--output',str(result_file)],stdout=log,stderr=subprocess.STDOUT)
        started=time.monotonic();acted=None
        try:
            while proc.poll() is None:
                now=time.monotonic()
                if now-started>600:raise RuntimeError('ENGINEERING_OPERATOR_OBSERVATION_BOUND')
                prefix=committed(workspace) if action in ('pause','worker_loss') and acted is None else None
                if action in ('pause','worker_loss') and acted is None and prefix and len(prefix['tasks'])>=2:
                    # Use worker/parent identities emitted by the captured fixture.
                    # A managed host may expose a different PID namespace in /proc.
                    children=[pid for pid,parent in prefix['worker_identities'] if parent==proc.pid]
                    if children:
                        write(output/'PREFIX_BEFORE_EVENT.json',prefix)
                        if action=='pause':
                            from infinity_grid.preservation import request_pause
                            event=request_pause(workspace,'PREREGISTERED_STEP2_ACTIVE_WORK_PAUSE')
                        else:
                            victim=min(children);os.kill(victim,signal.SIGTERM)
                            event={'action':'SIGTERM_ONE_OWNED_POOL_WORKER','pid':victim,'controller_pid':proc.pid}
                        acted=time.monotonic();events.append(dict(event,seconds_since_start=acted-started,committed_tasks=len(prefix['tasks'])))
                        write(output/'OPERATOR_EVENTS.json',events)
                time.sleep(0.05)
            result={'return_code':proc.returncode,'wall_seconds':time.monotonic()-started,
                'action':action,'event_to_exit_seconds':None if acted is None else time.monotonic()-acted,'events':events,
                'committed_prefix':committed(workspace)}
            write(output/'OPERATOR_RESULT.json',result)
            return {k:v for k,v in result.items() if k!='committed_prefix'}
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=15)


def acknowledge(workspace,mapping_file,scope,output):
    from infinity_grid import submission as sub,preservation as pr
    mapping=json.loads(mapping_file.read_text())
    start=time.monotonic()
    if scope=='capture':
        pending=sub.save_status(workspace)['pending_objects']
        for row in pending:
            proof=mapping[row['sha256']]
            sub.confirm_save(workspace,row['sha256'],proof['path'],proof['drive_file_id'],role=row['role'],logical_name=row['logical_name'])
        result={'status':sub.save_status(workspace)['status'],'roles_confirmed':len(pending)}
    else:
        plan=pr.drain_plan(workspace)
        if not plan['obligations']:return {'status':'PRESERVED','roles_confirmed':0}
        batch={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':plan['capture_id'],'obligations':plan['obligations'],
            'readbacks':[{'sha256':row['sha256'],'kind':'RAW','path':mapping[row['sha256']]['path'],
                          'drive_file_id':mapping[row['sha256']]['drive_file_id']} for row in plan['physical_objects']]}
        request=output.with_suffix('.request.json');write(request,batch)
        result=pr.confirm_batch(workspace,request)
    result['operator_ack_wall_seconds']=time.monotonic()-start;write(output,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','native-run','monitor','pending','ack','export','restore'])
    p.add_argument('--workspace',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--action',default='ordinary');p.add_argument('--scope',default='checkpoint');p.add_argument('--mapping',type=Path)
    p.add_argument('--archive',type=Path);p.add_argument('--sha256');a=p.parse_args()
    if a.command=='prepare':r=prepare(a.output)
    elif a.command=='native-run':native_run(a.workspace,a.output);return
    elif a.command=='monitor':r=monitor(a.workspace,a.output,a.action)
    elif a.command=='ack':r=acknowledge(a.workspace,a.mapping,a.scope,a.output)
    else:
        from infinity_grid import submission as sub,preservation as pr
        if a.command=='pending':
            r=sub.save_status(a.workspace) if a.scope=='capture' else pr.drain_plan(a.workspace)
            write(a.output,r)
        elif a.command=='export':r=pr.export_checkpoint(a.workspace,a.output)
        else:r=pr.restore_checkpoint(a.archive,a.output,a.sha256)
    print(json.dumps(r,indent=2))

if __name__=='__main__':main()
