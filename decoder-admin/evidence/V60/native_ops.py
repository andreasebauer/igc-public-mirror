from pathlib import Path
import sys,json,hashlib,runpy,time,traceback,shutil,os
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev138_20260930')
RT=Path('/tmp/ig_runtime_v55_fresh_20260930');S=R/'decoder';JOB='RC.CLEANUP.V60.DEV143'
assert not sys.flags.optimize and sys.flags.dont_write_bytecode
assert Path(sys.executable).resolve()==RT/'base/bin/python3.13'
def read(p):return json.loads(p.read_text())
def write(n,x):
    with (D/(n+'.json')).open('x') as f:json.dump(x,f,indent=2)
mode=sys.argv[1];verify=runpy.run_path(str(R/'decoder-admin/decoder.py'))['verify_runtime']
if mode=='prepare':
    assert read(D/'RESULT.json')['returncode']==0
    write('NATIVE_PRE_RUNTIME',verify(RT));runpy.run_path(str(R/'decoder-import/verify_source.py'))['verify']()
    sys.path.insert(0,str(S))
    from infinity_grid import submission as sub,portable_registry
    from infinity_grid.sqlite_attestation import require_wal_fix
    write('SQLITE_BINDING',{'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'REQUIRE_REVIEWED_WAL_BUILD','observation':require_wal_fix()})
    artifacts=[]
    for name,p in [('sqlite_runtime_binding',D/'SQLITE_BINDING.json'),('host_runtime_manifest',R/'decoder-admin/RUNTIME_MANIFEST.json'),('runtime_mode_policy',R/'decoder-admin/RUNTIME_POLICY.json'),('gate_protocol',D/'NATIVE_PROTOCOL.txt'),('operator_script',Path(__file__))]:
        artifacts.append({'logical_name':name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    names=['test_cancel_before_worker_ready','test_cancel_active_selector_and_child','test_cleanup_already_exited_worker','test_missing_cleanup_ack_refused','test_malformed_cleanup_ack_refused']
    nodes=['tests/test_validation_cancellation.py::'+n for n in names]
    profile=read(S/'qualification/PROFILE.json')
    spec={'schema_id':sub.SPEC_SCHEMA,'job_id':JOB,'engine_source':str(S),'project_source':None,'question':{'stage_id':'RC:DEV143:CLEANUP','description':'V60 native cleanup regression gate','outcomes':['PASS','FAIL'],'stopping_rule':(D/'NATIVE_PROTOCOL.txt').read_text()},'execution':{'kind':'VALIDATION','nodes':nodes},'resources':{'workers':1,'start_method':'fork','memory_budget_bytes':1073741824,'workspace_budget_bytes':2147483648,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},'inputs':[],'environment':{'python':'3.13','requirements':[{k:r[k] for k in ['distribution','version','imports']} for r in profile['environment']],'artifacts':artifacts},'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'VALIDATION','required_artifacts':[],'result_checks':[{'pointer':'/status','equals':'PASS'}],'prerequisites':[]}}
    write('CAPTURE_SPEC',spec);store=Path('/tmp/ig_gate_v60_20261001/store')
    assert not store.exists();portable_registry.initialize(store,'V60 native dev143 cleanup regressions',S)
    state=sub.capture(store,spec);write('CAPTURE_SAVE_STATUS',state)
    print(json.dumps(state));sys.exit(0)
state=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(state['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub,preservation as pr
from infinity_grid.v05_controller_event_loop import run_workspace_job,validate_workspace_job,verified_completion,_source_ids
if mode=='capture_ack':
    for x in read(D/'CAPTURE_READBACKS.json'):
        sub.confirm_save(w,x['sha256'],x['readback'],x['drive_id'],role=x['role'],logical_name=x['logical_name'])
    assert not sub.save_status(w)['pending_objects'];write('CAPTURE_ACK',sub.save_status(w))
elif mode=='run':
    assert not sub.save_status(w)['pending_objects'] and not pr.status(w)['pending_objects']
    assert not list((w/'runtime/attempts').rglob('*.json'))
    ids=_source_ids(w/'source');write('NATIVE_STARTED',{'unix':time.time()})
    try:write('NATIVE_RESULT',run_workspace_job(w,JOB))
    except BaseException as exc:
        write('NATIVE_EXCEPTION',{'reason':str(exc),'traceback':traceback.format_exc()});raise
    finally:
        write('NATIVE_FINISHED',{'unix':time.time()});write('NATIVE_POST_RUNTIME',verify(RT));assert _source_ids(w/'source')==ids
elif mode=='status':
    status=pr.status(w);write('TERMINAL_STATUS',status);print(json.dumps(status))
elif mode=='ack':
    s=read(D/'TERMINAL_STATUS.json');by={x['sha256']:x for x in read(D/'TERMINAL_READBACKS.json')}
    batch={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':state['capture_id'],'obligations':[{k:x[k] for k in ['obligation_id','obligation_scope','role','logical_name','sha256','size_bytes']} for x in s['pending_objects']],'readbacks':[{'sha256':h,'kind':'RAW','drive_file_id':by[h]['drive_id'],'path':by[h]['readback']} for h in sorted(by)]}
    write('ACK_BATCH',batch);write('ACK_RESULT',pr.confirm_batch(w,D/'ACK_BATCH.json'))
    assert not pr.status(w)['pending_objects']
    done=verified_completion(validate_workspace_job(w,JOB,check_loaded=False),allow_pending_checkpoint=True)
    assert done and pr.terminal_completion_proof(w,done);write('VERIFIED_COMPLETION',done)
    write('NATIVE_EXPORT',pr.export_checkpoint(w,D/'V60_TERMINAL_CHECKPOINT.zip',slim=True))
elif mode=='restore':
    objects=D/'restore_objects';objects.mkdir()
    for x in read(D/'TERMINAL_READBACKS.json'):
        p=Path(x['readback']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
        shutil.copyfile(p,objects/(x['sha256']+'.bin'))
    dest=Path('/tmp/ig_gate_v60_restored_20261001')
    result=pr.restore_checkpoint(Path(read(D/'EXPORT_READBACK.json')['readback']),dest,read(D/'NATIVE_EXPORT.json')['sha256'],objects)
    done=verified_completion(validate_workspace_job(dest,JOB,check_loaded=False),allow_pending_checkpoint=True)
    assert done==read(D/'VERIFIED_COMPLETION.json') and pr.terminal_completion_proof(dest,done)
    write('RESTORE_VERIFICATION',{'restore':result,'completion_sha256':done['completion_sha256'],'exact_completion_equal':True,'terminal_completion_proof':True,'tests_dispatched':0})
else:raise RuntimeError('UNKNOWN_MODE')
