from pathlib import Path
import json,sys,hashlib,shutil
from lifecycle_monitor import inventory
D=Path(__file__).parent;c=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(c['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr,submission as sub
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
def read(p):return json.loads(p.read_text())
def write(n,x):
    with (D/(n+'.json')).open('x') as f:json.dump(x,f,indent=2)
mode=sys.argv[1]
if mode=='capture_ack':
    for x in read(D/'CAPTURE_READBACKS.json'):sub.confirm_save(w,x['sha256'],x['readback'],x['drive_id'],role=x['role'],logical_name=x['logical_name'])
    s=sub.save_status(w);assert not s['pending_objects'];write('CAPTURE_ACK',s)
elif mode=='status':
    assert read(D/'LIFECYCLE_RESULT.json')['lifecycle_pass']
    assert read(D/'RUN_EXCEPTION.json')['type']=='KeyboardInterrupt'
    attempts=[read(p) for p in (w/'runtime/attempts').rglob('*.json')]
    assert len(attempts)==1 and attempts[0]['status']=='PAUSED'
    assert verified_completion(validate_workspace_job(w,c['job_id'],check_loaded=False),allow_pending_checkpoint=True) is None
    assert not (w/'durability/CHECKPOINT_FAILURE.json').exists()
    write('PAUSE_VERIFICATION',{'attempts':attempts,'accepted_completion':None,'runtime_inventory':inventory(w/'runtime')})
    s=pr.status(w);write('PAUSE_STATUS',s);print(json.dumps({'roles':len(s['pending_objects']),'unique':len({x['sha256'] for x in s['pending_objects']})}))
elif mode=='ack':
    s=read(D/'PAUSE_STATUS.json');by={x['sha256']:x for x in read(D/'PAUSE_READBACKS.json')}
    batch={'schema_id':'IG_CHECKPOINT_ACK_BATCH_V1','capture_id':c['capture_id'],'obligations':[{k:x[k] for k in ['obligation_id','obligation_scope','role','logical_name','sha256','size_bytes']} for x in s['pending_objects']],'readbacks':[{'sha256':h,'kind':'RAW','drive_file_id':by[h]['drive_id'],'path':by[h]['readback']} for h in sorted(by)]}
    write('ACK_BATCH',batch);write('ACK_RESULT',pr.confirm_batch(w,D/'ACK_BATCH.json'))
    assert not pr.status(w)['pending_objects']
    assert inventory(w/'runtime')==read(D/'PAUSE_VERIFICATION.json')['runtime_inventory']
    write('NATIVE_EXPORT',pr.export_checkpoint(w,D/'V62_PAUSED_CHECKPOINT.zip',slim=True))
elif mode=='restore':
    objects=D/'restore_objects';objects.mkdir()
    for x in read(D/'PAUSE_READBACKS.json'):
        p=Path(x['readback']);assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
        shutil.copyfile(p,objects/(x['sha256']+'.bin'))
    dest=Path('/tmp/ig_gate_v62_restored_20261001')
    result=pr.restore_checkpoint(Path(read(D/'EXPORT_READBACK.json')['readback']),dest,read(D/'NATIVE_EXPORT.json')['sha256'],objects)
    assert inventory(dest/'runtime')==read(D/'PAUSE_VERIFICATION.json')['runtime_inventory']
    attempts=[read(p) for p in (dest/'runtime/attempts').rglob('*.json')]
    assert attempts==read(D/'PAUSE_VERIFICATION.json')['attempts']
    assert verified_completion(validate_workspace_job(dest,c['job_id'],check_loaded=False),allow_pending_checkpoint=True) is None
    write('RESTORE_VERIFICATION',{'restore':result,'runtime_bytes_and_modes_equal':True,'paused_attempt_equal':True,'accepted_completion':None,'tests_dispatched':0})
else:raise RuntimeError('UNKNOWN_MODE')
