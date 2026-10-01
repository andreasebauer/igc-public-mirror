from pathlib import Path
import sys,json,runpy,traceback,time
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev147_20261001');RT=Path('/tmp/ig_runtime_v55_fresh_20260930');read=lambda p:json.loads(p.read_text())
assert not sys.flags.optimize and sys.flags.dont_write_bytecode and Path(sys.executable).resolve()==RT/'base/bin/python3.13'
s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub,preservation as pr
from infinity_grid.v05_controller_event_loop import run_workspace_job
verify=runpy.run_path(str(R/'decoder-admin/decoder.py'))['verify_runtime']
def write(n,x):
 with (D/(n+'.json')).open('x') as f:json.dump(x,f,indent=2)
assert not sub.save_status(w)['pending_objects']
# One preregistered, saved checkpoint awaits genuine acknowledgment during startup.
assert len(pr.status(w)['pending_checkpoints'])==1
assert not list((w/'runtime/attempts').rglob('*.json'))
write('PREEXEC_RUNTIME',verify(RT));runpy.run_path(str(R/'decoder-import/verify_source.py'))['verify']();write('STARTED',{'unix':time.time(),'job_id':s['job_id']})
from infinity_grid import v05_controller_event_loop as loop
original_flock=loop.fcntl.flock
def observed_flock(handle, flags):
    target=Path(getattr(handle,'name',''))
    observe=target==pr._root(w)/'.runner.lock' and flags==loop.fcntl.LOCK_EX
    if observe:
        with (D/'LOCK_EVENTS.jsonl').open('a') as out:out.write(json.dumps({'event':'WAIT','unix':time.time()})+'\n')
    result=original_flock(handle,flags)
    if observe:
        with (D/'LOCK_EVENTS.jsonl').open('a') as out:out.write(json.dumps({'event':'ACQUIRED','unix':time.time()})+'\n')
    return result
loop.fcntl.flock=observed_flock
try:write('NATIVE_RESULT',run_workspace_job(w,s['job_id']))
except BaseException as exc:write('NATIVE_EXCEPTION',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
finally:write('POSTEXEC_RUNTIME',verify(RT));write('FINISHED',{'unix':time.time()})
