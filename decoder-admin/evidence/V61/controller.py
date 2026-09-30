from pathlib import Path
import sys,json,time,os,runpy,traceback
from lifecycle_monitor import process
D=Path(__file__).parent;R=Path('/tmp/ig_decoder_dev138_20260930');RT=Path('/tmp/ig_runtime_v55_fresh_20260930')
def write(n,x):
    with (D/(n+'.json')).open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
assert not sys.flags.optimize and sys.flags.dont_write_bytecode and Path(sys.executable).resolve()==RT/'base/bin/python3.13'
c=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(c['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import submission as sub,preservation as pr
from infinity_grid.v05_controller_event_loop import run_workspace_job,_source_ids
verify=runpy.run_path(str(R/'decoder-admin/decoder.py'))['verify_runtime']
assert not sub.save_status(w)['pending_objects'] and not pr.status(w)['pending_objects']
assert not list((w/'runtime/attempts').rglob('*.json'))
ids=_source_ids(w/'source');write('PRE_RUNTIME',verify(RT))
write('CONTROLLER_READY',{'owner':process('self'),'workspace':str(w),'capture_id':c['capture_id']})
try:write('RUN_RESULT',run_workspace_job(w,c['job_id']))
except BaseException as exc:
    write('RUN_EXCEPTION',{'type':type(exc).__name__,'reason':str(exc),'traceback':traceback.format_exc()});raise
finally:
    write('CONTROLLER_RETURNED',{'unix':time.time()})
    write('POST_RUNTIME',verify(RT));assert _source_ids(w/'source')==ids
    write('POST_SOURCE',{'source_sha256':ids[0],'package_sha256':ids[1],'unchanged':True})
