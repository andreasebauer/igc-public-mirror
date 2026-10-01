from pathlib import Path
import json,hashlib
D=Path(__file__).parent;P=D.parent/'v72_native_pause'
for name in ['prepare_capture.py','run_native.py','transport_admin.py','verify_and_export.py','restore_exact.py']:
 s=(P/name).read_text().replace('dev146','dev147').replace('DEV146','DEV147').replace('V72','V79').replace('v72','v79')
 if name=='run_native.py':
  s=s.replace("assert not sub.save_status(w)['pending_objects'] and not pr.status(w)['pending_objects']","assert not sub.save_status(w)['pending_objects']\n# One preregistered, saved checkpoint awaits genuine acknowledgment during startup.\nassert len(pr.status(w)['pending_checkpoints'])==1")
  insert='''from infinity_grid import v05_controller_event_loop as loop
original_flock=loop.fcntl.flock
def observed_flock(handle, flags):
    target=Path(getattr(handle,'name',''))
    observe=target==pr._root(w)/'.runner.lock' and flags==loop.fcntl.LOCK_EX
    if observe:
        with (D/'LOCK_EVENTS.jsonl').open('a') as out:out.write(json.dumps({'event':'WAIT','unix':time.time()})+'\\n')
    result=original_flock(handle,flags)
    if observe:
        with (D/'LOCK_EVENTS.jsonl').open('a') as out:out.write(json.dumps({'event':'ACQUIRED','unix':time.time()})+'\\n')
    return result
loop.fcntl.flock=observed_flock
'''
  s=s.replace("try:write('NATIVE_RESULT'",insert+"try:write('NATIVE_RESULT'")
 (D/name).write_text(s)
(D/'PROTOCOL.txt').write_text('''V79 DEV147 NATIVE STARTUP CONTENTION AND PAUSE — 2026-10-01
Exact dev147 source 098d505dfba5575aec3efc9f2ecf3614a28be3b7caeda41482a94dda2e5a1ea8.
One fresh capture and native attempt. Reviewed CPython runtime, PYTHONOPTIMIZE=0, -B. One worker, fork; default checkpoint/backlog policy unchanged.
Select only native_save_wave_probe.py::test_long_durable_progress_selector. No workload retries.
Save/read back every capture prerequisite. Create a pre-run checkpoint without workload dispatch; save and read back its objects. Keep this one pending checkpoint within native backlog limits at startup.
The separate acknowledgment operator gates its first _verify_raw call while owning the outbox lock. It verifies genuine downloaded bytes and publishes genuine receipts after release. It has a 60-second gate guard.
After ACK_LOCKED launch native controller. Observe flock WAIT/ACQUIRED without changing flags or behavior. After WAIT remains unacquired for 0.5 seconds, release acknowledgment. Require ACK_DONE and native ACQUIRED in sequence. This is induced startup checkpoint contention, not evidence of incidental overlap during an active worker.
Then observe durable worker START and request native pause with V79_PREREGISTERED_ACTIVE_PAUSE. Never signal workload. A 45-second observation guard releases the gate and requests native pause if the expected overlap/progress is absent; it does not turn absence into PASS.
Require PAUSED attempt, matching pause acknowledgment, retained REQUESTED_PAUSE refusal, no completion. Worker cleanup is inferred from reviewed fail-closed native path, not independently persisted pipe acknowledgment.
Drain saves, export slim checkpoint, save/download export, restore from downloaded dependencies without workload execution. Compare every state/input byte and ordinary mode. Verify source/runtime unchanged. Any failure remains immutable. All 23 RC rows OPEN; no independent-host claim.
''')
(D/'prepare_checkpoint.py').write_text('''from pathlib import Path
import sys,json
D=Path(__file__).parent;s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr,submission as sub
assert not sub.save_status(w)['pending_objects']
assert not list((w/'runtime/attempts').rglob('*.json'))
s=pr.make_checkpoint(w,'V79_PREREGISTERED_PRE_RUN')
(D/'PRE_RUN_STATUS.json').write_text(json.dumps(s,indent=2))
print(json.dumps({'pending_roles':len(s['pending_objects']),'pending_checkpoints':len(s['pending_checkpoints'])}))
''')
(D/'gated_ack.py').write_text('''from pathlib import Path
import sys,json,time
D=Path(__file__).parent;s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation_batch as batch
original=batch._verify_raw;first=True
def verify(path,expected):
 global first
 if first:
  first=False;(D/'ACK_LOCKED.json').write_text(json.dumps({'unix':time.time()}));start=time.monotonic()
  while not (D/'ACK_RELEASE.json').exists():
   if time.monotonic()-start>60:raise RuntimeError('ACK_GATE_GUARD')
   time.sleep(.02)
 return original(path,expected)
batch._verify_raw=verify
try:
 result=batch.confirm_batch(w,D/'PRE_RUN_BATCH.json');(D/'GATED_ACK.json').write_text(json.dumps({'unix':time.time(),'result':result},indent=2))
except BaseException as exc:
 (D/'ACK_ERROR.json').write_text(json.dumps({'unix':time.time(),'error':repr(exc)}));raise
''')
(D/'observe_and_pause.py').write_text('''from pathlib import Path
import json,subprocess,time,os,hashlib
D=Path(__file__).parent;R=Path('/tmp/ig_decoder_dev147_20261001');wrapper='/tmp/ig_runtime_v55_fresh_20260930/python-fixed-host';env={**os.environ,'PYTHONOPTIMIZE':'0'}
s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace'])
def write(n,x):(D/(n+'.json')).write_text(json.dumps(x,indent=2))
assert not (D/'STARTED.json').exists();begin=time.monotonic()
with (D/'ACK_CONSOLE.txt').open('x') as a,(D/'NATIVE_CONSOLE.txt').open('x') as log:
 ack=subprocess.Popen([wrapper,'-B',str(D/'gated_ack.py')],stdout=a,stderr=subprocess.STDOUT,env=env)
 while not (D/'ACK_LOCKED.json').exists():
  if ack.poll() is not None:raise RuntimeError('ACK_FAILED_BEFORE_GATE')
  time.sleep(.02)
 p=subprocess.Popen([wrapper,'-B',str(D/'run_native.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
 overlap=False;pause=False
 while p.poll() is None:
  events=[json.loads(x) for x in (D/'LOCK_EVENTS.jsonl').read_text().splitlines()] if (D/'LOCK_EVENTS.jsonl').exists() else []
  if not (D/'ACK_RELEASE.json').exists() and events and events[-1]['event']=='WAIT' and time.time()-events[-1]['unix']>=.5:
   overlap=True;write('ACK_RELEASE',{'unix':time.time(),'reason':'OBSERVED_WAIT_WITH_ACK_HELD'})
  logs=list((w/'runtime/runs').rglob('long_durable_progress.log'))
  active=logs and logs[0].stat().st_size and (D/'GATED_ACK.json').exists()
  guard=time.monotonic()-begin>45
  if active or guard:
   if not (D/'ACK_RELEASE.json').exists():write('ACK_RELEASE',{'unix':time.time(),'reason':'GUARD_RELEASE'})
   if active:
    raw=logs[0].read_bytes();write('ACTIVE_OBSERVATION',{'unix':time.time(),'sha256':hashlib.sha256(raw).hexdigest(),'rows':[json.loads(x) for x in raw.splitlines()]})
   q=subprocess.run([wrapper,'-B',str(R/'decoder-admin/request_pause.py'),str(w),'V79_PREREGISTERED_ACTIVE_PAUSE'],capture_output=True,text=True,env=env)
   write('PAUSE_COMMAND',{'unix':time.time(),'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'guard':guard});pause=True;break
  time.sleep(.02)
 if not (D/'ACK_RELEASE.json').exists():write('ACK_RELEASE',{'unix':time.time(),'reason':'NATIVE_EXIT_RELEASE'})
 code=p.wait();ackcode=ack.wait()
write('OPERATOR_EXIT',{'native_exit':code,'ack_exit':ackcode,'pause_requested':pause,'overlap_observed':overlap,'unix':time.time()})
print((D/'OPERATOR_EXIT.json').read_text())
''')
names=['prepare_capture.py','run_native.py','transport_admin.py','verify_and_export.py','restore_exact.py','prepare_checkpoint.py','gated_ack.py','observe_and_pause.py']
(D/'OPERATOR_SCRIPT_HASHES.json').write_text(json.dumps({n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in names},indent=2))
