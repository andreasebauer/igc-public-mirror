from pathlib import Path
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
