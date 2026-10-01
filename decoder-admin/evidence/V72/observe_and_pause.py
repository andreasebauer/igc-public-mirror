from pathlib import Path
import json,subprocess,time,os,hashlib
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');wrapper='/tmp/ig_runtime_v55_fresh_20260930/python-fixed-host';env={**os.environ,'PYTHONOPTIMIZE':'0'}
s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace'])
assert not (D/'STARTED.json').exists()
with (D/'NATIVE_CONSOLE.txt').open('x') as log:
 p=subprocess.Popen([wrapper,'-B',str(D/'run_native.py')],stdout=log,stderr=subprocess.STDOUT,env=env)
 while p.poll() is None:
  logs=list((w/'runtime/runs').rglob('long_durable_progress.log'))
  if logs and logs[0].stat().st_size:
   raw=logs[0].read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
   assert rows[0]['event']=='START'
   (D/'ACTIVE_OBSERVATION.json').write_text(json.dumps({'unix':time.time(),'path':str(logs[0]),'sha256':hashlib.sha256(raw).hexdigest(),'rows':rows},indent=2))
   q=subprocess.run([wrapper,'-B',str(R/'decoder-admin/request_pause.py'),str(w),'V72_PREREGISTERED_ACTIVE_PAUSE'],capture_output=True,text=True,env=env)
   (D/'PAUSE_COMMAND.json').write_text(json.dumps({'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'unix':time.time()},indent=2))
   break
  time.sleep(.1)
 code=p.wait()
(D/'OPERATOR_EXIT.json').write_text(json.dumps({'native_exit':code,'pause_requested':(D/'PAUSE_COMMAND.json').exists(),'unix':time.time()},indent=2))
print((D/'OPERATOR_EXIT.json').read_text())
