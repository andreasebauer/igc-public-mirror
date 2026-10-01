from pathlib import Path
import json,subprocess,time,os
D=Path(__file__).resolve().parent;R=Path('/tmp/ig_decoder_dev146_20261001');wrapper='/tmp/ig_runtime_v55_fresh_20260930/python-fixed-host';env={**os.environ,'PYTHONOPTIMIZE':'0'}
s=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(s['workspace']);assert not (D/'STARTED.json').exists()
seen={};requested=False
with (D/'NATIVE_CONSOLE.txt').open('x') as out:
 p=subprocess.Popen([wrapper,'-B',str(D/'run_native.py')],stdout=out,stderr=subprocess.STDOUT,env=env)
 while p.poll() is None:
  if not requested:
   bad=[]
   for path in (w/'runtime/runs').rglob('*.json'):
    if path.parent.name not in ['nodes','collection_errors']:continue
    st=path.stat().st_mtime_ns
    if seen.get(str(path))==st:continue
    seen[str(path)]=st;r=json.loads(path.read_text())
    if path.parent.name=='collection_errors' or any(v.get('outcome')!='passed' or v.get('xfail') for v in r.get('phases',{}).values()):bad.append({'path':str(path),'record':r})
   if bad:
    (D/'FIRST_NONPASS.json').write_text(json.dumps({'unix':time.time(),'reports':bad},indent=2));requested=True
    q=subprocess.run([wrapper,'-B',str(R/'decoder-admin/request_pause.py'),str(w),'V77_FIRST_NONPASS_STOP'],capture_output=True,text=True,env=env)
    (D/'PAUSE_COMMAND.json').write_text(json.dumps({'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr,'unix':time.time()},indent=2))
  time.sleep(.2)
 code=p.wait()
(D/'MONITOR_EXIT.json').write_text(json.dumps({'native_exit':code,'pause_requested':requested,'unix':time.time()},indent=2));print((D/'MONITOR_EXIT.json').read_text())
