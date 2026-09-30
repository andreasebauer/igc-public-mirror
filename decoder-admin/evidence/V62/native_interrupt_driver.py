from pathlib import Path
import json,subprocess,time,signal,os
from lifecycle_monitor import Monitor,identity,inventory
D=Path(__file__).parent;RT=Path('/tmp/ig_runtime_v55_fresh_20260930')
def write(n,x):
    with (D/(n+'.json')).open('x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
assert not (D/'OPERATOR_STARTED.json').exists()
c=json.loads((D/'CAPTURE_SAVE_STATUS.json').read_text());w=Path(c['workspace'])
write('OPERATOR_STARTED',{'unix':time.time(),'workspace':str(w)})
with (D/'CONTROLLER_OUTPUT.txt').open('x') as log:
    p=subprocess.Popen([str(RT/'python-fixed-host'),'-B',str(D/'controller.py')],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONOPTIMIZE='0'))
    mon=None;sent=False;returned=None;first=None;last=None;late=False;survivors=[];start=time.monotonic();stable=None
    try:
        while time.monotonic()-start<230:
            if mon is None and (D/'CONTROLLER_READY.json').exists():
                ready=json.loads((D/'CONTROLLER_READY.json').read_text());assert ready['workspace']==str(w)
                mon=Monitor(ready['owner']['pid'],w);assert identity(mon.owner)==identity(ready['owner'])
            row=mon.sample() if mon else None
            if row:
                with (D/'PROCESS_SAMPLES.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                children=[x for x in row['processes'] if identity(x)!=identity(mon.owner)]
                if not sent and time.monotonic()-start<60:
                    logs=list((w/'runtime/runs').rglob('probe_progress/long_durable_progress.log'))
                    if len(logs)==1:
                        rows=[json.loads(x) for x in logs[0].read_text().splitlines()]
                        live=[x for x in children if 'infinity_grid.validation_node_runner' in x['cmdline'] and x['state']!='Z']
                        if rows and rows[0]['event']=='START' and any(x['event']=='PROGRESS' and x['tick']==1 for x in rows) and not any(x['event']=='END' for x in rows) and live and p.poll() is None:
                            write('INTERRUPT_INTENT',{'unix':time.time(),'owned_popen_pid':p.pid,'observed_owner':mon.owner,'live_selectors':live,'durable_progress':rows})
                            sent=True;p.send_signal(signal.SIGINT)
                            write('INTERRUPT_SENT',{'unix':time.time(),'signal':'SIGINT','target':'owned unreaped direct Popen child','count':1})
                if (D/'CONTROLLER_RETURNED.json').exists():
                    if returned is None:returned=time.monotonic();survivors=children
                    current=inventory(w/'runtime')
                    if first is None:first=current
                    if last is not None and current!=last:late=True
                    last=current
                    if not children and p.poll() is not None:
                        if stable is None:stable=time.monotonic()
                        if time.monotonic()-stable>=3:
                            write('LIFECYCLE_RESULT',{'interrupt_sent':sent,'controller_returncode':p.returncode,'survivors_at_controller_return':survivors,'late_runtime_changes':late,'quiescent':True,'lifecycle_pass':sent and not survivors and not late,'first_runtime_inventory':first,'final_runtime_inventory':last,'scope':'sampled identities; short-lived descendants may be missed'})
                            break
                    else:stable=None
            if p.poll() is not None and not (D/'CONTROLLER_RETURNED.json').exists():raise RuntimeError('CONTROLLER_NO_RETURN_MARKER')
            time.sleep(.1)
        else:raise RuntimeError('OBSERVATION_TIMEOUT')
    finally:
        if mon:mon.save(D/'LIFECYCLE_OBSERVATIONS.json')
assert json.loads((D/'LIFECYCLE_RESULT.json').read_text())['lifecycle_pass']
assert json.loads((D/'RUN_EXCEPTION.json').read_text())['type']=='KeyboardInterrupt'
assert not (D/'RUN_RESULT.json').exists()
print('INTERRUPT_CLEANUP_OBSERVATION_PASS')
