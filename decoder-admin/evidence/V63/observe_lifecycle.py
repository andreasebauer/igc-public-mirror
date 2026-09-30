"""Active external observation only; never dispatches, signals, or repairs."""
import json,os,time
from pathlib import Path
from lifecycle_monitor import Monitor,identity,inventory
D=Path(__file__).resolve().parent

def write(name,data):
    with (D/name).open('x') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())

def main():
    c=json.loads((D/'CONTROLLER_READY.json').read_text());w=Path(c['workspace']).resolve(strict=True)
    if not str(w).startswith('/tmp/ig_gate_v63_') or w.name!=c['capture_id']:raise RuntimeError('WORKSPACE_BINDING')
    mon=Monitor(c['owner']['pid'],w)
    if identity(mon.owner)!=identity(c['owner']):raise RuntimeError('OWNER_REUSED')
    first=mon.sample();write('MONITOR_READY.json',{'owner':c['owner'],'workspace':str(w),'first_sample':first})
    returned=None;first_inv=None;last_inv=None;stable_since=None;late_changes=False;survivors_at_return=[]
    start=time.monotonic()
    try:
        while time.monotonic()-start<360:
            row=mon.sample();now=time.monotonic()
            with (D/'PROCESS_SAMPLES.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n');f.flush()
            children=[p for p in row['processes'] if identity(p)!=identity(mon.owner)]
            if (D/'CONTROLLER_RETURNED.json').exists():
                if returned is None:
                    returned=now;survivors_at_return=children
                # Limit live inventory to runtime evidence; full snapshot waits for no processes.
                try:current=inventory(w/'runtime')
                except RuntimeError as e:
                    if str(e)!='EVIDENCE_CHANGED_DURING_INVENTORY':raise
                    late_changes=True;stable_since=None;time.sleep(.25);continue
                if first_inv is None:first_inv=current
                if last_inv is not None and current!=last_inv:late_changes=True;stable_since=None
                last_inv=current
                if not row['processes']:
                    if stable_since is None:stable_since=now
                    if now-stable_since>=3:
                        write('LIFECYCLE_RESULT.json',{'quiescent':True,'survivors_at_controller_return':survivors_at_return,'late_runtime_changes':late_changes,'lifecycle_pass':not survivors_at_return and not late_changes,'first_runtime_inventory':first_inv,'final_runtime_inventory':current,'scope':'sampled registered process identities; short-lived descendants can be missed'})
                        return
                else:stable_since=None
                if now-returned>150:raise RuntimeError('QUIESCENCE_NOT_ESTABLISHED')
            time.sleep(.25)
        raise RuntimeError('OBSERVATION_TIMEOUT')
    except BaseException as e:
        write('LIFECYCLE_ERROR.json',{'reason':str(e),'unix':time.time()});raise
    finally:mon.save(D/'LIFECYCLE_OBSERVATIONS.json')

if __name__=='__main__':main()
