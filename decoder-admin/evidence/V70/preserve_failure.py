from pathlib import Path
import sys,json,hashlib,runpy,collections
D=Path(__file__).resolve().parent;P=D.parent/'v69b_preparation';R=Path('/tmp/ig_decoder_dev145_20261001');RT=Path('/tmp/ig_runtime_v55_fresh_20260930');read=lambda p:json.loads(p.read_text());s=read(P/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid.v05_controller_event_loop import export_workspace
rows=[]
for p in (w/'runtime/runs').rglob('nodes/*.json'):rows.append(read(p))
bad=[r for r in rows if any(v.get('outcome') not in ['passed'] for v in r.get('phases',{}).values())]
passed=[r for r in rows if r.get('finished') and set(r.get('phases',{}))=={'setup','call','teardown'} and all(v.get('outcome')=='passed' for v in r['phases'].values())]
(D/'NODE_REPORTS.json').write_text(json.dumps(rows,indent=2));(D/'NONPASS_REPORTS.json').write_text(json.dumps(bad,indent=2))
attempts={p.relative_to(w).as_posix():{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'record':read(p)} for p in (w/'runtime/attempts').rglob('*.json')};(D/'ATTEMPTS.json').write_text(json.dumps(attempts,indent=2))
summary={'node_reports':len(rows),'finished_all_phases_passed':len(passed),'nonpass_reports':len(bad),'unfinished':sum(not r.get('finished') for r in rows),'native_completion':False,'attempt_statuses':[r['record']['status'] for r in attempts.values()],'native_session_exit':130,'operator_stop':'Session Ctrl-C after first observed core-pin failure. Direct PID signals could not reach process across namespace. Wrapper ended before native pause/finally; retained RUNNING record is not evidence of a live process.','reruns':0};(D/'OBSERVED_RESULT.json').write_text(json.dumps(summary,indent=2))
verify=runpy.run_path(str(R/'decoder-admin/decoder.py'))['verify_runtime'];(D/'POST_RUNTIME.json').write_text(json.dumps(verify(RT),indent=2));runpy.run_path(str(R/'decoder-import/verify_source.py'))['verify']()
e=export_workspace(w,D/'V70_FAILURE_CHECKPOINT.zip',slim=True);(D/'EXPORT.json').write_text(json.dumps(e,indent=2));print(json.dumps(summary))
