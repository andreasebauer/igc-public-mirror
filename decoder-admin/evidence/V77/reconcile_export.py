from pathlib import Path
import json,sys,hashlib,runpy
D=Path(__file__).resolve().parent;G=D/sys.argv[1];R=Path('/tmp/ig_decoder_dev146_20261001');read=lambda p:json.loads(p.read_text());s=read(G/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion,export_workspace
assert (G/'MONITOR_EXIT.json').exists()
spec=read(G/'CAPTURE_SPEC.json');selectors=spec['execution']['nodes'];rows=[read(p) for p in (w/'runtime/runs').rglob('nodes/*.json')];cols=[read(p) for p in (w/'runtime/runs').rglob('collections/*.json')];errors=[read(p) for p in (w/'runtime/runs').rglob('collection_errors/*.json')]
passed=[r for r in rows if r.get('finished') and set(r['phases'])=={'setup','call','teardown'} and all(x['outcome']=='passed' and not x.get('xfail') for x in r['phases'].values())];bad=[r for r in rows if any(x['outcome']!='passed' or x.get('xfail') for x in r['phases'].values())]
collected_selectors=[x for c in cols for x in c['selectors']];nodes=[n for c in cols for n in c['nodes']];reported=[r['node'] for r in rows]
admission=validate_workspace_job(w,s['job_id'],check_loaded=False);done=verified_completion(admission,allow_pending_checkpoint=True)
complete=bool(done) and not errors and not bad and len(passed)==len(rows)==len(nodes) and len(set(nodes))==len(nodes) and set(nodes)==set(reported) and sorted(collected_selectors)==sorted(selectors) and read(G/'MONITOR_EXIT.json')['native_exit']==0 and not (G/'FIRST_NONPASS.json').exists()
if complete:assert pr.terminal_completion_proof(w,done)
report={'group':G.name,'status':'PASS' if complete else 'FAILED_INCOMPLETE','registered_files':len(selectors),'collected_files':len(collected_selectors),'collected_nodes':len(nodes),'node_reports':len(rows),'all_phases_passed':len(passed),'nonpass_nodes':len(bad),'unfinished_nodes':sum(not r.get('finished') for r in rows),'collection_errors':len(errors),'uncollected_selectors':sorted(set(selectors)-set(collected_selectors)),'native_completion':done['completion_sha256'] if done else None,'source_sha256':s.get('source_sha256',admission['source_sha256'] if 'source_sha256' in admission else None),'workload_attempts':len(list((w/'runtime/attempts').rglob('*.json')))}
for name,obj in [('RECONCILIATION',report),('NODE_REPORTS',rows),('COLLECTION_RECEIPTS',cols),('COLLECTION_ERRORS',errors),('NONPASS_REPORTS',bad),('VERIFIED_COMPLETION',done)]:
 with (G/(name+'.json')).open('x') as f:json.dump(obj,f,indent=2)
runpy.run_path(str(R/'decoder-import/verify_source.py'))['verify']()
assert not pr.status(w)['pending_objects']
e=export_workspace(w,G/'WORKSPACE_CHECKPOINT.zip',slim=True);(G/'EXPORT.json').write_text(json.dumps(e,indent=2));print(json.dumps(report))
