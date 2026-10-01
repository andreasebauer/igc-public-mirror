from pathlib import Path
import json,hashlib,zipfile,io,stat,sys
D=Path(__file__).resolve().parent
w=Path('/tmp/ig_qualification_v84_20261001/functional/store/captures/853b210f67a06c50ec51c262a72eae13918090e9e1246463a15327b074e88731');sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
from infinity_grid.validation_reports import pending_selectors
parts=json.loads((D/'PARTS.json').read_text());catalog=[]
archive=D/'V84_PAUSED_REMOTE_READBACK.zip'
with archive.open('xb') as out:
 for part in parts:
  saved=json.loads((D/(part['name']+'.readback.json')).read_text());raw=Path(saved['readback']).read_bytes();assert hashlib.sha256(raw).hexdigest()==part['sha256'];out.write(raw);catalog.append(saved)
e=json.loads((D/'EXPORT.json').read_text());assert hashlib.sha256(archive.read_bytes()).hexdigest()==e['sha256']
dest=Path('/tmp/ig_v84_paused_verified_recovery_20261001')
restored=pr.restore_checkpoint(archive,dest,e['sha256'])
checks=[]
with zipfile.ZipFile(archive) as outer:
 packet=json.loads(outer.read('CHECKPOINT.json'));row=packet['checkpoint']
 with zipfile.ZipFile(io.BytesIO(outer.read('objects/'+row['state']['sha256']))) as state:
  for name in state.namelist():
   raw=state.read(name);assert (dest/name).read_bytes()==raw;checks.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest()})
for name,mode in row['state_file_modes'].items():assert stat.S_IMODE((dest/name).stat().st_mode)==mode
a=[json.loads(p.read_text()) for p in (dest/'runtime/attempts').rglob('*.json')];assert len(a)==1 and a[0]['status']=='PAUSED'
assert verified_completion(validate_workspace_job(dest,a[0]['job_id'],check_loaded=False),allow_pending_checkpoint=True) is None
root=Path(a[0]['output_root']);relative=root.relative_to(w);rr=dest/relative
plan=json.loads((rr/'logs/REPORT_PLAN.json').read_text())
remaining=pending_selectors(rr/'logs',plan['binding'],plan['nodes'])
rows=[json.loads(p.read_text()) for p in (dest/'runtime/runs').rglob('nodes/*.json')]
passed=[r for r in rows if r.get('finished') and set(r['phases'])=={'setup','call','teardown'} and all(v['outcome']=='passed' and not v.get('xfail') for v in r['phases'].values())]
assert len(passed)==1406
assert not {r['node'] for r in passed}.intersection(remaining)
(D/'RECOVERY_PLAN.json').write_text(json.dumps({'status':'READ_ONLY_ASSESSMENT_NOT_EXECUTION_AUTHORIZATION','registered_selectors':len(plan['nodes']),'remaining_selectors':remaining,'remaining_count':len(remaining),'completed_nodes_not_selected':len(passed),'native_refusal_retry_is_safe':False,'requirements_before_any_dispatch':['Keep original paused capture immutable','Review prior finalization evidence','Resolve save gate using genuine remote readbacks','Reverify captured source/runtime and replaced environment artifact paths','Record explicit continuation deviation from original one-attempt qualification protocol'],'recommended_next':'Use preserved evidence; plan bounded master-data generation and older non-library decoder differential comparison. No blanket rerun or release promotion.'},indent=2))
report={'status':'EXACT_PAUSED_RECOVERY_VERIFIED','archive_sha256':e['sha256'],'state_files_verified':len(checks),'state_modes_verified':len(row['state_file_modes']),'passed_nodes':len(passed),'attempt_status':'PAUSED','native_completion':None,'workloads_executed':0,'restore':restored,'pending_save_obligations_not_acknowledged':243,'checks':checks}
(D/'RECOVERY_PROOF.json').write_text(json.dumps(report,indent=2));(D/'REMOTE_PART_CATALOG.json').write_text(json.dumps(catalog,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
