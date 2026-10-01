from pathlib import Path
import json,sys,hashlib,shutil,zipfile,io,stat
D=Path(__file__).resolve().parent/'functional';read=lambda p:json.loads(p.read_text());s=read(D/'CAPTURE_SAVE_STATUS.json');w=Path(s['workspace']);sys.path.insert(0,str(w/'source'))
from infinity_grid import preservation as pr
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
needed={x['sha256'] for x in read(D/'EXPORT.json')['dependencies']};objects=D/'restore_objects';objects.mkdir()
for r in read(D/'READBACKS.json'):
 
 if r['sha256'] not in needed:continue
 if r['kind']=='RAW':
  src=Path(r['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==r['sha256'];shutil.copyfile(src,objects/(r['sha256']+'.bin'))
 else:
  from infinity_grid.save_transport import unpack
  unpack(r['manifest'],r['parts'],objects/(r['sha256']+'.bin'))
e=read(D/'EXPORT.json');rb=read(D/'EXPORT_READBACK.json');raw=Path(rb['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==e['sha256']
with zipfile.ZipFile(io.BytesIO(raw)) as z:packet=json.loads(z.read('CHECKPOINT.json'))
row=packet['checkpoint'];dest=Path('/tmp/ig_v77_failure_restored_20261001');result=pr.restore_checkpoint(rb['path'],dest,e['sha256'],objects)
checks=[]
with zipfile.ZipFile(objects/(row['state']['sha256']+'.bin')) as z:
 for name in z.namelist():
  data=z.read(name);assert (dest/name).read_bytes()==data
  checks.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes_equal':True})
for name,mode in row['state_file_modes'].items():
 assert stat.S_IMODE((dest/name).stat().st_mode)==mode
 assert (dest/name).read_bytes()==(w/name).read_bytes()
checks_by={x['path']:x for x in checks}
for name,mode in row['state_file_modes'].items():
 checks_by.setdefault(name,{'path':name,'sha256':hashlib.sha256((dest/name).read_bytes()).hexdigest(),'bytes_equal':True}).update(mode=mode,mode_equal=True)
a=[read(p) for p in (dest/'runtime/attempts').rglob('*.json')];assert len(a)==1 and a[0]['status']=='PAUSED'
assert list((dest/'runtime/refusals').glob('*.json'))
assert verified_completion(validate_workspace_job(dest,s['job_id'],check_loaded=False),allow_pending_checkpoint=True) is None
report={'status':'EXACT_PAUSED_RESTORATION_PASS','restore':result,'export_sha256':e['sha256'],'files_checked':len(checks_by),'checks':list(checks_by.values()),'attempt_status':'PAUSED','refusal_retained':True,'native_completion':None,'recovery_workloads_dispatched':0}
(D/'RESTORE_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
