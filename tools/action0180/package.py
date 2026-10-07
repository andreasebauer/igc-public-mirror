from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from verify_saved import validate
R=Path.cwd();B=R/'g_public_export0180';D=R/'handoff0180_public_export';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=validate();assert v==json.loads((B/'VALIDATION.json').read_text())
reg=json.loads((R/'g_public_register0179/FAMILY_REGISTER.json').read_text());reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='SAVED_PUBLIC_PROJECTION_NATIVE_COMPLETION_AND_COLD_REUSE_VERIFIED;SCOPED_ADMISSION_PENDING;EXACT_DAG_BLOCKED';dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='NATIVE_SAVED_PUBLIC_EXPORT_PASS;SCOPED_PROJECTION_ADMISSION_PENDING',scope_reconciliation='g_public_export0180/FAMILY_REGISTER.json',G1_public_export='g_public_export0180/EXPORT_BINDINGS.json',G1_public_export_result='g_public_export0180/NATIVE_RESULT.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['operate.py','cold_ops.py','prepare_saved.py','verify_saved.py','package.py','REPORT.txt','SPEC.json','POINTER.json','NATIVE_RESULT.json','CHECKPOINT_ACK.json','EXPORT.json','FINAL_CHECKPOINT_SLIM.zip','CHECKPOINT_ARCHIVE_SAVED.json','COLD_RESTORE.json','COLD_REUSE_RESULT.json','SCIENTIFIC_EXPORT.zip','SCIENTIFIC_EXPORT_SAVED.json','EXPORT_BINDINGS.json','VALIDATION.json','FAMILY_REGISTER.json','CODE_MIRROR.json','READBACKS.json','CHECKPOINT_READBACKS.json']:shutil.copy2(B/n,t/n)
dump(D/'STATUS.json',s)
deps=json.loads((B/'EXPORT.json').read_text())['dependencies'];mapping=json.loads((B/'CHECKPOINT_READBACKS.json').read_text())
dump(D/'NATIVE_RECOVERY_DEPENDENCIES.json',{'slim_checkpoint':json.loads((B/'CHECKPOINT_ARCHIVE_SAVED.json').read_text()),'dependencies':[dict(d,drive_file_id=mapping[d['sha256']]['id']) for d in deps],'restore_rule':'Fetch every dependency by Drive ID, verify exact SHA256/size, place as <sha>.bin; use existing Decoder preservation.restore_checkpoint. Local readback paths are historical host paths.'})
dump(D/'RECOVERY_DEPENDENCIES.json',{'predecessor0179':{'drive_file_id':'12yU6IDs2B9GwZvWXS-ddJAip9X72tnyc','sha256':'4174493f68e94ec000622e0b5cb2b95f9458e5a69a303584a0fc2f0c0066338a'},'master149':{'drive_file_id':'1lpkZRQYLLynJrq6K_TLbRScyZ8O-qH2G','sha256':'b919e3bfeeb2e11f6e07ad977f3355d8ce9475f95419b6d892cf2dbf017bcb6e'}})
dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0180_NATIVE_PUBLIC_PROJECTION_EXPORT','next_scope':v['next_scope'],'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'native_capture_completed':True,'native_execution_completed':True,'cold_reuse':True,'pending_bytes':0,'generation_calls':0,'new_admissions':0})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g_public_export0180/verify_saved.py. Native restore uses NATIVE_RECOVERY_DEPENDENCIES.json. Do not regenerate completed constructions.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER149_PUBLIC_PROJECTION_NATIVE_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig180_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_public_export0180/verify_saved.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==v
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m),'fresh_unpack':'PASS'};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER149_PUBLIC_PROJECTION_NATIVE_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0180_public_export/READ_FIRST.txt\n');print(json.dumps(meta))
