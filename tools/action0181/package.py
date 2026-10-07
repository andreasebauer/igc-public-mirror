from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g_public_integrate0181';D=R/'handoff0181_master150';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
shutil.copy2(R/'o7_additional_integrate0174/CATALOG_0149.json',B/'CATALOG_0149.json');shutil.copy2(R/'g_public_export0180/SCIENTIFIC_EXPORT.zip',B/'SCIENTIFIC_EXPORT.zip')
v=validate();dump(B/'VALIDATION.json',v)
reg=json.loads((R/'g_public_export0180/FAMILY_REGISTER.json').read_text());reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='G1_SAVED_PUBLIC_PROJECTION_SCOPED_ADMISSION_COMPLETE;G2_REPAIRED_S1_READINESS_NEXT;EXACT_DAG_BLOCKED'
for f in reg['families']:
 if f['family']=='G1':f.update(status='SAVED_PUBLIC_PROJECTION_SCOPED_ADMITTED_AND_UNIFIED_READER_INTEGRATED',scope='193 saved interfaces/192 observer classes/1351 reservations and continuation hashes',gap='Exact parent DAG and witnesses unavailable; projection admission is not exact carrier recovery')
dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_committed_release='MASTER_DATA_V1_0150',catalog_sha256=sha((B/'CATALOG_0150.json').read_bytes()),catalog_path='g_public_integrate0181/CATALOG_0150.json',scientific_slices=150,next_scope=v['next_scope'],next_scope_status='G1_PUBLIC_PROJECTION_ADMITTED;G2_REPAIRED_S1_EXPORT_READINESS_PENDING',scope_reconciliation='g_public_integrate0181/FAMILY_REGISTER.json',G1_public_projection_integration='g_public_integrate0181/NATIVE_RESULT.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['prepare.py','operate.py','cold_ops.py','validate.py','package.py','REPORT.txt','SPEC.json','SCOPED_ADMISSION.json','CATALOG_0149.json','CATALOG_0150.json','PREDECESSOR_SOURCE_HASHES.json','POINTER.json','NATIVE_RESULT.json','CHECKPOINT_ACK.json','EXPORT.json','FINAL_CHECKPOINT_SLIM.zip','CHECKPOINT_ARCHIVE_SAVED.json','COLD_RESTORE.json','COLD_REUSE_RESULT.json','SCIENTIFIC_EXPORT.zip','VALIDATION.json','FAMILY_REGISTER.json','CODE_MIRROR.json','READBACKS.json','CHECKPOINT_READBACKS.json']:shutil.copy2(B/n,t/n)
for p in (B/'project').rglob('*.py'):
 target=t/'project'/p.relative_to(B/'project');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
dump(D/'STATUS.json',s)
deps=json.loads((B/'EXPORT.json').read_text())['dependencies'];mapping=json.loads((B/'CHECKPOINT_READBACKS.json').read_text());dump(D/'NATIVE_RECOVERY_DEPENDENCIES.json',{'slim_checkpoint':json.loads((B/'CHECKPOINT_ARCHIVE_SAVED.json').read_text()),'dependencies':[dict(d,drive_file_id=mapping[d['sha256']]['id']) for d in deps],'restore_rule':'Fetch dependencies by Drive ID and verify SHA256/size; place as <sha>.bin; use Decoder preservation.restore_checkpoint. Historical local paths require rebinding.'})
dump(D/'RECOVERY_DEPENDENCIES.json',{'predecessor0180':{'drive_file_id':'1AyRj9_M0f8cfPqzkwM508Had2SLT9hh3','sha256':'9904237a8dea2a747d6ba7d0ed84c6922ca18b2377ddf1efcdc545612ee0be5e'},'master149':{'drive_file_id':'1lpkZRQYLLynJrq6K_TLbRScyZ8O-qH2G','sha256':'b919e3bfeeb2e11f6e07ad977f3355d8ce9475f95419b6d892cf2dbf017bcb6e'}})
dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0181_MASTER150_G1_PROJECTION_INTEGRATION','next_scope':v['next_scope'],'master_release':'MASTER_DATA_V1_0150','scientific_slices':150,'cold_reuse':True,'pending_bytes':0,'generation_calls':0,'new_admissions':1,'admission_scope':'SAVED_PUBLIC_PROJECTION_ONLY'})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g_public_integrate0181/validate.py. Native restore uses NATIVE_RECOVERY_DEPENDENCIES.json.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER_DATA_V1_0150_G1_PROJECTION_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig181_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_public_integrate0181/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==v
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m),'fresh_unpack':'PASS'};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER_DATA_V1_0150_G1_PROJECTION_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0181_master150/READ_FIRST.txt\n');print(json.dumps(meta))
