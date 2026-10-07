from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g_public_register0179';D=R/'handoff0179_public_registration';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=validate();assert v==json.loads((B/'VALIDATION.json').read_text())
assert json.loads((B/'NATIVE_PREFLIGHT.json').read_text())['status']=='PASS_NATIVE_CONTRACT_NORMALIZATION_AND_PROJECT_CLOSURE'
reg=json.loads((R/'g_sufficiency0178/FAMILY_REGISTER.json').read_text());reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='FINITE_PUBLIC_PROJECTION_CONTRACT_AND_READER_FROZEN;CAPTURE_PENDING;EXACT_DAG_BLOCKED';dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='CONTRACT_AND_READER_PREFLIGHT_PASS;NATIVE_CAPTURE_PENDING',scope_reconciliation='g_public_register0179/FAMILY_REGISTER.json',G1_public_export_registration='g_public_register0179/PROJECT_REGISTRATION.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['prepare.py','validate.py','package.py','REPORT.txt','INPUT_BINDINGS.json','VALIDATION.json','EXPORT_CONTRACT.json','SPEC.json','PROJECT_REGISTRATION.json','NATIVE_PREFLIGHT.json','FAMILY_REGISTER.json','CODE_MIRROR.json']:shutil.copy2(B/n,t/n)
shutil.copytree(B/'inputs',t/'inputs',dirs_exist_ok=True)
(t/'project').mkdir(exist_ok=True)
for p in (B/'project').glob('*.py'):shutil.copy2(p,t/'project'/p.name)
dump(D/'STATUS.json',s)
dump(D/'RECOVERY_DEPENDENCIES.json',{'predecessor0178':{'drive_file_id':'1wIZoZ69XnmfLXwmad_1vSKlrFEovfiOS','sha256':'022e7b14dc22727eef61b041e0d9134a137f62caffb7cfdcad2fa81a1479e973'},'master149':{'drive_file_id':'1lpkZRQYLLynJrq6K_TLbRScyZ8O-qH2G','sha256':'b919e3bfeeb2e11f6e07ad977f3355d8ce9475f95419b6d892cf2dbf017bcb6e'},'native_capture_restore':'Master149 includes engine/project/runtime dependencies. Restore pinned engine/runtime before native capture. Archived SPEC paths require rebinding in a new workspace.'})
dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0179_PUBLIC_PROJECTION_EXPORT_REGISTRATION','next_scope':v['next_scope'],'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'native_capture_completed':False,'pending_bytes':0,'generation_calls':0,'new_admissions':0})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g_public_register0179/validate.py. Do not call the native handler directly.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER149_PUBLIC_PROJECTION_REGISTRATION_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig179_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_public_register0179/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==v
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m),'fresh_unpack':'PASS'};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER149_PUBLIC_PROJECTION_REGISTRATION_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0179_public_registration/READ_FIRST.txt\n');print(json.dumps(meta))
