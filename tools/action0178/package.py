from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g_sufficiency0178';D=R/'handoff0178_local_sufficiency';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=validate();assert v==json.loads((B/'VALIDATION.json').read_text())
reg=json.loads((R/'g_repair0177/FAMILY_REGISTER.json').read_text())
reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='LOCAL_SUFFICIENCY_PASS_SAVED_PUBLIC_PROJECTION;EXACT_PARENT_DAG_BLOCKED'
dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='REGISTER_FINITE_BYTE_ONLY_EXPORT;EXACT_G1_DAG_RECOVERY_REMAINS_OPEN',scope_reconciliation='g_sufficiency0178/FAMILY_REGISTER.json',G1_local_sufficiency='g_sufficiency0178/LOCAL_SUFFICIENCY_GATE.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['validate.py','package.py','REPORT.txt','INPUT_PINS.json','VALIDATION.json','LOCAL_SUFFICIENCY_GATE.json','FAMILY_REGISTER.json','CODE_MIRROR.json']:shutil.copy2(B/n,t/n)
for n in ['sources','inputs']:shutil.copytree(B/n,t/n,dirs_exist_ok=True)
dump(D/'STATUS.json',s)
dump(D/'RECOVERY_DEPENDENCIES.json',{'predecessor0177':{'drive_file_id':'1tWHCC1B2v0IkV5qQ8os6OXEdaPp-Ebb0','sha256':'2c9b0c3ed6cf07cf91b8974e17ad62e63c6739da834f8ce34ef5f1d8b3f1515d'},'master149':{'drive_file_id':'1lpkZRQYLLynJrq6K_TLbRScyZ8O-qH2G','sha256':'b919e3bfeeb2e11f6e07ad977f3355d8ce9475f95419b6d892cf2dbf017bcb6e'},'S4_restart_original_library_file_id':'libfile_cd9644b6c8f88191b4c8c18f0c28fac7','S4_closeout_original_library_file_id':'libfile_24aacaa75080819189175271d6361321','original_capsules_included':'g_sufficiency0178/sources'})
dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0178_LOCAL_SUFFICIENCY_GATE','next_scope':v['next_scope'],'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'pending_bytes':0,'generation_calls':0,'new_admissions':0})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g_sufficiency0178/validate.py. Never execute source producer scripts.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER149_LOCAL_SUFFICIENCY_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig178_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_sufficiency0178/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==v
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m),'fresh_unpack':'PASS'};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER149_LOCAL_SUFFICIENCY_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n')
(R/'CURRENT_START.txt').write_text('handoff0178_local_sufficiency/READ_FIRST.txt\n');print(json.dumps(meta))
