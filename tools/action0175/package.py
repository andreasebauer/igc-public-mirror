from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
R=Path.cwd();B=R/'g_readiness0175';D=R/'handoff0175_g_readiness';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=json.loads((B/'READINESS_VALIDATION.json').read_bytes());assert v['generation_calls']==v['new_admissions']==0
s=json.loads((R/'CURRENT_STATUS.json').read_bytes());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='SAVED_R100_EVIDENCE_BOUND_EXACT_CARRIER_CLOSURE_PENDING',scope_reconciliation='g_readiness0175/FAMILY_REGISTER.json',G1_G2_readiness='g_readiness0175/READINESS_VALIDATION.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_bytes()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['validate.py','prepare.py','package.py','inspect_saved.py','REPORT.txt','READINESS_VALIDATION.json','FAMILY_REGISTER.json','INPUT_PINS.json','CODE_MIRROR.json','SAVED_SOURCE_SEARCH.json']:
 shutil.copy2(B/n,t/n)
shutil.copytree(B/'inputs',t/'inputs',dirs_exist_ok=True)
dump(D/'STATUS.json',s)
dump(D/'RECOVERY_DEPENDENCIES.json',{'master149':{'drive_file_id':'1lpkZRQYLLynJrq6K_TLbRScyZ8O-qH2G','sha256':'b919e3bfeeb2e11f6e07ad977f3355d8ce9475f95419b6d892cf2dbf017bcb6e'},'runtime':'Use saved master149 runtime transport/bootstrap;this byte-only validator also runs with standard Python3.12+','authority':'Ancestor completed constructions must not be regenerated. This handoff independently verifies readiness inputs;fullmaster149 native reader/checkpoint remains in pinned predecessor bundle.'})
dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'completed_action':'0175_WP5_WP6_READINESS','next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0,'pending_bytes':0,'next_job_captured':False,'full_l0_to_g8_complete':False})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json, then run python3 g_readiness0175/validate.py from unpack root. No producer imports or generation.\n')
manifest={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p.name!='MANIFEST.json'};dump(D/'MANIFEST.json',manifest)
out=R/'IG_MASTER149_WP5_WP6_READINESS_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*manifest,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig175_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,p in manifest.items():assert sha(z.read(n))==p['sha256'] and len(z.read(n))==p['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_readiness0175/validate.py')],capture_output=True,text=True,check=True)
 assert json.loads(p.stdout)==v
dump(B/'FRESH_UNPACK_VERIFICATION.json',{'status':'PASS','independent_readiness_validator':'PASS','manifest_files':len(manifest)})
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(manifest)};dump(B/'DELIVERY_BINDINGS.json',meta)
start=R/'IG_MASTER149_WP5_WP6_READINESS_START_2026-10-06.txt';start.write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n')
(R/'CURRENT_START.txt').write_text('handoff0175_g_readiness/READ_FIRST.txt\n');print(json.dumps(meta))
