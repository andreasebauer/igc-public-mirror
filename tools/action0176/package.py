from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g_closure0176';D=R/'handoff0176_g1_public_recovery';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=json.loads((B/'VALIDATION.json').read_bytes());base,recipes,rows=validate();assert {k:v[k] for k in base}==base
assert v['stream_validation']['rows_checked']==580351 and v['stream_validation']['status']=='PASS_PROJECTED_SEMANTICS_ONLY'
assert json.loads((B/'CANDIDATE_RECIPES.json').read_bytes())['recipes']==recipes
reg=json.loads((R/'g_readiness0175/FAMILY_REGISTER.json').read_bytes())
for f in reg['families']:
 if f['family']=='G1':f.update(status='SAVED193_PUBLIC_INTERFACES_AND_RECIPE_DEFINITIONS_BOUND_EXACT_DAG_PENDING',scope='193 saved interfaces/192 classes;1351 reservation rows;193 recipe definitions matched historical AST and saved R100 motif IDs',gap='Exact parent DAG,nested witness closure and resource skins from exact carriers unavailable;public saved projection is not full exact carrier admission')
 if f['family']=='G2':f.update(status='SAVED_PROJECTED_S1_VERIFIED_LATER_FAILURE_RETAINED_REPAIRED_SUCCESSOR_PENDING',scope='580351 saved projected records/31 operators checked;later historical realization FAIL has4844 split classes. Later G2 historical graduation remains separately pinned',gap='Recover repaired S1 successor and exact G1 parent/DAG evidence before fresh export/admission;do not reuse failed projected quotient')
reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='SAVED_G1_PUBLIC_SCOPE_RECOVERED;EXACT_DAG_AND_REPAIRED_G2_SCOPE_BINDING_PENDING';dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_bytes());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='SAVED_PUBLIC_INTERFACES_VERIFIED_REPAIRED_SUCCESSOR_AND_EXACT_DAG_PENDING',scope_reconciliation='g_closure0176/FAMILY_REGISTER.json',G1_public_recovery='g_closure0176/VALIDATION.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_bytes()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['recover.py','validate.py','package.py','REPORT.txt','INPUT_PINS.json','VALIDATION.json','CANDIDATE_RECIPES.json','SOURCE_RECEIPTS.json','FAMILY_REGISTER.json','CODE_MIRROR.json']:shutil.copy2(B/n,t/n)
shutil.copytree(B/'inputs',t/'inputs',dirs_exist_ok=True)
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0176_SAVED_G1_PUBLIC_SCOPE_RECOVERY','master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0,'pending_bytes':0,'next_job_captured':False})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g_closure0176/validate.py. Optional fullstream recheck:fetch hash-pinned audit_source and pass its local ZIP path to that script. No scientific producer import. Restore master149 from0175 pinned predecessor dependencies for later native work.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p.name!='MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER149_G1_PUBLIC_RECOVERY_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig176_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_closure0176/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==base
dump(B/'FRESH_UNPACK_VERIFICATION.json',{'status':'PASS','manifest_files':len(m),'public_scope_validation':'PASS','fullstream_validation':'Already checked against pinned saved source;not rerun during compact unpack'})
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER149_G1_PUBLIC_RECOVERY_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0176_g1_public_recovery/READ_FIRST.txt\n');print(json.dumps(meta))
