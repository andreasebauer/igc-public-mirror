from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g2_s1_register0183';D=R/'handoff0183_s1_registration';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=validate();dump(B/'VALIDATION.json',v)
assert json.loads((B/'NATIVE_PREFLIGHT.json').read_text())['status']=='PASS_NATIVE_PROJECT_CLOSURE_RESULT_CONTRACT_AND_ARCHITECTURE'
reg=json.loads((R/'g2_s1_readiness0182/FAMILY_REGISTER.json').read_text());reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='G2_S1_RECORD_READER_AND_BYTE_INDEX_EXPORT_REGISTERED;NATIVE_CAPTURE_PENDING;EXACT_ANCESTRY_OPEN';dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='SAVED_G2_S1_INDEX_PROJECT_PREFLIGHT_PASS;NATIVE_CAPTURE_PENDING',scope_reconciliation='g2_s1_register0183/FAMILY_REGISTER.json',G2_S1_index_registration='g2_s1_register0183/PROJECT_REGISTRATION.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['validate.py','package.py','REPORT.txt','EXPORT_CONTRACT.json','SPEC.json','PROJECT_REGISTRATION.json','INPUT_PINS.json','NATIVE_PREFLIGHT.json','VALIDATION.json','FAMILY_REGISTER.json','CODE_MIRROR.json']:shutil.copy2(B/n,t/n)
shutil.copytree(B/'inputs',t/'inputs',dirs_exist_ok=True)
(t/'project').mkdir(exist_ok=True)
for p in (B/'project').glob('*.py'):shutil.copy2(p,t/'project'/p.name)
dump(D/'STATUS.json',s)
dump(D/'RECOVERY_DEPENDENCIES.json',{'predecessor0182':{'drive_file_id':'1p4M-QCJ4l41Oo1NzrYjSEXBe8KHdMYad','sha256':'bb1888686c7b039ac09935a571f2aa370ddf635c2f24faa212e338c4c4d7406c'},'master150':{'drive_file_id':'1cXd6JsBjH607vtTZ2Pb6AY4QNfdiOf7L','sha256':'f2b015c33d50385eb21cb0bfb28849a29d0695b8596ebee9a09f8d4219c7b9f2'},'full_source_parts':'g2_s1_register0183/inputs/TRANSPORT.json','new_workspace_rule':'Restore source parts by saved IDs/hash; rebind historical absolute SPEC input/engine/project/environment paths and capture a fresh exact specification. Preserve this historical SPEC.'})
dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0183_SAVED_G2_S1_INDEX_EXPORT_REGISTRATION','next_scope':v['next_scope'],'master_release':'MASTER_DATA_V1_0150','scientific_slices':150,'generation_calls':0,'new_admissions':0,'pending_bytes':0,'native_handler_called':False,'full_index_built':False})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g2_s1_register0183/validate.py. Do not invoke handler directly.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER150_G2_S1_INDEX_REGISTRATION_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig183_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g2_s1_register0183/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==v
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m),'fresh_unpack':'PASS'};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER150_G2_S1_INDEX_REGISTRATION_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0183_s1_registration/READ_FIRST.txt\n');print(json.dumps(meta))
