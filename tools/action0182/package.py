from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g2_s1_readiness0182';D=R/'handoff0182_s1_readiness';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=validate();dump(B/'VALIDATION.json',v)
reg=json.loads((R/'g_public_integrate0181/FAMILY_REGISTER.json').read_text());reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='G1_PROJECTIONS_ADMITTED;REPAIRED_G2_S1_BYTE_ONLY_EXPORT_SCOPE_FROZEN;REGISTRATION_PENDING;EXACT_ANCESTRY_OPEN';dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='SAVED_G2_S1_READINESS_PASS;RECORD_READER_REGISTRATION_PENDING',scope_reconciliation='g2_s1_readiness0182/FAMILY_REGISTER.json',G2_S1_export_readiness='g2_s1_readiness0182/VALIDATION.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['prepare.py','validate.py','package.py','REPORT.txt','EXPORT_CONTRACT.json','INPUT_PINS.json','SOURCE_BINDINGS.json','FIRST_ROW_BINDING_PREVIEW.json','VALIDATION.json','FAMILY_REGISTER.json','CODE_MIRROR.json']:shutil.copy2(B/n,t/n)
shutil.copytree(B/'inputs',t/'inputs',dirs_exist_ok=True)
dump(D/'STATUS.json',s)
dump(D/'RECOVERY_DEPENDENCIES.json',{'master150':{'drive_file_id':'1cXd6JsBjH607vtTZ2Pb6AY4QNfdiOf7L','sha256':'f2b015c33d50385eb21cb0bfb28849a29d0695b8596ebee9a09f8d4219c7b9f2'},'source0177':{'drive_file_id':'1tWHCC1B2v0IkV5qQ8os6OXEdaPp-Ebb0','sha256':'2c9b0c3ed6cf07cf91b8974e17ad62e63c6739da834f8ce34ef5f1d8b3f1515d'},'full_saved_source':'g2_s1_readiness0182/inputs/TRANSPORT.json (concatenate ordered parts,verify whole hash) or whole archive Library ID in SOURCE_BINDINGS.json'})
dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0182_SAVED_G2_S1_EXPORT_READINESS','next_scope':v['next_scope'],'master_release':'MASTER_DATA_V1_0150','scientific_slices':150,'generation_calls':0,'new_admissions':0,'pending_bytes':0,'native_reader_executed':False})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g2_s1_readiness0182/validate.py. Do not regenerate completed constructions.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER150_G2_S1_READINESS_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig182_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g2_s1_readiness0182/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==v
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m),'fresh_unpack':'PASS'};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER150_G2_S1_READINESS_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0182_s1_readiness/READ_FIRST.txt\n');print(json.dumps(meta))
