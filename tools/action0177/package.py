from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys,tempfile,zipfile
from validate import validate
R=Path.cwd();B=R/'g_repair0177';D=R/'handoff0177_s1_repair';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=json.loads((B/'VALIDATION.json').read_bytes());base=validate();assert {k:v[k] for k in base}==base
assert v['stream_validation']['record_rows_checked']==580351 and v['stream_validation']['stored_observer_split_classes']==0
reg=json.loads((R/'g_closure0176/FAMILY_REGISTER.json').read_bytes())
for f in reg['families']:
 if f['family']=='G2':f.update(status='SAVED_REPAIRED_S1_AND_STORED_OBSERVER_CONGRUENCE_VERIFIED_EXACT_INPUT_PENDING',scope='580351 repaired D4+Q2 records;576785 classes;0 stored observer-signature splits. Historical S2 pair quotient authorization recovered',gap='Q2 payloads not serialized;exact G1 DAG/witnesses pending. Stored signatures not fresh exact-carrier realization. Historical G2 certificate remains separate')
reg['next_scope']=v['next_scope'];reg['work_packages']['WP6']='SAVED_G1_PUBLIC_AND_REPAIRED_S1_SCOPES_VERIFIED;EXACT_DAG_AND_EXPORT_LOCAL_SUFFICIENCY_PENDING';dump(B/'FAMILY_REGISTER.json',reg)
s=json.loads((R/'CURRENT_STATUS.json').read_bytes());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='REPAIRED_SAVED_SCOPE_VERIFIED_EXACT_DAG_AND_EXPORT_SUFFICIENCY_PENDING',scope_reconciliation='g_repair0177/FAMILY_REGISTER.json',G2_S1_repair_recovery='g_repair0177/VALIDATION.json',code_mirror=json.loads((B/'CODE_MIRROR.json').read_bytes()));dump(R/'CURRENT_STATUS.json',s)
t=D/B.name;t.mkdir(exist_ok=True)
for n in ['recover.py','validate.py','package.py','REPORT.txt','SOURCE_PINS.json','INPUT_PINS.json','VALIDATION.json','TRANSPORT.json','SOURCE_READBACKS.json','FAMILY_REGISTER.json','CODE_MIRROR.json']:shutil.copy2(B/n,t/n)
shutil.copytree(B/'inputs',t/'inputs',dirs_exist_ok=True)
dump(D/'RECOVERY_DEPENDENCIES.json',{'predecessor0176':{'drive_file_id':'1h0p4YbVZZoctTFU2oRgJNw9peuq-jano','sha256':'4e47c062ce227526b5883dac7314a0d9edb0b6330a3c2f2d18dcd36b57aa3d44'},'master149':{'drive_file_id':'1lpkZRQYLLynJrq6K_TLbRScyZ8O-qH2G','sha256':'b919e3bfeeb2e11f6e07ad977f3355d8ce9475f95419b6d892cf2dbf017bcb6e'},'D4_source':{'drive_file_id':'12hZ_N3Aq_iQZv_1He7Ih-h2y64XU94Iw','library_file_id':'libfile_74c6c800891c8191a7914a222fd4bbbc'},'recon_source':{'drive_file_id':'1Yy0gx6jYToQr0lUQO32oA9njgm3VqxF8','library_file_id':'libfile_5c085e019a2881918de7735baba3fb84'},'reaudit_source':{'library_file_id':'libfile_31daf44b95d88191a02cebcc4aa03078','Drive_transport':'g_repair0177/TRANSPORT.json:concatenate five parts in listed order,verify archive hash before opening'},'stream_restore':'Full record/audit streams stay in saved source archive;compact inputs permit independent small binding checks. No new producer execution.'})
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'completed_action':'0177_SAVED_S1_REPAIR_RECOVERY','master_release':'MASTER_DATA_V1_0149','scientific_slices':149,'next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0,'pending_bytes':0,'next_job_captured':False})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run python3 g_repair0177/validate.py. Optional full saved-stream verification:restore137MB source using TRANSPORT.json,pass its local ZIP path to validator. Do not execute original scientific producers.\n')
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file() and p!=D/'MANIFEST.json'};dump(D/'MANIFEST.json',m)
out=R/'IG_MASTER149_S1_REPAIR_RECOVERY_HANDOFF_2026-10-07.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with tempfile.TemporaryDirectory(prefix='ig177_fresh_') as td:
 with zipfile.ZipFile(out) as z:
  for n,x in m.items():assert sha(z.read(n))==x['sha256'] and len(z.read(n))==x['bytes']
  z.extractall(td)
 p=subprocess.run([sys.executable,str(Path(td)/'g_repair0177/validate.py')],capture_output=True,text=True,check=True);assert json.loads(p.stdout)==base
dump(B/'FRESH_UNPACK_VERIFICATION.json',{'status':'PASS','manifest_files':len(m),'small_binding_checks':'PASS','fullstream_checks':'Verified against exact saved archive before packaging;not repeated during compact unpack'})
meta={'sha256':sha(out.read_bytes()),'bytes':out.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER149_S1_REPAIR_RECOVERY_START_2026-10-07.txt').write_text((D/'READ_FIRST.txt').read_text()+'\nBundle:'+out.name+'\nSHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0177_s1_repair/READ_FIRST.txt\n');print(json.dumps(meta))
