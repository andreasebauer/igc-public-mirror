from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
R=Path.cwd();B=R/'scope_reconcile0171';D=R/'handoff0171_scope_reconciliation';D.mkdir(exist_ok=True);sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=load(B/'RECOVERY_VALIDATION.json');assert v['new_root_states']==24 and v['survivor_component_records']==205
s=load(R/'CURRENT_STATUS.json');s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='SAVED_BYTES_AND_TYPED_BINDINGS_VERIFIED_EXPORT_NOT_PREPARED',scope_reconciliation='scope_reconcile0171/FAMILY_REGISTER.json',additional_O7_saved_recovery='scope_reconcile0171/RECOVERY_VALIDATION.json',code_mirror=load(B/'CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
shutil.copytree(B,D/B.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('SAVE_RECEIPT.json','DELIVERY_BINDINGS.json','FRESH_UNPACK_VERIFICATION.json'))
for folder in ['o7_export0169','o6_export0165','o5_export0162','o4_export0159','o3_export0156']:
 t=D/folder;t.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',t/'SCIENTIFIC_EXPORT.zip')
 if folder=='o7_export0169':shutil.copy2(R/folder/'EXPORT_BINDINGS.json',t/'EXPORT_BINDINGS.json');shutil.copytree(R/folder/'project',t/'project',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
t=D/'o7_readiness0168';t.mkdir(exist_ok=True)
for n in ['SELECTED.json','TWINS.json']:shutil.copy2(R/'o7_readiness0168'/n,t/n)
t=D/'o7_integrate0170';t.mkdir(exist_ok=True)
for n in ['CATALOG_0148.json','RELEASE_VERIFICATION.json','SCOPED_ADMISSION.json']:shutil.copy2(R/'o7_integrate0170'/n,t/n)
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0148','completed_action':'0171_SCOPE_RECONCILIATION_AND_ADDITIONAL_O7_RECOVERY','next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0,'full_l0_to_g8_complete':False});dump(D/'RECOVERY_DEPENDENCIES.json',{'master148':{'drive_file_id':'1pRZIf4KuJzajRIaruSGCUSxHXdmikNQA','sha256':'9269ae6ed7563a83e9beb59e918bd65c7e8a062d2100a5c73202e10d8e55b9f3'},'restore':'This compact bundle independently validates newly recovered O7 root/component rows. Restore master148 and its dependencies before native export/admission. Latest STATUS/cursor override ancestor prose. Original archive sources referenced by exact chain/hashes are not copied in full.'})
(D/'READ_FIRST.txt').write_text((B/'REPORT.txt').read_text()+'\nVerify MANIFEST.json then run pinned Python scope_reconcile0171/validate_recovery.py from unpack root.\n')
(D/'MANIFEST.json').unlink(missing_ok=True);m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',m);p=R/'IG_MASTER_0148_SCOPE_RECONCILIATION_O7_RECOVERY_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*m,'MANIFEST.json']:z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in m.items():assert sha(z.read(n))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta);(R/'IG_MASTER_0148_SCOPE_RECONCILIATION_O7_RECOVERY_START_2026-10-06.txt').write_text((B/'REPORT.txt').read_text()+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0171_scope_reconciliation/READ_FIRST.txt\n');print(json.dumps(meta))
