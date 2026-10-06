from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
R=Path.cwd();B=R/'o7_additional_readiness0172';D=R/'handoff0172_o7_additional_readiness'
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=load(B/'READINESS_VALIDATION.json');assert v['status']=='FINITE_ADDITIONAL_O7_EXPORT_SCOPE_READY'
s=load(R/'CURRENT_STATUS.json');s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='FINITE_EXPORT_CONTRACT_AND_EXACT_BINDINGS_READY',additional_O7_readiness='o7_additional_readiness0172/READINESS_VALIDATION.json',additional_O7_export_contract='o7_additional_readiness0172/EXPORT_CONTRACT.json',code_mirror=load(B/'CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
shutil.copytree(R/'handoff0171_scope_reconciliation',D,dirs_exist_ok=True)
shutil.copytree(B,D/B.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('SAVE_RECEIPT.json','DELIVERY_BINDINGS.json','FRESH_UNPACK_VERIFICATION.json'))
t=D/'o7_readiness0168';t.mkdir(exist_ok=True)
for n in ['O6_BINDINGS.json','EXTERNAL_TWIN_BINDINGS.json','READINESS_VALIDATION.json']:shutil.copy2(R/'o7_readiness0168'/n,t/n)
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0148','completed_action':'0172_ADDITIONAL_O7_EXPORT_READINESS','next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0,'full_l0_to_g8_complete':False})
report='Action0172 complete: finite additional saved O7 export scope ready.\nMaster remains MASTER_DATA_V1_0148; no new admission or generator calls.\n24 whole HOM6 roots (ranks2 and3),60 E7 edges,144 owner occurrences.\n205 component occurrences,164 exact contextual objects,512 edge occurrences,533 owner occurrences.\nFour admitted O6 prototype bindings and two explicitly external adversarial twins.\n96 available and68 unavailable source whole-root references; no missing forests synthesized.\nSaved profile summaries retained as source evidence; full Counter/fresh certification unavailable.\nNext: registered standalone/native export, exact checkpoint and cold replay, then scoped admission.\nDo not regenerate completed constructions or run original producer mains.\nRestore master148 dependency chain from RECOVERY_DEPENDENCIES.json for native execution.\nFor independent checks: verify MANIFEST.json and run pinned Python o7_additional_readiness0172/prepare.py from unpack root; compare readiness output pins.\n'
(D/'READ_FIRST.txt').write_text(report);(B/'REPORT.txt').write_text(report)
(D/'MANIFEST.json').unlink(missing_ok=True)
m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',m)
p=R/'IG_MASTER_0148_ADDITIONAL_O7_EXPORT_READINESS_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in sorted([*m,'MANIFEST.json']):z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in m.items():assert sha(z.read(n))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_MASTER_0148_ADDITIONAL_O7_EXPORT_READINESS_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n')
(R/'CURRENT_START.txt').write_text('handoff0172_o7_additional_readiness/READ_FIRST.txt\n');print(json.dumps(meta))
