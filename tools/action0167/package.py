from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
R=Path.cwd();B=R/'o7_readiness0167';D=R/'handoff0167_o7_recovery';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=load(B/'RECOVERY_VALIDATION.json')
report='''INFINITY GRID O7 SAVED SCOPE RECOVERY — ACTION0167 — 2026-10-06

Master0147 retained,147 scientific slices. Recovered74 unique literal O7 saved states:72 nonseed and2 rank0 seeds. HET4 m7=0..5,HOM6 m7=0..1;192 typed E7 edges.87 source occurrences agree on duplicate payloads. Recovered363 partial closeout selection records; their payload hashes pass. The older progress snapshot reports320 completed/1730 pending of2050 and is preserved as historical evidence, not an updated counter.

Recovered source bytes, checkpoint payload hashes, binary E7 digests and duplicate row agreement pass. This is recovery only: O6 owner/resource bindings, E7 legality and accounting still require independent verification. Later COMPLETE campaign snapshots do not supply full final survivor/profile coverage; closeout debts D1-D6 remain. No construction generation, no new scientific admission, no O7 graduation or automatic O8 authorization. Full L0-G8 completion remains false.

NEXT:WP5_SAVED_O7_RECOVERED_SCOPE_BINDING_VALIDATION. Resolve literal O6 owner order against admitted O6 components; validate E7 capacities/accounting and distinguish adversarial external twin roots. Do not regenerate completed constructions.
'''
(B/'REPORT.txt').write_text(report)
s=load(R/'CURRENT_STATUS.json');s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='NOT_PREPARED',o7_recovery='o7_readiness0167/RECOVERY_VALIDATION.json',code_mirror=load(B/'CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
shutil.copytree(B,D/B.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('SAVE_RECEIPT.json','DELIVERY_BINDINGS.json'))
for folder in ['o6_export0165','o5_export0162','o4_export0159','o3_export0156']:
 t=D/folder;t.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',t/'SCIENTIFIC_EXPORT.zip')
 if folder=='o6_export0165':shutil.copytree(R/folder/'project',t/'project',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
t=D/'o6_integrate0166';t.mkdir(exist_ok=True);shutil.copy2(R/'o6_integrate0166/CATALOG_0147.json',t/'CATALOG_0147.json')
# Preserve previously recovered O7 preregistration and closeout sources verbatim.
for p in (R/'o_readiness0158/sources').iterdir():
 if 'O7_' in p.name:shutil.copytree(p,D/'o_readiness0158/sources'/p.name,dirs_exist_ok=True)
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0147','completed_action':'0167_O7_LITERAL_SCOPE_RECOVERY','next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0})
dump(D/'RECOVERY_DEPENDENCIES.json',{'master147':{'drive_file_id':'1EcDvoG2BozFXbLN08T1fSmZoMF8rzs34','sha256':'9f3d5bb4b413ebb6dfecacd3f5adc53535caf5869af417a836665d9b6e338b34'},'higher_O_sources':{'drive_file_id':'1Q1flImNUfUsqjeDAaguySy8FyrTL05-J','sha256':'90d2d33536a5263ad0dc9e832745c703089dde96f3561a585db27456ca40aeac'},'restore':'Overlay this compact recovery bundle after master147 dependencies. Latest STATUS/cursor override ancestor prose. Full original attachment is not bundled; recovered member hashes and archive locators are included.'})
(D/'READ_FIRST.txt').write_text(report+'\nVerify MANIFEST.json,then run python o7_readiness0167/verify_saved.py from the unpack root.\n')
(D/'MANIFEST.json').unlink(missing_ok=True);m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',m)
p=R/'IG_O7_SAVED_SCOPE_RECOVERY_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*m,'MANIFEST.json']:z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in m.items():assert sha(z.read(n))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_O7_SAVED_SCOPE_RECOVERY_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0167_o7_recovery/READ_FIRST.txt\n');print(json.dumps(meta))
