from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
R=Path.cwd();B=R/'o5_readiness0161';load=lambda p:json.loads(p.read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest();dump=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');v=load(B/'READINESS_VALIDATION.json');assert v['status']=='SAVED_O5_SCOPE_SOURCE_BOUND_READY'
contract={'schema':'IG_SAVED_TYPED_O5_EXPORT_CONTRACT_V1','states':v['counts'],'identity':'lane/m5/edge-repr digest plus full canonical JSON payload SHA256; no global isomorphism claim','required':['literal selected rows and owner order','all typed 10-coordinate E5 edges, parallel multiplicity','seven exact O4 component/root bindings and root payload hashes','base typed E4 edges and nested O3 root/component references','ordered nested resource coordinates and E5 reservations','original spec, selector, features, algebra, templates, producer and source manifest','component incidence/accounting and explicit unavailable parent/action ancestry'],'master_catalog_sha256':v['catalog_sha256'],'generation_calls':0,'new_admissions':0,'microscopic_O2_source_closed':False,'missing_O2_panels':76}
dump(B/'EXPORT_CONTRACT.json',contract)
report='''INFINITY GRID SAVED O5 SOURCE-BOUND READINESS — ACTION0161 — 2026-10-06

MASTER_DATA_V1_0145 retained,145 scientific slices. Verified134 saved O5 selected states:132 nonseed states and2 external rank0 seeds;432 typed E5 edges,682 O4 owner occurrences and169 nontrivial component occurrences. HET4 m5=0..5,HOM6 m5=0..6. All six selected O4 prototypes plus pinned actual component resolve to admitted O4 roots with exact ordered nested resources, normalized typed E4 incidence and explicit O3 component/root links. Original selector and recovered source hashes pass. All saved E5 coordinates, compatibility, capacity, digest and component accounting pass.

No construction generation and no new scientific admission. Full literal payload hashes supplement lane/rank/edge-only digest; no global canonical-isomorphism claim. Complete parent/action occurrence ancestry is unavailable. Inherited anonymous Qbank boundary and76 missing microscopic O2 panels remain unresolved. Historical graduation evidence is not a fresh census or graduation. Full L0-G8 completion remains false.

NEXT:WP5_SAVED_O5_TYPED_EXPORT_AND_READER. Implement finite lossless source-bound export and reader under EXPORT_CONTRACT.json; capture, save/readback, checkpoint and cold exact reuse before scoped admission. Do not regenerate completed constructions.
''';(B/'REPORT.txt').write_text(report)
s=load(R/'CURRENT_STATUS.json');s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope='WP5_SAVED_O5_TYPED_EXPORT_AND_READER',next_scope_status='NOT_PREPARED',o5_readiness='o5_readiness0161/READINESS_VALIDATION.json',o5_export_contract='o5_readiness0161/EXPORT_CONTRACT.json',code_mirror=load(B/'CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
D=R/'handoff0161_o5_readiness';D.mkdir(exist_ok=True);shutil.copytree(B,D/B.name,dirs_exist_ok=True)
for folder in ['o4_export0159','o3_export0156']:
 target=D/folder;target.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',target/'SCIENTIFIC_EXPORT.zip')
 if folder=='o4_export0159':
  shutil.copy2(R/folder/'EXPORT_BINDINGS.json',target/'EXPORT_BINDINGS.json');shutil.copytree(R/folder/'project',target/'project',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
target=D/'o4_integrate0160';target.mkdir(exist_ok=True);shutil.copy2(R/'o4_integrate0160/CATALOG_0145.json',target/'CATALOG_0145.json')
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0145','completed_action':'0161_O5_SOURCE_BOUND_READINESS','next_scope':s['next_scope'],'pending_bytes':0,'generation_calls':0,'new_admissions':0})
dump(D/'RECOVERY_DEPENDENCIES.json',{'master145':{'drive_file_id':'1jG069nObAHMoSJHnXmGUeM9KjIuzn7fZ','sha256':'5fc9acdf7f7b4d62fc07467924d98a30d1c188cab3d5d6b5aecca4919643c7ac'},'higher_O_sources':{'drive_file_id':'1Q1flImNUfUsqjeDAaguySy8FyrTL05-J','sha256':'90d2d33536a5263ad0dc9e832745c703089dde96f3561a585db27456ca40aeac'},'restore':'This compact bundle independently rechecks O5 readiness. Restore master145 and its pinned predecessor dependencies for native integration. Latest STATUS/cursor override ancestor prose.'})
(D/'READ_FIRST.txt').write_text(report+'\nVerify MANIFEST.json then run pinned Python o5_readiness0161/validate.py.\n');(D/'MANIFEST.json').unlink(missing_ok=True)
manifest={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',manifest)
p=R/'IG_O5_SOURCE_BOUND_READINESS_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*manifest,'MANIFEST.json']:z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in manifest.items():assert sha(z.read(n))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(manifest)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_O5_SOURCE_BOUND_READINESS_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0161_o5_readiness/READ_FIRST.txt\n');print(json.dumps(meta))
