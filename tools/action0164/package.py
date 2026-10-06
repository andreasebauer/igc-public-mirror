from pathlib import Path
import json,hashlib,shutil,zipfile,datetime
R=Path.cwd();B=R/'o6_readiness0164';load=lambda p:json.loads(p.read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest();dump=lambda p,x:p.write_text(json.dumps(x,indent=2)+'\n');v=load(B/'READINESS_VALIDATION.json');assert v['status']=='SAVED_O6_SCOPE_SOURCE_BOUND_READY'
contract={'schema':'IG_SAVED_TYPED_O6_EXPORT_CONTRACT_V1','states':v['counts'],'identity':'lane/m6/binary edge digest plus full canonical JSON payload SHA256; no global isomorphism claim','state_digest_encoding':'SHA256(prefix IG-E6-STATE-v1| plus sorted edges encoded as twelve unsigned 32-bit big-endian integers)','prototype_alias':'HOM6 uses PHASE0_PINNED_ACTUAL_O5_COMPONENT; canonical pool ID is explicitly bound in readiness','required':['literal selected rows and owner order','all typed 12-coordinate E6 edges, parallel multiplicity','selected and pinned exact O5 component/root bindings and root payload hashes','base typed E5 edges and nested O4/O3 root/component references','ordered nested resource coordinates and E6 reservations','original full132-entry prototype pool, pin, spec, selector, features, algebra, templates, producer and source manifest','component incidence/accounting and explicit unavailable parent/action ancestry'],'master_catalog_sha256':v['catalog_sha256'],'generation_calls':0,'new_admissions':0,'microscopic_O2_source_closed':False,'missing_O2_panels':76}
dump(B/'EXPORT_CONTRACT.json',contract)
report='''INFINITY GRID SAVED O6 SOURCE-BOUND READINESS — ACTION0164 — 2026-10-06

MASTER_DATA_V1_0146 retained,146 scientific slices. Verified134 saved O6 selected states:132 nonseed states and2 external rank0 seeds;432 typed E6 edges,682 O5 owner occurrences and172 nontrivial component occurrences. HET4 m6=0..5,HOM6 m6=0..6. All132 O5 prototype-pool entries plus the pinned actual component resolve to admitted O5 roots with exact ordered nested resources, normalized typed E5 incidence and explicit O4 component/root links. The169 admitted O5 component occurrences yield the same132 R5 resource classes. Original selector and recovered source hashes pass. Binary digest encoding and the HOM6 pinned alias are preserved. All saved E6 coordinates, compatibility, capacity, digest and component accounting pass.

No construction generation and no new scientific admission. Full literal payload hashes supplement lane/rank/binary edge-only digest; no global canonical-isomorphism claim. Complete parent/action occurrence ancestry is unavailable. Inherited anonymous Qbank boundary and76 missing microscopic O2 panels remain unresolved. Historical graduation evidence is not a fresh census or graduation. Full L0-G8 completion remains false.

NEXT:WP5_SAVED_O6_TYPED_EXPORT_AND_READER. Implement finite lossless source-bound export and reader under EXPORT_CONTRACT.json; capture, save/readback, checkpoint and cold exact reuse before scoped admission. Do not regenerate completed constructions.
''';(B/'REPORT.txt').write_text(report)
s=load(R/'CURRENT_STATUS.json');s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope='WP5_SAVED_O6_TYPED_EXPORT_AND_READER',next_scope_status='NOT_PREPARED',o6_readiness='o6_readiness0164/READINESS_VALIDATION.json',o6_export_contract='o6_readiness0164/EXPORT_CONTRACT.json',code_mirror=load(B/'CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
D=R/'handoff0164_o6_readiness';D.mkdir(exist_ok=True);shutil.copytree(B,D/B.name,dirs_exist_ok=True)
for folder in ['o5_export0162','o4_export0159','o3_export0156']:
 target=D/folder;target.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',target/'SCIENTIFIC_EXPORT.zip')
 if folder=='o5_export0162':
  shutil.copy2(R/folder/'EXPORT_BINDINGS.json',target/'EXPORT_BINDINGS.json');shutil.copytree(R/folder/'project',target/'project',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
target=D/'o5_integrate0163';target.mkdir(exist_ok=True);shutil.copy2(R/'o5_integrate0163/CATALOG_0146.json',target/'CATALOG_0146.json')
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0146','completed_action':'0164_O6_SOURCE_BOUND_READINESS','next_scope':s['next_scope'],'pending_bytes':0,'generation_calls':0,'new_admissions':0})
dump(D/'RECOVERY_DEPENDENCIES.json',{'master146':{'drive_file_id':'1nYMBOKyUapLp9aHAHMXMUXIQKTbOWgBw','sha256':'b402b8b2ed0f665cddd6996579ac560e86f5175d91b1b22620bbea62608b7a80'},'higher_O_sources':{'drive_file_id':'1Q1flImNUfUsqjeDAaguySy8FyrTL05-J','sha256':'90d2d33536a5263ad0dc9e832745c703089dde96f3561a585db27456ca40aeac'},'restore':'This compact bundle independently rechecks O6 readiness. Restore master146 and its pinned predecessor dependencies for native integration. Latest STATUS/cursor override ancestor prose.'})
(D/'READ_FIRST.txt').write_text(report+'\nVerify MANIFEST.json then run pinned Python o6_readiness0164/validate.py.\n');(D/'MANIFEST.json').unlink(missing_ok=True)
manifest={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',manifest)
p=R/'IG_O6_SOURCE_BOUND_READINESS_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*manifest,'MANIFEST.json']:z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in manifest.items():assert sha(z.read(n))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(manifest)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_O6_SOURCE_BOUND_READINESS_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0164_o6_readiness/READ_FIRST.txt\n');print(json.dumps(meta))
