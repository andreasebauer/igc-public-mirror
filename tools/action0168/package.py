from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
R=Path.cwd();B=R/'o7_readiness0168';D=R/'handoff0168_o7_readiness';D.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda p:json.loads(p.read_bytes())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
v=load(B/'READINESS_VALIDATION.json');assert v['status']=='SAVED_O7_RECOVERED_SCOPE_SOURCE_BOUND_READY'
report='''INFINITY GRID O7 SAVED SCOPE SOURCE-BOUND READINESS — ACTION0168 — 2026-10-06

Master0147 retained,147 scientific slices. All74 recovered O7 states pass exact O6 owner bindings, typed E7 coordinates, compatibility, endpoint capacity, ordered resource materialization and saved accounting.72 nonseed states+2 external rank0 seeds,HET4 m7=0..5,HOM6 m7=0..1.192 typed E7 edges,322 O6 owner occurrences,82 nontrivial components. All8 selected O6 prototypes resolve to admitted O6 roots and exact components; source owner labels are explicitly normalized to component labels for accounting. Two saved same-R6/different-E6 topology twins validate as external adversarial O6 roots,not new admissions.

This is a finite saved scope. Later campaign COMPLETE snapshots lack full final survivor/profile coverage. Historical closeout D1-D6, complete parent/action ancestry and76 missing microscopic O2 panels remain unresolved. No construction generation, no new scientific admission, no O7 graduation or automatic O8 authorization. Full L0-G8 completion remains false.

NEXT:WP5_SAVED_O7_SCOPED_TYPED_EXPORT_AND_READER. Preserve74 literal rows,original producer/spec/selector/feature pool,8 exact selected O6 bindings,nested O5/O4/O3 resources,reservations and component accounting. Implement lossless reader,then native capture/save/readback/checkpoint/cold exact reuse before scoped admission. No generation or canonicalization search is needed.
'''
(B/'REPORT.txt').write_text(report)
contract={'schema':'IG_SAVED_O7_SCOPED_EXPORT_CONTRACT_V1','counts':v['counts'],'identity':'lane/rank/binary E7 edge digest plus full literal payload hash; no global canonical isomorphism claim','digest_encoding':'SHA256(IG-E7-STATE-v1| plus sorted typed edges,each14 unsigned32bit big-endian integers)','required':['all74 literal saved rows and frozen lane owner order','192 typed E7 edges with reservations and multiplicity','eight exact O6 component/root bindings and nested resources','original producer with pure helper AST provenance,spec,selector,feature pool and source pins','82 source-owner component incidence/accounting records','explicit absent complete parent/action occurrence ancestry','external twin witnesses distinguished from admitted O6 roots'],'generation_calls':0,'new_admissions':0,'master_catalog_sha256':v['catalog_sha256'],'scope_excludes':'unrecovered final HOM6 m7=2..6 and TWIN4 campaign states/profiles; no historical closeout certification'};dump(B/'EXPORT_CONTRACT.json',contract)
s=load(R/'CURRENT_STATUS.json');s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope=v['next_scope'],next_scope_status='NOT_PREPARED',o7_readiness='o7_readiness0168/READINESS_VALIDATION.json',o7_export_contract='o7_readiness0168/EXPORT_CONTRACT.json',code_mirror=load(B/'CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
shutil.copytree(B,D/B.name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('SAVE_RECEIPT.json','DELIVERY_BINDINGS.json','FRESH_UNPACK_VERIFICATION.json'))
shutil.copytree(R/'o7_readiness0167',D/'o7_readiness0167',dirs_exist_ok=True,ignore=shutil.ignore_patterns('SAVE_RECEIPT.json','DELIVERY_BINDINGS.json','FRESH_UNPACK_VERIFICATION.json'))
for folder in ['o6_export0165','o5_export0162','o4_export0159','o3_export0156']:
 t=D/folder;t.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',t/'SCIENTIFIC_EXPORT.zip')
 if folder=='o6_export0165':
  shutil.copy2(R/folder/'EXPORT_BINDINGS.json',t/'EXPORT_BINDINGS.json');shutil.copytree(R/folder/'project',t/'project',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
t=D/'o6_integrate0166';t.mkdir(exist_ok=True);shutil.copy2(R/'o6_integrate0166/CATALOG_0147.json',t/'CATALOG_0147.json')
for p in (R/'o_readiness0158/sources').iterdir():
 if 'O7_' in p.name:shutil.copytree(p,D/'o_readiness0158/sources'/p.name,dirs_exist_ok=True)
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0147','completed_action':'0168_O7_RECOVERED_SCOPE_SOURCE_BOUND_READINESS','next_scope':v['next_scope'],'generation_calls':0,'new_admissions':0})
dump(D/'RECOVERY_DEPENDENCIES.json',{'master147':{'drive_file_id':'1EcDvoG2BozFXbLN08T1fSmZoMF8rzs34','sha256':'9f3d5bb4b413ebb6dfecacd3f5adc53535caf5869af417a836665d9b6e338b34'},'higher_O_sources':{'drive_file_id':'1Q1flImNUfUsqjeDAaguySy8FyrTL05-J','sha256':'90d2d33536a5263ad0dc9e832745c703089dde96f3561a585db27456ca40aeac'},'restore':'This compact bundle independently validates O7 readiness. Restore master147 and dependencies before native integration. Latest STATUS/cursor override ancestor prose.'})
(D/'READ_FIRST.txt').write_text(report+'\nVerify MANIFEST.json then run pinned Python o7_readiness0168/validate.py from unpack root.\n')
(D/'MANIFEST.json').unlink(missing_ok=True);m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',m)
p=R/'IG_O7_SOURCE_BOUND_READINESS_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for n in [*m,'MANIFEST.json']:z.write(D/n,n)
with zipfile.ZipFile(p) as z:
 for n,x in m.items():assert sha(z.read(n))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_O7_SOURCE_BOUND_READINESS_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0168_o7_readiness/READ_FIRST.txt\n');print(json.dumps(meta))
