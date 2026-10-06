from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
R=Path.cwd();B=R/'o7_export0169';load=lambda n:json.loads((B/n).read_bytes());sha=lambda b:hashlib.sha256(b).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
n=load('NATIVE_RESULT.json');c=load('COLD_REUSE_RESULT.json');assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and n['result']['outcome']=='PASS' and c['status']=='PASS' and c['pending_bytes']==0 and load('CHECKPOINT_ACK.json')['outbox']['pending_bytes']==0
report='''INFINITY GRID REGISTERED SAVED O7 EXPORT/READER GATE0169 — 2026-10-06

PASS:COMPLETED/VERIFIED. Saved74 literal O7 states:72 nonseed carriers+2 external rank0 seeds,HET4 m7=0..5,HOM6 m7=0..1. All74 occurrence routes,322 nested O6 owner routes and82 nontrivial component routes checked.192 typed E7 edges. Eight selected O6 prototypes bind exact admitted O6 roots/components and ordered O5/O4/O3 resources. Two saved topology twins preserved and independently verified as external adversarial roots only. Full literal row hashes supplement lane/rank/binary E7 digests (IG-E7-STATE-v1| plus sorted14 unsigned32bit big-endian integers); no global canonicality claim.

E7 compatibility,capacity,remaining resource hashes and saved state/component accounting pass. Excluded unrecovered HOM6/TWIN4 routes,invalid owners,unavailable parent/action ancestry,root/nested/component copy isolation and closed-reader checks pass. Native capture and terminal checkpoint saved with exact byte readbacks. Cold restore/exact completion reuse PASS,pending0,generation0. Pure AST helper extraction and original producer/spec/selector/feature pool/source pins preserved. No original generator main or canonicalization search ran.

Master0147/147 scientific slices unchanged; no O7 admission yet. Full final campaign survivor/profile coverage,closeout D1-D6,complete derivation ancestry and76 microscopic O2 panels remain unresolved. No O7 graduation or automatic O8 authorization. Full L0-G8 completion remains false.
NEXT:Saved finite O7 scoped admission and registered catalog/unified reader integration,preserving all147 predecessor slices/routes. Do not regenerate completed constructions.
''';(B/'REPORT.txt').write_text(report)
s=json.loads((R/'CURRENT_STATUS.json').read_bytes());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope='SAVED_O7_TYPED_INCIDENCE_SCOPED_ADMISSION_AND_CATALOG_INTEGRATION',next_scope_status='REGISTERED_EXPORT_READER_VERIFIED_PRESERVED_COLD_REUSED',o7_export_result='o7_export0169/NATIVE_RESULT.json',code_mirror=load('CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
D=R/'handoff0169_o7_export';D.mkdir(exist_ok=True);dest=D/B.name;dest.mkdir(exist_ok=True)
for p in B.iterdir():
 if p.is_file() and p.suffix in ['.json','.py','.txt'] and not p.name.startswith('TO_SAVE') and p.name not in ['SAVE_RECEIPT.json','DELIVERY_BINDINGS.json','FRESH_UNPACK_VERIFICATION.json']:shutil.copy2(p,dest/p.name)
shutil.copytree(B/'project',dest/'project',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
for name in ['SCIENTIFIC_EXPORT.zip','FINAL_CHECKPOINT_SLIM.zip']:shutil.copy2(B/name,dest/name)
for folder in ['o6_export0165','o5_export0162','o4_export0159','o3_export0156']:
 t=D/folder;t.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',t/'SCIENTIFIC_EXPORT.zip')
for name,hashes in [('READBACKS.json',{r['sha256'] for r in load('PENDING.json')['pending_objects']}),('CHECKPOINT_READBACKS.json',{r['sha256'] for r in load('EXPORT.json')['dependencies']}|{load('EXPORT.json')['sha256']})]:
 m=load(name);dump(dest/name,{h:m[h] for h in sorted(hashes)})
dump(D/'RECOVERY_DEPENDENCIES.json',{'master147':{'drive_file_id':'1EcDvoG2BozFXbLN08T1fSmZoMF8rzs34','sha256':'9f3d5bb4b413ebb6dfecacd3f5adc53535caf5869af417a836665d9b6e338b34'},'readiness':{'drive_file_id':'1_StVwZ3B_zeRUrpBnktwA6ZQB4ikHUzn','sha256':'49ac3c837b8a31c8714e2f9c50c9c5cbc51ec1b28f0f4e82cba414c39ef070ae'},'native_restore':'EXPORT.json and CHECKPOINT_READBACKS.json pin exact saved dependency IDs/hashes. Rebind local paths only;restore pinned checkpoint then reuse completion. Do not regenerate.'})
(D/'READ_FIRST.txt').write_text(report);dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0147','scientific_slices':147,'completed_gate':'0169_SAVED_O7_EXPORT_READER','next_scope':s['next_scope'],'scientific_admission_issued':False,'capture_id':load('POINTER.json')['capture_id'],'completion_sha256':n['completion_sha256'],'result_sha256':n['result_sha256'],'pending_bytes':0,'generation_calls':0})
(D/'MANIFEST.json').unlink(missing_ok=True);m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',m);p=R/'IG_O7_GATE_0169_VERIFIED_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for name in [*m,'MANIFEST.json']:z.write(D/name,name)
with zipfile.ZipFile(p) as z:
 for name,x in m.items():assert sha(z.read(name))==x['sha256']
(R/'CURRENT_START.txt').write_text('handoff0169_o7_export/READ_FIRST.txt\n');(R/'IG_O7_GATE_0169_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+sha(p.read_bytes())+'\n');meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta);print(json.dumps(meta))
