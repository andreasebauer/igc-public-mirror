from pathlib import Path
import json,hashlib,zipfile,shutil,datetime
R=Path.cwd();B=R/'o7_additional_export0173';sha=lambda b:hashlib.sha256(b).hexdigest();load=lambda n:json.loads((B/n).read_bytes())
def dump(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
n=load('NATIVE_RESULT.json');c=load('COLD_REUSE_RESULT.json')
assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and n['result']['outcome']=='PASS' and c['status']=='PASS' and c['pending_bytes']==0 and load('CHECKPOINT_ACK.json')['outbox']['pending_bytes']==0
hashes={str(p.relative_to(B/'project/previous_o7')):sha(p.read_bytes()) for p in (B/'project/previous_o7').rglob('*.py')}
for name,h in hashes.items():assert sha((R/'o7_export0169/project'/name).read_bytes())==h
dump(B/'PREDECESSOR_SOURCE_HASHES.json',hashes)
report='INFINITY GRID ACTION0173 ADDITIONAL SAVED O7 EXPORT/READER GATE — 2026-10-06\n\nCOMPLETED/VERIFIED/PASS.24 additional whole HOM6 roots (12 at rank2 and12 at rank3),60 E7 edges,144 O6 owner occurrences.205 source component occurrences retained independently of164 exact contextual objects,512 E7 edge occurrences,533 O6 owner occurrences. All root,component,object,nested owner and remaining-resource routes checked. Four admitted O6 component prototypes and two independently verified external O6 topology twins with admitted O5 bindings.\n\nSource identities preserve ordered exact parent identities/colors plus E7 binary digest and full literal payload hashes. Original checkpoint bytes and science pins,205 profile/survivor literal equality,typed incidence/capacity/accounting,component connectedness,exact R7 resource hashes and readiness pins verified.96 available and68 unavailable source whole-root references remain explicit; source-owner mappings and complete action ancestry unavailable. Saved profile summaries remain source evidence only; full Counter and fresh profile certification unavailable.\n\nCaptured source/input saves and terminal checkpoint byte readbacks verified. Cold restore/exact completion reuse PASS; pending0,generation0. Predecessor reader modules unchanged. MASTER_DATA_V1_0148/148 scientific slices unchanged; no new admission,no graduation or automatic O8,full L0-G8 completion false.\n\nNEXT: additional saved O7 root/component scoped admission and catalog/unified reader integration, preserving all148 predecessor slices/routes. Do not regenerate completed constructions or invoke original producer mains.\n'
(B/'REPORT.txt').write_text(report)
s=json.loads((R/'CURRENT_STATUS.json').read_bytes());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),next_scope='WP5_ADDITIONAL_SAVED_O7_ROOT_AND_COMPONENT_SCOPED_ADMISSION_AND_INTEGRATION',next_scope_status='NATIVE_EXPORT_READER_VERIFIED_PRESERVED_COLD_REUSED',additional_O7_export_result='o7_additional_export0173/NATIVE_RESULT.json',code_mirror=load('CODE_MIRROR.json'));dump(R/'CURRENT_STATUS.json',s)
D=R/'handoff0173_o7_additional_export';D.mkdir(exist_ok=True);dest=D/B.name;dest.mkdir(exist_ok=True)
for p in B.iterdir():
 if p.is_file() and p.suffix in ['.json','.py','.txt'] and not p.name.startswith('TO_SAVE') and p.name not in ['SAVE_RECEIPT.json','DELIVERY_BINDINGS.json','FRESH_UNPACK_VERIFICATION.json']:shutil.copy2(p,dest/p.name)
shutil.copytree(B/'project',dest/'project',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
for name in ['SCIENTIFIC_EXPORT.zip','FINAL_CHECKPOINT_SLIM.zip']:shutil.copy2(B/name,dest/name)
for folder in ['o7_export0169','o6_export0165','o5_export0162','o4_export0159','o3_export0156']:
 t=D/folder;t.mkdir(exist_ok=True);shutil.copy2(R/folder/'SCIENTIFIC_EXPORT.zip',t/'SCIENTIFIC_EXPORT.zip')
t=D/'o7_integrate0170';t.mkdir(exist_ok=True);shutil.copy2(R/'o7_integrate0170/CATALOG_0148.json',t/'CATALOG_0148.json')
for name,hs in [('READBACKS.json',{r['sha256'] for r in load('PENDING.json')['pending_objects']}),('CHECKPOINT_READBACKS.json',{r['sha256'] for r in load('EXPORT.json')['dependencies']})]:
 mapping=load(name);dump(dest/name,{h:mapping[h] for h in sorted(hs)})
dump(D/'RECOVERY_DEPENDENCIES.json',{'master148':{'drive_file_id':'1pRZIf4KuJzajRIaruSGCUSxHXdmikNQA','sha256':'9269ae6ed7563a83e9beb59e918bd65c7e8a062d2100a5c73202e10d8e55b9f3'},'readiness172':{'drive_file_id':'1te-f2rEDcgTtCKn3TwPiFCxoUG83Ct_a','sha256':'5ebe540440ae76a6b62650e6aa37956b832c6d97611d3b1c8eafa4710945ad88'},'native_restore':'EXPORT.json and CHECKPOINT_READBACKS.json pin exact saved dependency IDs/hashes. Rebind paths only,restore pinned checkpoint,reuse exact completion. Do not regenerate.'})
(D/'READ_FIRST.txt').write_text(report+'\nVerify MANIFEST.json and run pinned Python o7_additional_export0173/verify_saved.py BUNDLE.zip.\n')
dump(D/'STATUS.json',s);dump(D/'CONTINUATION_CURSOR.json',{'master_release':'MASTER_DATA_V1_0148','scientific_slices':148,'completed_gate':'0173_ADDITIONAL_SAVED_O7_EXPORT_READER','next_scope':s['next_scope'],'scientific_admission_issued':False,'capture_id':load('POINTER.json')['capture_id'],'completion_sha256':n['completion_sha256'],'result_sha256':n['result_sha256'],'pending_bytes':0,'generation_calls':0})
(D/'MANIFEST.json').unlink(missing_ok=True);m={str(p.relative_to(D)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in D.rglob('*') if p.is_file()};dump(D/'MANIFEST.json',m)
p=R/'IG_O7_ADDITIONAL_GATE_0173_VERIFIED_HANDOFF_2026-10-06.zip'
with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:
 for name in sorted([*m,'MANIFEST.json']):z.write(D/name,name)
with zipfile.ZipFile(p) as z:
 for name,x in m.items():assert sha(z.read(name))==x['sha256']
meta={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'manifest_files':len(m)};dump(B/'DELIVERY_BINDINGS.json',meta)
(R/'IG_O7_ADDITIONAL_GATE_0173_START_2026-10-06.txt').write_text(report+'\nBundle:'+p.name+'\nBundle SHA256:'+meta['sha256']+'\n');(R/'CURRENT_START.txt').write_text('handoff0173_o7_additional_export/READ_FIRST.txt\n');print(json.dumps(meta))
