import pathlib,json,hashlib,zipfile
W=pathlib.Path('/workspace/scratch/2a87972b51af');B=W/'continuation0288';Q=W/'continuation0285';O=pathlib.Path(__file__).parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((B/'BATCH_AUDIT.json').read_text());assert audit['official_members_compared']==372 and audit['official_complete_classes_compared']==186
report=O/'IG_MASTER152_G2_OBSERVER_CONTINUATION0288_2026-10-11.txt'
report.write_text('Checkpoint 0288 — official historical public observer\n\nObserver chunks 003–005 passed native execution and independent cold verification.\nAdded: 186 records / 93 complete classes.\nCumulative official observer coverage: 372 records / 186 complete classes (chunks 000–005).\nRemaining: 108 chunks / 6,686 records / 3,337 classes.\nNext: official observer chunks 006–008.\nDry-run coverage credit: zero.\n\nMaster 152; no admissions or G2 promotion.\nFull 580,351-member historical public-observer comparison, canonicalization impact, full L0–G8 and RC remain open. Original v0292 controller identity is not claimed. Original Q2 campaign coverage remains unchanged.\n\nEach new unit has exact external native-result, cold-audit and checkpoint readbacks. Earned ledger activates only after exact final report, ZIP and receipt saves. Frozen registration, scientific source and specifications remain unchanged; one scientific worker, 4 GiB memory bound, no automatic overall runtime deadline.\n\nOperator code mirror commit: '+json.loads((B/'CODE_MIRROR.json').read_text())['commit']+'.\n')
files={}
for p in B.rglob('*'):
 if p.is_file() and p.suffix!='.txt' and p.name!='ACK_BATCH.json' and '__pycache__' not in p.parts:files['continuation0288/'+str(p.relative_to(B))]=p
for n in ['EARNED_LEDGER.json','DRIVE_READBACKS.json','LIBRARY_SAVE_RECEIPT.json','DELIVERY_RECEIPT_SAVED.json','ACTIVATION_SAVED.json']:
 p=W/'continuation0287'/n;files[str(p.relative_to(W))]=p
for i in range(3):
 for n in ['SPEC.json','NATIVE_RESULT.json','COLD_AUDIT.json','CHECKPOINT_PRESERVED.json','UNIT_SAVE_RECEIPT.json']:
  p=W/'continuation0287'/f'{i:03d}'/n;files[str(p.relative_to(W))]=p
for p in [Q/'PREREGISTRATION.json',W/'continuation0286/GATE_EARNED.json']:
 files[str(p.relative_to(W))]=p
for p in (Q/'project').rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(W))]=p
for idx in range(6,9):
 p=Q/'specs'/f'{idx:03d}.json';files[str(p.relative_to(W))]=p
 s=json.loads(p.read_text())
 for inp in s['inputs']:
  if inp['logical_name'] in ['cases','observer_expected']:
   a=pathlib.Path(inp['path']);files[str(a.relative_to(W))]=a
files[report.name]=report
manifest={name:{'sha256':sha(p),'bytes':p.stat().st_size} for name,p in files.items()}
out=O/'IG_MASTER152_G2_OBSERVER_CONTINUATION0288_HANDOFF_2026-10-11.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for name,p in files.items():z.write(p,name)
 z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for name,row in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==row['sha256']
print(json.dumps({'report':str(report),'zip':str(out),'zip_bytes':out.stat().st_size,'files':len(files)}))
