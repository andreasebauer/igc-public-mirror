"""Unchanged official observer specs; native dry gate and earlier unit proofs required."""
from pathlib import Path
import hashlib,json,sys,shutil
B=Path(__file__).resolve().parent;idx=int(B.name);W=B.parent.parent;Q=W/'continuation0285';E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid import submission as sub,portable_registry
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
gate=json.loads((W/'continuation0286/GATE_EARNED.json').read_text())
assert gate['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED_DRY_GATE'
assert sha(W/'continuation0286/NATIVE_RESULT.json')==gate['native_result_sha256'] and sha(W/'continuation0286/COLD_AUDIT.json')==gate['cold_audit_sha256']
assert json.loads((W/'continuation0286/DELIVERY_RECEIPT_SAVED.json').read_text())['drive_exact_readback_verified']
reg=json.loads((Q/'PREREGISTRATION.json').read_text());assert sha(Q/'PREREGISTRATION.json')==gate['registration_sha256']
for name,h in reg['source_sha256'].items():assert sha(Q/name)==h
for name,h in reg['required_engine_files'].items():assert sha(E/'infinity_grid'/name.split('/',1)[1])==h
assert sha(B/'SPEC.json')==reg['specifications'][idx]['spec_sha256']
assert shutil.disk_usage(B).free>=reg['minimum_free_disk_bytes']
if idx>3:
    prior=B.parent/f'{idx-1:03d}';receipt=json.loads((prior/'UNIT_SAVE_RECEIPT.json').read_text())
    assert receipt['status']=='PASS_NATIVE_COLD_AND_EXACT_UNIT_SAVED'
    for x in receipt['artifacts']:assert x['drive_exact_readback_verified'] and sha(Path(x['path']))==x['sha256']
prior_ledger=json.loads((W/'continuation0287/EARNED_LEDGER.json').read_text());assert prior_ledger['status']=='EARNED_NATIVE_COLD_AND_EXACT_SAVED' and prior_ledger['next_chunk_index']==3
for x in prior_ledger['save_receipts']:assert sha(Path(x['path']))==x['sha256']
store=Path(f'/tmp/ig_native0288_{idx:03d}/store');assert not (store/'coordination/PROJECT.json').exists()
portable_registry.initialize(store,f'Official historical observer chunk{idx:03d}',E)
r=sub.capture(store,B/'SPEC.json');(B/'POINTER.json').write_text(json.dumps(r,indent=2)+'\n')
s=sub.save_status(r['workspace']);(B/'CAPTURE_PENDING.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({'capture_id':r['capture_id'],'pending_objects':len(s['pending_objects'])}))
