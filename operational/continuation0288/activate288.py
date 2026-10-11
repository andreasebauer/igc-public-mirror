import pathlib,json,hashlib,datetime
W=pathlib.Path('/workspace/scratch/2a87972b51af');B=W/'continuation0288'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
dr=json.loads((B/'DRIVE_READBACKS.json').read_text());rr=json.loads((B/'DELIVERY_RECEIPT_SAVED.json').read_text());lr=json.loads((B/'LIBRARY_SAVE_RECEIPT.json').read_text())
assert len(dr)==2 and len(rr)==1
for r in dr+rr:assert r['drive_exact_readback_verified'] and sha(pathlib.Path(r['path']))==r['sha256']
assert len(lr['results'])==3
for r in lr['results']:assert r['status']=='succeeded' and r['local_metadata_applied']
out=json.loads((B/'LEDGER_CANDIDATE.json').read_text());assert out['official_members_compared']==372 and out['official_complete_classes_compared']==186
out.update(status='EARNED_NATIVE_COLD_AND_EXACT_SAVED',save_receipts=[{'path':str(B/n),'sha256':sha(B/n)} for n in ['DRIVE_READBACKS.json','LIBRARY_SAVE_RECEIPT.json','DELIVERY_RECEIPT_SAVED.json']])
(B/'EARNED_LEDGER.json').write_text(json.dumps(out,indent=2)+'\n')
p=W/'CURRENT_STATUS.json';s=json.loads(p.read_text());assert s['latest_checkpoint']==287
s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=288,current_phase='G2_OFFICIAL_OBSERVER_NATIVE_COLD_IN_PROGRESS',checkpoint0288_status=out['status'],checkpoint0288_earned_ledger=str(B/'EARNED_LEDGER.json'),official_observer_completed_chunks=list(range(6)),official_observer_next_chunk=6,official_observer_pending_chunks=108,official_observer_members_compared=372,official_observer_complete_classes_compared=186,next_scope=out['next_scope'],observer_batch_in_flight=None,pending_bytes=0)
p.write_text(json.dumps(s,indent=2)+'\n');print('Checkpoint 0288 earned ledger activated; next 006–008; 108 chunks remaining')
