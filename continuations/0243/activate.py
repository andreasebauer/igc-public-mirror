"""Activate the earned chunk ledger only after exact saved-artifact evidence."""
from pathlib import Path
import json,sys,hashlib,os
B=Path(__file__).resolve().parent;W=B.parent;Q=W/'continuation0240'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    receipt=json.load(sys.stdin);assert receipt['checkpoint']==243 and receipt['new_admissions']==0 and receipt['G2_promotion'] is False
    assert set(receipt['artifacts'])=={'report','handoff'}
    for row in receipt['artifacts'].values():
        path=Path(row['path']);assert row['drive_exact_readback_verified'] and row['local_metadata_applied'] and row['drive_file_id'] and row['library_file_id']
        assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes'] and os.listxattr(path)
        assert sha(W/'drive_objects'/(row['sha256']+'.bin'))==row['sha256']
    ledger=json.loads((B/'LEDGER_CANDIDATE.json').read_text());item=ledger['completed_chunks'][-1];cold=json.loads((B/'COLD_AUDIT.json').read_text());native=json.loads((B/'NATIVE_RESULT.json').read_text());preserved=json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())
    assert ledger['plan_sha256']==sha(Q/'PLAN.json') and ledger['preregistration_sha256']==sha(Q/'PREREGISTRATION.json')
    assert item['native_result_sha256']==sha(B/'NATIVE_RESULT.json') and item['cold_audit_sha256']==sha(B/'COLD_AUDIT.json') and item['specification_sha256']==sha(B/'SPEC.json')
    assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and cold['cases_exact']==62 and cold['complete_Q2_classes_exact']==31 and cold['public_projections_and_operational_witnesses_exact']
    assert not preserved['pending_objects'] and preserved['pending_bytes']==0
    (B/'SAVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    ledger['status']='EARNED_NATIVE_COLD_AND_EXACT_SAVED';ledger['save_receipt_sha256']=sha(B/'SAVE_RECEIPT.json');item['save_receipt_path']=str(B/'SAVE_RECEIPT.json');item['save_receipt_sha256']=ledger['save_receipt_sha256']
    (B/'EARNED_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n')
    status=json.loads((B/'STATUS_CANDIDATE.json').read_text());status['checkpoint0243_save_receipt']=receipt;status['remaining_Q2_earned_ledger']=str(B/'EARNED_LEDGER.json');status['remaining_Q2_earned_ledger_sha256']=sha(B/'EARNED_LEDGER.json')
    catalog=W/'continuation0233/CATALOG_0152.json';data=json.loads(catalog.read_text());assert data['release_id']=='MASTER_DATA_V1_0152'
    status.update(latest_committed_release=data['release_id'],scientific_slices=152,catalog_sha256=sha(catalog),catalog_path=str(catalog))
    for target in (W/'CURRENT_STATUS.json',Path('/workspace/scratch/75d85ae95659/CURRENT_STATUS.json')):
        if target.parent.exists():target.write_text(json.dumps(status,indent=2)+'\n')
    print('PASS exact artifact saves, earned ledger and checkpoint243 activation')
if __name__=='__main__':main()
