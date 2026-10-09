"""Gate the three-chunk earned ledger on exact saved deliverables and proofs."""
from pathlib import Path
import json,hashlib,sys,os
B=Path(__file__).resolve().parent;W=B.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    r=json.load(sys.stdin);assert r['checkpoint']==252 and r['new_admissions']==0 and r['G2_promotion'] is False
    for v in r['artifacts'].values():
        p=Path(v['path']);assert v['drive_exact_readback_verified'] and v['local_metadata_applied'] and v['drive_file_id'] and v['library_file_id'] and sha(p)==v['sha256'] and p.stat().st_size==v['bytes'] and os.listxattr(p)
        assert sha(W/'drive_objects'/(v['sha256']+'.bin'))==v['sha256']
    ledger=json.loads((B/'LEDGER_CANDIDATE.json').read_text());prior=json.loads((W/'continuation0251/EARNED_LEDGER.json').read_text());assert ledger['completed_chunks'][:26]==prior['completed_chunks']
    for item in ledger['completed_chunks'][26:]:
        d=B/item['unit_directory'];n=json.loads((d/'NATIVE_RESULT.json').read_text());c=json.loads((d/'COLD_AUDIT.json').read_text());p=json.loads((d/'CHECKPOINT_PRESERVED.json').read_text())
        assert item['native_result_sha256']==sha(d/'NATIVE_RESULT.json') and item['cold_audit_sha256']==sha(d/'COLD_AUDIT.json') and item['specification_sha256']==sha(d/'SPEC.json')
        assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and c['cases_exact']==n['result']['cases_checked'] and c['complete_Q2_classes_exact']==n['result']['complete_Q2_classes_checked'] and c['public_projections_and_operational_witnesses_exact'] and not p['pending_objects'] and p['pending_bytes']==0
    (B/'SAVE_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');ledger['status']='EARNED_NATIVE_COLD_AND_EXACT_SAVED';ledger['save_receipt_sha256']=sha(B/'SAVE_RECEIPT.json')
    for item in ledger['completed_chunks'][26:]:item.update(save_receipt_path=str(B/'SAVE_RECEIPT.json'),save_receipt_sha256=ledger['save_receipt_sha256'])
    (B/'EARNED_LEDGER.json').write_text(json.dumps(ledger,indent=2)+'\n');status=json.loads((B/'STATUS_CANDIDATE.json').read_text());status.update(checkpoint0252_save_receipt=r,remaining_Q2_earned_ledger=str(B/'EARNED_LEDGER.json'),remaining_Q2_earned_ledger_sha256=sha(B/'EARNED_LEDGER.json'))
    for path in (W/'CURRENT_STATUS.json',Path('/workspace/scratch/75d85ae95659/CURRENT_STATUS.json')):
        if path.parent.exists():path.write_text(json.dumps(status,indent=2)+'\n')
    print('PASS three-chunk exact saves, earned ledger and checkpoint252 activation')
if __name__=='__main__':main()
