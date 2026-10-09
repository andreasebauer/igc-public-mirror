"""Package scoped seal and admission preparation; master remains unchanged."""
from pathlib import Path
import hashlib,json,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a=json.loads((B/'AUDIT.json').read_text());assert a['status']=='PASS_INDEPENDENT_SEALED_EXPORT_AUDIT'
    seal=json.loads((B/'SEAL_RECEIPT.json').read_text());proposal=json.loads((B/'SCOPED_ADMISSION_PROPOSAL.json').read_text())
    assert proposal['status']=='PREPARED_NOT_ADMITTED' and proposal['base_master_slices']==151
    report=W/'IG_MASTER151_G1_EXACT_PARENT_SEAL0232_2026-10-09.txt'
    report.write_text('Checkpoint 0232: qualified G1 exact parent export sealed; scoped admission prepared.\n\nHistorical V1 replay depth: 100. Master: 151. New admissions: 0. G2 promotion: false.\n\nThe preserved terminal cohort contains 193 carriers, 16,528 exact DAG nodes, 192 public interface classes and 1,351 one-endpoint reservation rows. The seal contains exact saved input transport bytes, historical V1 observations, separately retained V2 observations, the frozen O7 seed, decoder source, native completion evidence, independent cold qualification evidence and recovery dependencies.\n\nAn independent export audit verified every archived file hash and size, ZIP integrity, exact decompressed input identity, canonical DAG scientific identity, reachability, acyclicity and all 193 carrier joins to both populations. Previously earned full 193 DAG roundtrips remain bound to checkpoint 0231; no new DAG decoding or candidate generation was performed in 0232.\n\nThe scoped admission proposal links the newly available exact parent DAG to the existing historical public projection slice. Earlier slices retain their recorded authority. The proposal is prepared, not admitted. No Q2 payload, reservation witness certification, fresh pair realization or G2 graduation is asserted.\n\nSeal scientific root SHA256: '+seal['scientific_root_sha256']+'\nSeal archive SHA256: '+seal['sha256']+'\nBase master151 catalog SHA256: '+proposal['base_catalog_sha256']+'\n\nRecovery: verify the handoff MANIFEST.json and nested seal SEAL_MANIFEST.json. Use the preserved checkpoint READBACKS.json and CHECKPOINT_EXPORT.json for full frozen engine/runtime recovery; exact decoder and seed bytes are included. Use audit_sealed.py under the pinned runtime after restoring the frozen engine at /tmp/ig_engine0204.\n\nNext: review and register the scoped exact-parent admission against the unchanged master151 catalog, then review downstream G2 source bindings.\n')
    s=json.loads((W/'CURRENT_STATUS.json').read_text());s.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=232,current_phase='G1_EXACT_PARENT_SEALED_ADMISSION_PREPARED',master_slices=151,new_admissions=0,latest_audit=a,checkpoint0232=a,seal_scientific_root_sha256=seal['scientific_root_sha256'],scoped_admission_status='PREPARED_NOT_ADMITTED',next_scope='G1_EXACT_PARENT_SCOPED_ADMISSION_REVIEW_AND_REGISTRATION',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()))
    (B/'STATUS_CANDIDATE.json').write_text(json.dumps(s,indent=2)+'\n')
    files=[p for p in sorted(B.iterdir()) if p.is_file() and p.suffix in ('.json','.py','.zip') and p.name not in ('DELIVERABLES.json','SAVE_RECEIPT.json')]+[report]
    manifest={str(p.relative_to(W)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in files}
    out=W/'IG_MASTER151_G1_EXACT_PARENT_SEAL0232_HANDOFF_2026-10-09.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(W)))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for n,v in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    r=dict(handoff=dict(path=str(out),sha256=digest(out),bytes=out.stat().st_size),report=dict(path=str(report),sha256=digest(report),bytes=report.stat().st_size),seal=seal,checkpoint=232,status=a['status'],master_slices=151,new_admissions=0)
    (B/'DELIVERABLES.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
