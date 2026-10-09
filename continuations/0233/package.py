"""Package verified scoped admission; activate only after deliverables are saved."""
from pathlib import Path
import json,hashlib,zipfile,datetime
B=Path(__file__).resolve().parent;W=B.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    a=json.loads((B/'AUDIT.json').read_text());n=json.loads((B/'NATIVE_RESULT.json').read_text());p=json.loads((B/'CHECKPOINT_PRESERVED.json').read_text())
    assert a['status']=='PASS_COLD_SCOPED_EXACT_PARENT_ADMISSION' and n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED' and p['pending_bytes']==0
    assert json.loads((B/'NATIVE_EXPORT_SAVE.json').read_text())['raw_readback_verified']
    old=json.loads((B/'CATALOG_0151.json').read_text());cat=json.loads((B/'CATALOG_0152.json').read_text());assert cat['slices'][:-1]==old['slices'] and len(cat['slices'])==152
    ad=json.loads((B/'SCOPED_ADMISSION.json').read_text());assert cat['slices'][-1]['scoped_admission_sha256']==sha(B/'SCOPED_ADMISSION.json')
    report=W/'IG_MASTER152_G1_EXACT_PARENT_ADMITTED0233_2026-10-09.txt'
    report.write_text('Checkpoint 0233: scoped historical V1 exact-parent source admitted as master152.\n\nMaster: 152 scientific slices. One new admission. Historical V1 replay depth: 100. G2 promotion: false.\n\nThe admitted source contains 193 terminal G1 carriers and 16,528 exact DAG nodes. It binds 192 historical public interface classes and 1,351 one-endpoint reservation rows to the preserved exact parent DAG. All 151 preceding slices remain exactly unchanged. Their original limited authorities are retained. Q2 payloads, reservation witness certification, fresh pair realization and G2 graduation remain outside this admission.\n\nThe immutable release candidate and scope were preregistered before native execution. The registered native handler qualified every carrier route, all DAG nodes, archive/file identities, canonical DAG identity, reachability, acyclicity, interface joins, reader missing-key and closed-reader behavior, and the append-only catalog relation. Native completion: COMPLETED / VERIFIED. Its checkpoint was fully preserved with pending bytes zero. Independent cold restoration repeated the source-reader qualification using only restored source and saved dependencies and reproduced the native result exactly.\n\nNo candidate generation and no new scientific DAG decoding were performed. Full 193-root reconstruction roundtrips remain earned and provenance-bound at checkpoint0231. This step qualifies saved-data access and scoped admission.\n\nCatalog152 SHA256: '+sha(B/'CATALOG_0152.json')+'\nScoped admission SHA256: '+sha(B/'SCOPED_ADMISSION.json')+'\nSeal scientific root SHA256: '+ad['scientific_root_sha256']+'\nSeal archive SHA256: '+ad['archive']['sha256']+'\n\nRecovery: verify handoff MANIFEST.json. Restore NATIVE_CHECKPOINT_SLIM.zip from CHECKPOINT_EXPORT.json dependencies and exact Drive readbacks in READBACKS.json. Recover the sealed source from its recorded Drive file ID. The project ExactParentReader gives source DAG node and historical interface routes; earlier master readers remain unchanged. Use pinned CPython3.13.5 / SQLite3.51.3 with the frozen engine.\n\nNext: review downstream saved G2 source bindings against this exact G1 cohort. Do not reinterpret stored G2 audits as fresh realization or graduation.\n')
    status=json.loads((W/'CURRENT_STATUS.json').read_text());status.update(status_as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_checkpoint=233,latest_accepted_checkpoint=233,current_phase='G1_EXACT_PARENT_SCOPED_ADMITTED',master_slices=152,new_admissions=1,latest_audit=a,checkpoint0233=a,latest_capture_id=json.loads((B/'POINTER.json').read_text())['capture_id'],catalog_path=str(B/'CATALOG_0152.json'),scoped_admission_status='ADMITTED_WITHIN_HISTORICAL_V1_EXACT_PARENT_SOURCE_SCOPE',pending_bytes=0,latest_native_result=n,next_scope='G2_SAVED_SOURCE_BINDING_REVIEW_AGAINST_ADMITTED_EXACT_G1_COHORT',code_mirror=json.loads((B/'CODE_MIRROR.json').read_text()))
    (B/'STATUS_CANDIDATE.json').write_text(json.dumps(status,indent=2)+'\n')
    files=[p for p in sorted(B.rglob('*')) if p.is_file() and p.suffix in ('.json','.py','.zip','.txt','.log') and '__pycache__' not in p.parts and not p.name.startswith('private_') and p.name not in ('DELIVERABLES.json','SAVE_RECEIPT.json')]+[report]
    manifest={str(p.relative_to(W)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    out=W/'IG_MASTER152_G1_EXACT_PARENT_ADMITTED0233_HANDOFF_2026-10-09.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,str(p.relative_to(W)))
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for k,v in manifest.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256']
    result=dict(checkpoint=233,master_slices=152,new_admissions=1,report=dict(path=str(report),sha256=sha(report),bytes=report.stat().st_size),handoff=dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size))
    (B/'DELIVERABLES.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
