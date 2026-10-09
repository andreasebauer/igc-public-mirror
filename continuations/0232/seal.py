"""Seal previously qualified bytes; no candidate construction or admission."""
from pathlib import Path
import hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parent; P=B.parent/'continuation0231'; S=B/'sealed'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def main():
    A=json.loads((P/'AUDIT.json').read_text()); assert A['all193_DAG_roundtrip'] and A['native_exact_state_identity']
    assert A['status']=='PASS_COLD_HISTORICAL_V1_TERMINAL100_REQUALIFICATION'
    d=json.loads((P/'DELIVERABLES.json').read_text()); assert sha(Path(d['path']))==d['sha256']
    with zipfile.ZipFile(d['path']) as z:
        m=json.loads(z.read('MANIFEST.json'))
        for n,v in m.items():assert hashlib.sha256(z.read(n)).hexdigest()==v['sha256']
    S.mkdir(parents=True,exist_ok=True)
    names=['BOOTSTRAP100_HISTORICAL.json.gz','HISTORICAL_V1_INTERFACE_POPULATION.json','DIAGNOSTIC_V2_POPULATION.json','HISTORICAL_IMPLEMENTATION_SPEC_V1.json','HISTORICAL_DEPTH100_SOURCE_INPUT.json','SOURCE_INPUT_RECOVERY.json','AUDIT.json','SPEC_BINDING_DIAGNOSIS.json','NATIVE_RESULT.json','CAPTURE_PRESERVED.json','CHECKPOINT_PRESERVED.json','NATIVE_EXPORT_SAVE.json','CHECKPOINT_EXPORT.json','READBACKS.json','RUNTIME_MANIFEST.json','SAVE_RECEIPT.json','CODE_MIRROR.json']
    for n in names:shutil.copyfile(P/n,S/n)
    for n in ['maturation_parallel.py','regime_scanner.py','materialized_discovery.py']:
        dest=S/'decoder'/n;dest.parent.mkdir(exist_ok=True);shutil.copyfile(P/'project'/'historical'/n,dest)
    for n in ['public_interface.py','current_interface.py','partitions.py']:
        shutil.copyfile(P/'project'/n,S/'decoder'/n)
    shutil.copyfile('/tmp/ig_engine0204/infinity_grid/resources/decoder/O7_MATERIAL_ROOT_cb6f48641eb9.zip',S/'O7_MATERIAL_ROOT_cb6f48641eb9.zip')
    assert sha(S/'O7_MATERIAL_ROOT_cb6f48641eb9.zip')=='cb6f48641eb99d374d19d5dc8d5bfada13e12cf1ecb20ed0e66958629ec08f1b'
    files={str(p.relative_to(S)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(S.rglob('*')) if p.is_file()}
    recovery=json.loads((S/'SOURCE_INPUT_RECOVERY.json').read_text())
    c=json.loads((P/'CATALOG_0151.json').read_text()); assert len(c['slices'])==151
    projection=next(x for x in c['slices'] if x['dataset_id']=='G1_SAVED_PUBLIC_ONE_ENDPOINT_PROJECTIONS_V1')
    payload=dict(schema_id='IG_QUALIFIED_G1_EXACT_PARENT_EXPORT_V1',dataset_id='G1_HISTORICAL_V1_TERMINAL100_EXACT_PARENT_DAG_V1',qualification_scope=A['qualification_scope'],dag_science_sha256=recovery['dag_science_sha256'],counts=dict(carriers=193,DAG_nodes=16528,interface_classes=192,reservation_rows=1351),files=files,predecessor_handoff=dict(sha256=d['sha256'],bytes=d['bytes']),prior_projection_scientific_root=projection['scientific_root_sha256'],authority='EXPORT_ALREADY_NATIVE_AND_COLD_QUALIFIED_BYTES',Q2_payload_available=False,G2_promotion=False,fresh_realization=False)
    root=hashlib.sha256(canon(payload)).hexdigest();write(S/'SEAL_MANIFEST.json',dict(payload=payload,scientific_root_sha256=root))
    out=B/'G1_HISTORICAL_V1_EXACT_PARENT_SEAL0232.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(S.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(S)))
    proposal=dict(schema_id='IG_SCOPED_ADMISSION_PROPOSAL_V1',status='PREPARED_NOT_ADMITTED',base_master_slices=151,base_catalog_sha256=sha(P/'CATALOG_0151.json'),new_admissions=0,proposed_slice=dict(dataset_id=payload['dataset_id'],scientific_root_sha256=root,format=payload['schema_id'],scope=dict(authority=payload['qualification_scope'],exact_parent_DAG_available=True,reservation_witnesses_available=False,Q2_payload_available=False,fresh_realization=False,G2_promotion=False),counts=payload['counts'],prior_projection_scientific_root=projection['scientific_root_sha256'],archive=dict(file_name=out.name,sha256=sha(out),size_bytes=out.stat().st_size)))
    write(B/'SCOPED_ADMISSION_PROPOSAL.json',proposal)
    write(B/'SEAL_RECEIPT.json',dict(path=str(out),sha256=sha(out),bytes=out.stat().st_size,scientific_root_sha256=root))
    print(json.dumps(dict(status='SEALED_PENDING_EXPORT_AUDIT',scientific_root_sha256=root)))
if __name__=='__main__':main()
