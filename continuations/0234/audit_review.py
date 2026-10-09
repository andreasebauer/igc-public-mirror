"""Independent pin/scope audit of the full stream review, without regenerating data."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent;W=B.parent
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=json.loads((B/'REVIEW.json').read_text());m=json.loads((B/'INDEX_MANIFEST.json').read_text());p=json.loads((B/'PREREGISTRATION.json').read_text());cat=json.loads((W/'continuation0233/CATALOG_0152.json').read_text())
assert len(cat['slices'])==152 and cat['slices'][-1]['scientific_root_sha256']==r['admitted_G1_scientific_root_sha256']==p['admitted_G1_scientific_root_sha256']
assert sha(W/'continuation0233/CATALOG_0152.json')==p['catalog152_sha256']
for name in ['records.jsonl.gz','audits.jsonl.gz']:
    f=B/'data'/name;assert f.stat().st_size==m['files'][name]['bytes'] and sha(f)==m['files'][name]['sha256']
assert r['record_uncompressed_sha256']==m['record_uncompressed_sha256'] and r['audit_uncompressed_sha256']==m['audit_uncompressed_sha256']
assert r['G1_refs_joined']==193 and r['record_rows_checked']==r['audit_rows_checked']==580351 and r['stored_observer_split_classes']==0
assert r['pairs']==193*194//2 and r['record_rows_checked']==r['pairs']*r['bridge_operators']
assert r['interface_hash_bindings_checked']==r['stored_D4_bindings_checked']==2*r['record_rows_checked']
for k in ['Q2_payload_available','fresh_realization','G2_promotion','source_authorities_rewritten']:assert r[k] is False
assert r['master_slices']==152 and r['new_admissions']==r['generation_calls']==r['new_DAG_decodes']==0
ad=json.loads((W/'continuation0233/SCOPED_ADMISSION.json').read_text());assert ad['scope']['authority']=='HISTORICAL_V1_G1_TERMINAL_REPLAY_ONLY'
old=next(x for x in cat['slices'] if x['dataset_id']=='G2_SAVED_REPAIRED_S1_RECORDS_AND_STORED_AUDITS_V1');assert old['scope']['exact_G1_parent_DAG_available'] is False and old['scope']['fresh_realization'] is False
out=dict(status='PASS_INDEPENDENT_STREAM_PIN_AND_SCOPE_AUDIT',review_sha256=sha(B/'REVIEW.json'),preregistration_sha256=sha(B/'PREREGISTRATION.json'),audit_scope='Pin and scope audit of complete primary stream review; no second row-level replay claimed',master_slices=152,new_admissions=0,G2_promotion=False,generation_calls=0)
(B/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
