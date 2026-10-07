"""Verify admitted catalog and native saved integration bindings."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def validate():
 load=lambda n:json.loads((B/n).read_bytes())
 cat=load('CATALOG_0150.json');old=load('CATALOG_0149.json');ad=load('SCOPED_ADMISSION.json');n=load('NATIVE_RESULT.json');cold=load('COLD_REUSE_RESULT.json')
 assert sha((B/'CATALOG_0149.json').read_bytes())=='84641ecb46ea253c9a8cbbeaa418c6cd14bda93512a763c583ce3ed3af559f7c'
 assert cat['slices'][:-1]==old['slices'] and len(old['slices'])==149 and len(cat['slices'])==150
 s=cat['slices'][-1];assert s['scoped_admission_sha256']==sha((B/'SCOPED_ADMISSION.json').read_bytes())
 assert s['scope']==ad['scope'] and s['counts']==ad['counts'] and ad['decision']=='ACCEPTED_FOR_REUSE_WITHIN_SAVED_G1_PUBLIC_PROJECTION_SCOPE'
 assert s['scope']['authority']=='SAVED_PUBLIC_PROJECTION_ONLY' and s['scope']['exact_parent_DAG_available'] is False and s['scope']['Q2_payload_available'] is False
 sources=load('PREDECESSOR_SOURCE_HASHES.json');base=B/'project/previous'
 assert set(sources)=={str(p.relative_to(base)) for p in base.rglob('*.py')}
 for p,h in sources.items():assert sha((base/p).read_bytes())==h
 assert n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED'
 r=n['result'];assert r['outcome']=='PASS' and r['scientific_slices']==150 and r['prior_slices_unchanged']==149 and r['prior_reader_bytes_unchanged'] is True
 for field,want in {'all_g1_public_interface_routes_checked':193,'all_g1_public_class_routes_checked':192,'all_g1_public_reservation_routes_checked':1351,'all_g1_public_continuation_routes_checked':1351,'prior_additional_o7_delegation_routes_checked':24,'generation_calls':0}.items():assert r[field]==want
 assert cold['reused'] and cold['pending_bytes']==0 and cold['completion_sha256']==n['completion_sha256'] and cold['result_sha256']==n['result_sha256']
 assert sha((B/'SCIENTIFIC_EXPORT.zip').read_bytes())==s['archive']['sha256']
 return {'status':'PASS_PROJECTION_SCOPED_MASTER150_INTEGRATION','release_id':'MASTER_DATA_V1_0150','scientific_slices':150,'prior_slices_unchanged':149,'prior_reader_modules_unchanged':len(sources),'G1_public_projection':ad['counts'],'exact_G1_parent_DAG_available':False,'Q2_payload_available':False,'cold_reuse':True,'pending_bytes':0,'generation_calls':0,'full_l0_to_g8_complete':False,'next_scope':'WP6_SAVED_REPAIRED_G2_S1_RECORD_EXPORT_READINESS'}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
