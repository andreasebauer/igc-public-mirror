"""Pure saved-hash/outcome/congruence validation, not a realization rerun."""
from pathlib import Path
import ast,collections,gzip,hashlib,json,sys,zipfile
from itertools import zip_longest
B=Path(__file__).resolve().parent;I=B/'inputs'
sha=lambda b:hashlib.sha256(b).hexdigest()
h=lambda d:sha(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
load=lambda n:json.loads((I/n).read_bytes())
def ast_function(n,name):return ast.dump(next(x for x in ast.parse((I/n).read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
def validate():
 for n,p in json.loads((B/'INPUT_PINS.json').read_bytes()).items():
  b=(I/n).read_bytes();assert sha(b)==p['sha256'] and len(b)==p['bytes']
 d4=load('d4/probe/RESULT.json');q2=load('reaudit/prior/prior_q2_probe/RESULT.json');prior=load('reaudit/prior/prior_d4_probe/RESULT.json');assert d4==prior
 assert d4['selected_candidate'] is None and [x['split_class_count'] for x in d4['candidate_results']]==[1977,4844,1977,1952]
 recon=load('recon/RESULT.json');index=load('recon/evidence/D4_RESIDUAL_INDEX.json')['groups'];assert len(index)==1952 and sum(map(len,index.values()))==3923
 assert sum(int(x[:2],16)<192 for x in index)==1458
 assert len(index)-1458==494
 features=[json.loads(x) for x in (I/'recon/evidence/RESIDUAL_FEATURE_ROWS.jsonl').read_text().splitlines()];assert len(features)==3923
 assert {r['d4_class'] for r in features}==set(index)
 assert recon['nonclaims'] and recon['classification']=='RESIDUAL_SCANNER_RECON_COMPLETE_NO_AUTOMATIC_REPAIR'
 prereg=load('reaudit/prior/prior_q2_probe/PREREGISTRATION.json')
 assert 'children' in prereg['forbidden_reads'] and q2['s1r_winner_projection_promotion_status']=='REJECTED_AS_FORBIDDEN_INTERNAL_READ_FOR_PROMOTION'
 assert q2['winner']=='Q2_ONE_RESERVATION_SUCCESSOR_SKINS'
 assert [x['all_split_classes'] for x in q2['candidate_reports']]==[1952,1952,0,0,0]
 result=load('reaudit/RESULT.json');summary=load('reaudit/evidence/REPAIRED_STREAM_SUMMARY.json');unlock=load('reaudit/S2_UNLOCK_CERTIFICATE.json')
 assert result['status']=='PASS' and result['repaired_record_count']==580351 and result['repaired_outcome_classes_split_by_realized_public_observer_count']==0
 assert result['realization_failure_count']==result['immediate_projection_mismatch_count']==0
 assert unlock['s1_repair_result_sha256']==result['sealed_science_sha256'] and unlock['authorizes']=='G2:S2_PAIR_OUTCOME_QUOTIENT_ONLY'
 assert unlock['repaired_record_stream_sha256']==result['repaired_record_stream_sha256']==summary['record_uncompressed_stream_sha256']
 current=load('CURRENT_IMPLEMENTATION_SPEC_V2.json')['s1_strict_public_repair']
 assert current['precursor_science_sha256']==d4['sealed_science_sha256'] and current['winner_science_sha256']==q2['sealed_science_sha256']
 assert ast_function('uplift_structural.py','repaired_pair_outcome_semantics_v2')==ast_function('historical_uplift_structural.py','repaired_pair_outcome_semantics_v2')
 return {'schema':'IG_SAVED_G2_S1_REPAIR_RECOVERY_V1','status':'PASS_FOR_SAVED_EVIDENCE_BINDINGS','master_release':'MASTER_DATA_V1_0149','new_admissions':0,'generation_calls':0,'D4_candidates':'ALL_FAIL_SAFE_NO_SELECTION','D4_residual_classes':1952,'D4_residual_rows':3923,'discovery_classes':1458,'holdout_classes':494,'hidden_winner_projection_promotion':'REJECTED_IN_SAVED_STRICT_PUBLIC_PREREGISTRATION','strict_public_winner':'Q2_ONE_RESERVATION_SUCCESSOR_SKINS','saved_full_reaudit_records':580351,'saved_full_reaudit_outcome_classes':576785,'saved_full_reaudit_split_classes':0,'historical_S2_unlock':'PAIR_OUTCOME_QUOTIENT_ONLY','current_repaired_semantics_AST_matches_saved':True,'current_spec_status':'CANDIDATE_PENDING_FULL_REAUDIT_RETAINED_NOT_SILENTLY_PROMOTED','fresh_realization_rerun':False,'Q2_payload_available':False,'exact_G1_parent_DAG_available':False,'full_l0_to_g8_complete':False,'next_scope':'WP6_G1_EXACT_DAG_RECOVERY_AND_SAVED_PUBLIC_EXPORT_SUFFICIENCY'}
def streams(path):
 expected=json.loads((B/'SOURCE_PINS.json').read_bytes())['reaudit'];assert sha(Path(path).read_bytes())==expected['sha256']
 continuation=load('d4/probe/evidence/POST_RESERVATION_CONTINUATION_HASHES.json')['rows'];by={r['carrier_ref']:r for r in load('G2_S0_INTERFACE_POPULATION.json')['interfaces']}
 summary=load('reaudit/evidence/REPAIRED_STREAM_SUMMARY.json');R=hashlib.sha256();A=hashlib.sha256();classes={};seen=set();counts=collections.Counter();count=0
 with zipfile.ZipFile(path) as z:
  rn='evidence/G2_S1_REPAIRED_D4_Q2_PAIR_CONNECTION_RECORDS.jsonl.gz';an='evidence/G2_S1_REPAIRED_D4_Q2_AUDIT_SIGNATURES.jsonl.gz'
  for n,key in [(rn,'record_file_sha256'),(an,'audit_file_sha256')]:
   hh=hashlib.sha256()
   with z.open(n) as f:
    for chunk in iter(lambda:f.read(1024*1024),b''):hh.update(chunk)
   assert hh.hexdigest()==summary[key]
  with gzip.GzipFile(fileobj=z.open(rn)) as rf,gzip.GzipFile(fileobj=z.open(an)) as af:
   for rline,aline in zip_longest(rf,af):
    assert rline is not None and aline is not None
    R.update(rline);A.update(aline);r=json.loads(rline);a=json.loads(aline)
    left,right=r['left_carrier_ref'],r['right_carrier_ref'];t,u=map(int,r['connection_operator_ref'].rsplit(':',1)[1].split('>'))
    assert left<=right and r['left_interface_sha256']==by[left]['interface_sha256'] and r['right_interface_sha256']==by[right]['interface_sha256']
    assert r['d4_left_post_reservation_public_continuation_sha256']==continuation[left][str(t)] and r['d4_right_post_reservation_public_continuation_sha256']==continuation[right][str(u)]
    assert r['legality']=='LEGAL' and r['realization_status']=='PASS' and r['q2_projection_serialized'] is False
    payload={'schema_id':'IG_G_UPLIFT_S1_REPAIRED_PAIR_OUTCOME_SEMANTICS_V2','projected_outcome_science_sha256':r['projected_outcome_science_sha256'],'connection_operator_ref':r['connection_operator_ref'],'left_post_reservation_public_continuation_sha256':r['d4_left_post_reservation_public_continuation_sha256'],'right_post_reservation_public_continuation_sha256':r['d4_right_post_reservation_public_continuation_sha256'],'q2_one_reservation_successor_skins_sha256':r['q2_one_reservation_successor_skins_sha256'],'repair_basis':'D4_STRICT_PUBLIC_PRECURSOR_PLUS_Q2_STRICT_PUBLIC_RESIDUAL_WINNER','observer_scope':'ONE_STEP_REALIZED_PUBLIC_CONGRUENCE_CANDIDATE'}
    digest=h(payload);assert digest==r['outcome_science_sha256']==a['outcome_science_sha256'] and r['outcome_ref']=='G2S1O2:'+digest
    observer=a['realized_public_sha256'];assert classes.setdefault(digest,observer)==observer
    seen.add((left,right,t,u));counts[digest]+=1;count+=1
    if count%100000==0:print('Verified saved repaired rows:',count,file=sys.stderr,flush=True)
 assert count==len(seen)==580351 and len(classes)==576785
 assert sum(n>1 for n in counts.values())==3554 and max(counts.values())==3
 assert R.hexdigest()==summary['record_uncompressed_stream_sha256'] and A.hexdigest()==summary['audit_uncompressed_stream_sha256']
 operators=collections.Counter((t,u) for _,_,t,u in seen);assert len(operators)==31 and set(operators.values())=={18721}
 assert h([list(x) for x in sorted(operators)])=='f10a566c1f8e8faf7419cca50e1ee84977c388beb9c5d2948e01290b68034e1a'
 return {'status':'PASS_SAVED_REPAIRED_OUTCOME_BINDINGS_AND_STORED_OBSERVER_CONGRUENCE','record_rows_checked':count,'audit_signature_rows_checked':count,'outcome_classes_checked':len(classes),'stored_observer_split_classes':0,'collision_classes':3554,'maximum_class_size':3,'record_stream_sha256':R.hexdigest(),'audit_stream_sha256':A.hexdigest(),'Q2_hashes_role':'OPAQUE_SAVED_PROJECTION_HASHES_NO_SERIALIZED_PAYLOAD','stored_observer_hashes_role':'SAVED_SIGNATURES_NOT_RECOMPUTED_FROM_CARRIERS'}
if __name__=='__main__':
 result=validate()
 if len(sys.argv)>1:result['stream_validation']=streams(sys.argv[1])
 print(json.dumps(result,indent=2))
