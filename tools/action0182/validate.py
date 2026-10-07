"""Small bound readiness checks reuse already completed full-stream validation."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent;I=B/'inputs'
sha=lambda b:hashlib.sha256(b).hexdigest()
semantic=lambda d:sha(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
def validate():
 for n,p in json.loads((B/'INPUT_PINS.json').read_text()).items():
  b=(I/n).read_bytes();assert sha(b)==p['sha256'] and len(b)==p['bytes']
 load=lambda n:json.loads((I/n).read_bytes())
 catalog=load('CATALOG_0150.json');ad=load('G1_SCOPED_ADMISSION.json');last=catalog['slices'][-1]
 assert catalog['release_id']=='MASTER_DATA_V1_0150' and len(catalog['slices'])==150
 assert last['scoped_admission_sha256']==sha((I/'G1_SCOPED_ADMISSION.json').read_bytes()) and last['scope']==ad['scope']
 assert ad['decision']=='ACCEPTED_FOR_REUSE_WITHIN_SAVED_G1_PUBLIC_PROJECTION_SCOPE' and ad['scope']['authority']=='SAVED_PUBLIC_PROJECTION_ONLY'
 native=load('G1_INTEGRATION_RESULT.json');cold=load('G1_COLD_REUSE.json');assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and cold['reused'] and cold['completion_sha256']==native['completion_sha256'] and cold['result_sha256']==native['result_sha256'] and cold['pending_bytes']==0
 prior=load('REPAIR_VALIDATION.json')['stream_validation'];summary=load('REPAIRED_STREAM_SUMMARY.json');result=load('REAUDIT_RESULT.json');unlock=load('S2_UNLOCK_CERTIFICATE.json')
 assert prior['record_rows_checked']==prior['audit_signature_rows_checked']==580351 and prior['outcome_classes_checked']==576785 and prior['stored_observer_split_classes']==0
 assert prior['record_stream_sha256']==summary['record_uncompressed_stream_sha256']=='9023c8d4a53ff6bba87a5b6954bac8d37008257aaf5214c67c0c6ba8ddf28049'
 assert prior['audit_stream_sha256']==summary['audit_uncompressed_stream_sha256']=='7ff1a9ce2a63f99928e07d82f42c29feebda27122fd5629e4bbf2b796d52283d'
 assert result['status']=='PASS' and unlock['authorizes']=='G2:S2_PAIR_OUTCOME_QUOTIENT_ONLY' and unlock['s1_repair_result_sha256']==result['sealed_science_sha256']
 transport=load('TRANSPORT.json');pin=load('SOURCE_PINS.json')['reaudit'];assert transport['archive_sha256']==pin['sha256'] and sum(x['bytes'] for x in transport['parts'])==pin['bytes'] and len(transport['parts'])==5
 source=json.loads((B/'SOURCE_BINDINGS.json').read_text());assert source['archive_sha256']==pin['sha256'] and source['archive_bytes']==pin['bytes'] and source['archive_hash_freshly_verified'] is True
 assert source['members']['records']['gzip_bytes']==94212112 and source['members']['audits']['gzip_bytes']==45700750
 contract=json.loads((B/'EXPORT_CONTRACT.json').read_text());assert contract['record_identity']==['left_carrier_ref','right_carrier_ref','connection_operator_ref'] and contract['new_admission_authorized_by_contract'] is False
 preview=json.loads((B/'FIRST_ROW_BINDING_PREVIEW.json').read_text());r=preview['records']['payload'];a=preview['audits']['payload'];assert r['outcome_science_sha256']==a['outcome_science_sha256'] and r['q2_projection_serialized'] is False
 pop=load('G2_S0_INTERFACE_POPULATION.json');by={r['carrier_ref']:r for r in pop['interfaces']};cont=load('POST_RESERVATION_CONTINUATION_HASHES.json')['rows']
 assert sha((I/'G2_S0_INTERFACE_POPULATION.json').read_bytes())=='6b7405f8e3374061f167cecd20225db4f54f22ad45ed8ac97cace64f1fe8f666'
 assert sha((I/'POST_RESERVATION_CONTINUATION_HASHES.json').read_bytes())=='35bb6899ab4d1fae4da11f12393658a233f70f3ad7a3dfaad95e617424499d85'
 assert last['archive']['sha256']=='e7f850c41c8c496b74d9de4e0e9d6f4bf3b674cf95fbf141c6dcfba1f7693231'
 payload={'schema_id':'IG_G_UPLIFT_S1_REPAIRED_PAIR_OUTCOME_SEMANTICS_V2','projected_outcome_science_sha256':r['projected_outcome_science_sha256'],'connection_operator_ref':r['connection_operator_ref'],'left_post_reservation_public_continuation_sha256':r['d4_left_post_reservation_public_continuation_sha256'],'right_post_reservation_public_continuation_sha256':r['d4_right_post_reservation_public_continuation_sha256'],'q2_one_reservation_successor_skins_sha256':r['q2_one_reservation_successor_skins_sha256'],'repair_basis':'D4_STRICT_PUBLIC_PRECURSOR_PLUS_Q2_STRICT_PUBLIC_RESIDUAL_WINNER','observer_scope':'ONE_STEP_REALIZED_PUBLIC_CONGRUENCE_CANDIDATE'}
 assert semantic(payload)==r['outcome_science_sha256'] and r['outcome_ref']=='G2S1O2:'+r['outcome_science_sha256']
 for side,t in [('left',0),('right',0)]:
  ref=r[side+'_carrier_ref'];assert r[side+'_interface_sha256']==by[ref]['interface_sha256'] and r['d4_'+side+'_post_reservation_public_continuation_sha256']==cont[ref][str(t)]
 assert r['strict_public_repair_status']=='D4_PLUS_Q2_CANDIDATE_PENDING_FULL_580351_CONGRUENCE_REAUDIT'
 return {'status':'PASS_SAVED_REPAIRED_S1_EXPORT_READINESS','master_release':'MASTER_DATA_V1_0150','scientific_slices':150,'counts':contract['counts'],'input_G1_projection_admission_bound':True,'historical_row_pending_label_retained':True,'later_saved_reaudit_PASS_separate':True,'Q2_payload_available':False,'fresh_realization':False,'prior_full_stream_checks_reused':True,'full_stream_checks_reexecuted':False,'readiness_preview_rows':1,'generation_calls':0,'new_admissions':0,'next_scope':'WP6_REGISTERED_SAVED_G2_S1_RECORD_READER_AND_INDEX_EXPORT'}
if __name__=='__main__':print(json.dumps(validate(),indent=2))
