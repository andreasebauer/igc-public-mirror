"""Audit three official chunks; candidate credit needs final exact saved batch."""
import hashlib,json,pathlib
B=pathlib.Path(__file__).parent;W=B.parent;Q=W/'continuation0285'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    reg=json.loads((Q/'PREREGISTRATION.json').read_text());rows=[];ordinals=set();classes=set();members=0;peak=0
    for idx in range(6):
        D=(B if idx>=3 else W/'continuation0287')/f'{idx:03d}';row=reg['specifications'][idx];s=json.loads((D/'SPEC.json').read_text());n=json.loads((D/'NATIVE_RESULT.json').read_text());c=json.loads((D/'COLD_AUDIT.json').read_text());p=json.loads((D/'CHECKPOINT_PRESERVED.json').read_text());r=json.loads((D/'UNIT_SAVE_RECEIPT.json').read_text())
        assert sha(D/'SPEC.json')==row['spec_sha256'] and n['status']=='COMPLETED' and n['evidence_status']=='VERIFIED'
        assert n['result']['cases_checked']==c['cases_exact']==row['case_count'] and c['complete_observer_classes_exact']==row['class_count']
        assert c['full_public_observer_payloads_hashes_Q2_and_operational_witnesses_exact'] and c['peak_RSS_bytes']<=reg['memory_budget_bytes']
        assert not p['pending_objects'] and p['pending_bytes']==0 and r['status']=='PASS_NATIVE_COLD_AND_EXACT_UNIT_SAVED'
        for x in r['artifacts']:assert x['drive_exact_readback_verified'] and sha(pathlib.Path(x['path']))==x['sha256']
        inp=next(x for x in s['inputs'] if x['logical_name']=='cases');cases=json.loads(pathlib.Path(inp['path']).read_text());ids={x['ordinal'] for x in cases['cases']};keys=set(cases['class_keys'])
        assert not ids&ordinals and not keys&classes;ordinals|=ids;classes|=keys;members+=row['case_count'];peak=max(peak,c['peak_RSS_bytes'])
        rows.append({'chunk_index':idx,'case_count':row['case_count'],'class_count':row['class_count'],'unit_directory':str(D),'specification_sha256':sha(D/'SPEC.json'),'native_result_sha256':sha(D/'NATIVE_RESULT.json'),'cold_audit_sha256':sha(D/'COLD_AUDIT.json'),'checkpoint_preserved_sha256':sha(D/'CHECKPOINT_PRESERVED.json'),'unit_save_receipt_sha256':sha(D/'UNIT_SAVE_RECEIPT.json')})
    assert members==len(ordinals)==372 and len(classes)==186
    out={'checkpoint':288,'status':'PASS_OFFICIAL_NATIVE_COLD_UNITS_FINAL_BATCH_SAVE_PENDING','registration_sha256':sha(Q/'PREREGISTRATION.json'),'completed_chunks':rows,'pending_chunk_indices':list(range(6,114)),'next_chunk_index':6,'official_members_compared':members,'official_complete_classes_compared':len(classes),'remaining_members':reg['selected_members']-members,'remaining_classes':reg['selected_complete_classes']-len(classes),'remaining_chunks':108,'dry_coverage_credit':0,'peak_cold_RSS_bytes':peak,'master_slices':152,'new_admissions':0,'G2_promotion':False,'full_historical_public_observer_compared':False,'canonicalization_impact_assessed':False,'next_scope':'OFFICIAL_OBSERVER_CHUNKS_006_008_NATIVE_SAVE_COLD_THEN_LEDGER'}
    (B/'BATCH_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(B/'LEDGER_CANDIDATE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['completed_chunks','pending_chunk_indices']}))
if __name__=='__main__':main()
