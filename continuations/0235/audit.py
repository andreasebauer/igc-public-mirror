"""Independent saved-payload, registration and scope checks; no scientific replay."""
from pathlib import Path
import hashlib,json,sys
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid.canon import canonical_sha256
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pre=json.loads((B/'PREREGISTRATION.json').read_text());spec=json.loads((B/'SPEC.json').read_text())
    for n,h in pre['source_sha256'].items():assert sha(Path('/tmp/ig_engine0204/infinity_grid/uplift_structural.py') if n.startswith('engine/') else B/n)==h
    assert pre['official_case_count']==62 and pre['dry_case_ordinals']==[0] and pre['official_execution_in_this_step'] is False
    out={'status':'PASS_REGISTRATION_AND_SAVED_DRY_PAYLOAD_AUDIT','scope':'No second scientific replay or cold native qualification','preregistration_sha256':sha(B/'PREREGISTRATION.json'),'master_slices':152,'new_admissions':0,'G2_promotion':False,'official_cases_executed':0}
    if (B/'STOPPED.txt').exists():
        out['dry_status']='STOPPED';out['next_scope']='DIAGNOSE_REGISTERED_DRY_PREFLIGHT_FAILURE'
    else:
        d=json.loads((B/'DRY_CASE.json').read_text());r=json.loads((B/'DRY_RESULT.json').read_text());case=json.loads((B/'CASES.json').read_text())['cases'][0];saved=case['record'];q=d['public_projection']
        assert d['source_case_sha256']==canonical_sha256(case)
        assert canonical_sha256([{'t':x['endpoint_type'],'available':x['available'],'successor_skin':x['successor_boundary_resource_skin_sha256']} for x in q['rows']])==q['science_sha256']==saved['q2_one_reservation_successor_skins_sha256']
        assert d['projected_outcome_sha256']==saved['projected_outcome_science_sha256'] and d['repaired_outcome_sha256']==saved['outcome_science_sha256']
        assert d['whole_carrier_swap_q2_exact'] is True and q['hidden_internal_reads'] is False and q['reservation_witnesses_retained'] is False
        assert [x['endpoint_type'] for x in d['operational_witness']['Q2_reservation_witnesses']]==[x['endpoint_type'] for x in q['rows'] if x['available']]
        assert r['official_cases_executed']==0 and r['dry_cases_checked']==1 and r['peak_RSS_bytes']<=spec['resources']['memory_budget_bytes']
        out.update(dry_status=r['status'],dry_result_sha256=sha(B/'DRY_RESULT.json'),dry_case_sha256=sha(B/'DRY_CASE.json'),next_scope='NATIVE_CAPTURE_AND_OFFICIAL_BOUND62_HISTORICAL_S1_Q2_PROBE')
    (B/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
