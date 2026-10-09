"""Independent exact next-spec/source/case gates, with remaining gates explicit."""
from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0237')
from infinity_grid.canon import canonical_sha256
from infinity_grid.result_contracts import normalize
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((B/'NEXT_PREREGISTRATION.json').read_text());s=json.loads((B/'NEXT_SPEC.json').read_text());c=json.loads((B/'PILOT_CASES.json').read_text());g=json.loads((B/'NEXT_PREFLIGHT.json').read_text())
 assert 'output_checks' not in s and s['output_contract']['schema_id']=='IG_DECODER_RESULT_CONTRACT_V1'
 normalized=normalize(s['output_contract'],s['execution'],s['question']);assert normalized['claim']=='EXECUTION_ONLY' and normalized['declared_outcomes']==['PASS_COMPLETE_CLASS_Q2_PILOT_V1']
 for n,h in p['source_sha256'].items():
  path=Path('/tmp/ig_engine0237/infinity_grid')/n.split('/',1)[1] if n.startswith('engine/') else B/n
  assert sha(path)==h,n
 prior=json.loads((B.parent/'continuation0235/PREREGISTRATION.json').read_text())['source_sha256']
 for name,h in prior.items():
  if name!='project/handler.py':assert p['source_sha256'][name]==h
 assert p['source_sha256']['engine/workflow_guard.py']==json.loads((B.parent/'continuation0237/REPAIR_TESTS.json').read_text())['guard_after_sha256']
 assert p['source_binding']==canonical_sha256(p['source_sha256'])==s['execution']['parameters']['source_binding']
 for x in s['inputs']:assert sha(Path(x['path']))==x['sha256']==p['input_sha256'][x['logical_name']]
 checked={x['pointer']:x['equals'] for x in s['output_contract']['result_checks']};assert checked['/outcome']=='PASS_COMPLETE_CLASS_Q2_PILOT_V1' and checked['/cases_checked']==62 and checked['/complete_Q2_classes_checked']==31 and checked['/full_historical_public_observer_compared'] is False
 assert checked['/new_admissions']==0 and checked['/G2_promotion'] is False
 first=c['class_keys'][0];assert p['dry_case_ordinals']==[x['ordinal'] for x in c['cases'] if x['record']['outcome_science_sha256']==first] and len(p['dry_case_ordinals'])==2
 assert g['modules_checked']==22 and p['official_cases_executed']==0
 out=dict(status='PASS_INDEPENDENT_NEXT_Q2_PILOT_REGISTRATION',specification_sha256=sha(B/'NEXT_SPEC.json'),preregistration_sha256=sha(B/'NEXT_PREREGISTRATION.json'),cases_sha256=sha(B/'PILOT_CASES.json'),all_scientific_helpers_unchanged=True,cases=62,complete_classes=31,dry_complete_class_members=2,official_executed=0,resource_gate='PENDING_DRY',full_historical_observer_gate='BLOCKED_EXACT_SERIALIZER_BINDING',master_slices=152,new_admissions=0,G2_promotion=False)
 (B/'NEXT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
