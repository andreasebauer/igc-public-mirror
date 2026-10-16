"""Native qualification of installed original saved-record RunPlan admission."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import ast,hashlib,json,os,tempfile,zipfile
from infinity_grid import saved_g2_g3_read_plan as reader
from infinity_grid.run_plan_v2 import compile_run_plan_v2,execute_original_g2_g3_saved_read_run_plan,RunPlanV2Error
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid import semantic_sentinel,theorem_registry

def main():
 inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text());paths={k:inputs[k] for k in reader.PINS}
 parent=Path(inputs['q41']);assert hashlib.sha256(parent.read_bytes()).hexdigest()=='e51b10c81b81c1bdada1e0df5e2382bb5685c57d04912071ec19bfd8522b9785'
 with zipfile.ZipFile(parent) as z:
  c=json.loads(z.read('native/CAPTURE_RESPONSE.json'));d=json.loads(z.read('native/NATIVE_RESULT.json'));proof=json.loads(z.read('native/TERMINAL_COMPLETION_PROOF.json'))
  assert c['capture_id']=='9e3cd7dbf4888987b72d2bf8d3933fe39135e3ab75ac7dbfe926571194811ee1'
  assert d['completion_sha256']==canonical_sha256({k:v for k,v in d.items() if k not in ('completion_sha256','reused','preservation')})=='8adac01e166dda4ef242779e561ba25132cdd08331f5a714d6a21cc61a4edd56'
  assert d['status']=='COMPLETED' and d['evidence_status']=='VERIFIED' and proof['terminal_completion_preserved'] is True
  references={k:json.loads(z.read('accepted/saved_records/'+k+'_PRIMARY_RESULT.json')) for k in reader.PINS}
  # Bind expected records to Q41's accepted native evidence hashes, not a status label.
  evidence={x['path']:x for x in d['evidence']}
  for k in reader.PINS:
   raw=z.read('accepted/saved_records/'+k+'_PRIMARY_RESULT.json');e=evidence['script/work/work/saved_records/'+k+'_PRIMARY_RESULT.json']
   assert hashlib.sha256(raw).hexdigest()==e['sha256'] and len(raw)==e['size_bytes']
 plan={'schema_id':reader.SCHEMA,'route':reader.ROUTE,'requests':[{'kind':'saved_result','source_scope':k} for k in reader.PINS]+[{'kind':'s0_challenge_records','source_scope':'s0'},{'kind':'s1_target_summaries','source_scope':'s1'}],'generation_allowed':False,'production_authority':'NONE'}
 before=semantic_sentinel.verify_native_semantics(raise_on_change=False);stale=theorem_registry.verify_all_theorems(raise_on_stale=False)
 assert before['status']=='REOPEN_REQUIRED' and len(stale['stale_theorems'])==4
 # Producer/theorem functions are blocked while the actual installed route executes.
 from infinity_grid import uplift_g3_s0,uplift_g3_s1,regime_scanner,g2_relation,uplift_campaign
 blocked=[]
 def no_generation(*args,**kw):blocked.append('FORBIDDEN_CALL');raise AssertionError('producer/theorem called')
 targets=[(uplift_g3_s0,'run_g3_s0'),(uplift_g3_s1,'reproduce_s0_challenge_states'),(g2_relation,'compose_binary_relation'),(theorem_registry,'require_current_theorems'),(semantic_sentinel,'current_action_contract')]
 from contextlib import ExitStack
 with ExitStack() as stack:
  for obj,name in targets:stack.enter_context(patch.object(obj,name,no_generation))
  stack.enter_context(patch.object(uplift_campaign.UpliftCampaignEngine,'ensure_g1_r100_population',no_generation))
  compiled=compile_run_plan_v2(plan)
  result=execute_original_g2_g3_saved_read_run_plan(plan,archives=paths)
 assert compiled==result['compiled'] and not blocked
 for i,k in enumerate(reader.PINS):assert result['observations'][i]['payload']==references[k]
 assert result['observations'][3]['payload']==references['s0']['challenge_corpus']['records']
 assert result['observations'][4]['payload']==references['s1']['target_behavior_summaries']
 assert len(result['observations'][3]['payload'])==len(result['observations'][4]['payload'])==6
 # Read copies may not mutate subsequent stored results, including huge integer values.
 old=deepcopy(result['observations'][3]['payload']);result['observations'][3]['payload'][0]['public_interface']['coordinates'][0]=True
 again=execute_original_g2_g3_saved_read_run_plan(plan,archives=paths);assert again['observations'][3]['payload']==old
 assert all(type(v) is int for r in old for v in r['public_interface']['coordinates']) and max(v for r in old for v in r['public_interface']['coordinates'])>2**63
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (reader.SavedAuthorityReadError,RunPlanV2Error):refusals.append(name)
  else:raise AssertionError('REFUSAL_NOT_ENFORCED:'+name)
 stronger=['full_implementation_carriers','full_context_payloads','current_cohort_execution','regenerate_missing','topology_promotion']
 with patch.object(reader,'_load_archives',side_effect=AssertionError('archive IO before request refusal')):
  for kind in stronger:
   bad=deepcopy(plan);bad['requests']=[{'kind':kind,'source_scope':'s0'}];refuse(kind,lambda bad=bad:execute_original_g2_g3_saved_read_run_plan(bad,archives={}))
  for field,value in [('generation_allowed',True),('generation_allowed',0),('production_authority','MASTER'),('route','OTHER')]:
   bad=deepcopy(plan);bad[field]=value;refuse(field+'_'+str(value),lambda bad=bad:execute_original_g2_g3_saved_read_run_plan(bad,archives={}))
  for name,requests in [('wrong_scope',[{'kind':'s0_challenge_records','source_scope':'s1'}]),('bool_kind',[{'kind':True,'source_scope':'s0'}]),('extra_request_field',[{'kind':'saved_result','source_scope':'g2','execute':True}]),('empty',[]),('duplicate',[plan['requests'][0]]*2)]:
   bad=deepcopy(plan);bad['requests']=requests;refuse(name,lambda bad=bad:execute_original_g2_g3_saved_read_run_plan(bad,archives={}))
 refuse('frontier_adoption',lambda:compile_run_plan_v2(plan,frontier={}))
 refuse('old_calibration',lambda:compile_run_plan_v2(plan,saved_read_calibration={}))
 refuse('missing_archive',lambda:execute_original_g2_g3_saved_read_run_plan(plan,archives={'s0':inputs['s0']}))
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'changed';p.write_bytes(Path(inputs['s0']).read_bytes()+b'x');badpaths=dict(paths,s0=str(p));refuse('changed_archive',lambda:execute_original_g2_g3_saved_read_run_plan(plan,archives=badpaths))
 # Static route surface has no dynamic loading/eval or producer imports.
 tree=ast.parse(Path(reader.__file__).read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
 assert set(imports)=={'copy','io','pathlib','canon'}
 assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==before
 assert theorem_registry.verify_all_theorems(raise_on_stale=False)==stale
 write_json_atomic('INSTALLED_SAVED_READ_RESULT.json',again)
 write_json_atomic('REFUSALS.json',{'status':'PASS','count':len(refusals),'cases':refusals,'stronger_requests_refused_before_archive_io':True})
 write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'INSTALLED_ORIGINAL_G2_G3_SAVED_RECORD_ADMISSION_QUALIFIED','installed_runplan_dispatch':True,'reads':5,'challenge_records':6,'target_summaries':6,'reported_context_rows':2232,'refusal_count':len(refusals),'generation_calls':len(blocked),'full_carrier_execution_authorized':False,'stale_theorems':4,'semantic_status':'REOPEN_REQUIRED','original_producer_identities_preserved':True,'production_authority':'NONE','package2_complete':False,'parent_capture_id':c['capture_id'],'parent_completion_sha256':d['completion_sha256']})
if __name__=='__main__':main()
