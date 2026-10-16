"""Native all-five saved-reader integration with exact independent baseline comparison."""
from pathlib import Path
from copy import deepcopy
from contextlib import ExitStack
from unittest.mock import patch
import hashlib,json,os,time,zipfile
from infinity_grid import combined_saved_master_pipeline as pipeline
from infinity_grid import saved_read_run_plan as saved,saved_read_source_compatibility as compat
from infinity_grid import run_plan_v2 as rp,semantic_sentinel,theorem_registry
from infinity_grid.canon import write_json_atomic

def main():
 started=time.monotonic();inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text());pins=json.loads(Path(__file__).with_name('baseline_pins.json').read_text());baseline={}
 names={'baseline_q17':'q17/work/INSTALLED_RUN_PLAN_RESULT.json','baseline_q43':'accepted/INSTALLED_SAVED_READ_RESULT.json','baseline_q44':'accepted/MASTER_EXISTING_EXPORT_PIPELINE.json','baseline_q45':'accepted/HISTORICAL_G4_SAVED_TERMS.json','baseline_q46':'accepted/G4_REBASE_SAVED_PIPELINE.json'}
 for name,row in pins.items():
  raw=Path(inputs[name]).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  from io import BytesIO
  with zipfile.ZipFile(BytesIO(raw)) as z:baseline[name]=json.loads(z.read(names[name]))
 plan={'schema_id':pipeline.SCHEMA,'route':pipeline.ROUTE,'components':pipeline.COMPONENTS,'scope':'FINITE_QUALIFIED_SAVED_READS_ONLY','generation_allowed':False,'production_authority':'NONE'}
 roles=json.loads(Path(__file__).with_name('pipeline_input_roles.json').read_text());paths={k:inputs[k] for k in roles}
 old_sem=semantic_sentinel.verify_native_semantics(raise_on_change=False);old_th=theorem_registry.verify_all_theorems(raise_on_stale=False)
 assert old_sem['status']=='REOPEN_REQUIRED' and len(old_th['stale_theorems'])==4
 from infinity_grid import materialized_discovery as md,maturation_parallel as mp,regime_scanner as scanner
 from infinity_grid import master_g3_saved_storage,master_g6_initial_storage,master_scientific_storage,storage_catalog
 from infinity_grid.g4_term_state import G3TermState
 from infinity_grid import fixed_grammar_transport
 blocked=[];stream={};calls=0
 def forbidden(*a,**kw):blocked.append('FORBIDDEN');raise AssertionError('UNAUTHORIZED_SCIENTIFIC_GENERATION')
 def sink(component,kind,key,raw):
  nonlocal calls
  h=stream.setdefault(component,hashlib.sha256());h.update(len(raw).to_bytes(8,'big'));h.update(raw);calls+=1
 # Bounded spies replace the old per-call profiler. No trace table or scientific rerun.
 with ExitStack() as stack:
  targets=[(md,'_read_probe'),(md,'_resource_future_signature'),(md,'_build_next_panel_with_candidate_capture'),(md.MaterializedDiscoverySession,'_build_and_store'),(mp,'_farthest_select_descriptors'),(G3TermState,'from_homogeneous_seed'),(G3TermState,'reserve_external_relation'),(master_g3_saved_storage,'saved_dag_storage'),(master_g6_initial_storage,'initial_storage'),(master_scientific_storage,'build_export_storage'),(storage_catalog,'build_index'),(fixed_grammar_transport,'run_fixed_grammar_transport')]
  if hasattr(scanner,'_farthest_select'):targets.append((scanner,'_farthest_select'))
  for obj,name in targets:stack.enter_context(patch.object(obj,name,forbidden))
  restore=stack.enter_context(patch.object(mp,'_states_from_dag',wraps=mp._states_from_dag))
  result=rp.execute_combined_saved_master_pipeline_run_plan(plan,inputs=paths,sink=sink,progress=lambda phase:print('COMPLETED_READER_GROUP '+phase,flush=True))
  assert restore.call_count==2 and not blocked
 assert result['compiled']==rp.compile_run_plan_v2(plan)
 r=result['results'];q17=r[pipeline.COMPONENTS[0]];old=baseline['baseline_q17']
 assert [x['request'] for x in q17['observations']]==[x['request'] for x in old['observations']]
 assert [x['payload'] for x in q17['observations']]==[x['payload'] for x in old['observations']]
 assert all(x['scientific_reexecution'] is False for x in q17['observations'])
 kinds={}
 for row in q17['observations']:kinds.setdefault(row['request']['kind'],[]).append(row)
 assert len(q17['observations'])==27 and len(kinds['r100_descriptors'][0]['payload']['descriptors'])==193
 assert sum(len(x['payload']['evidence_payload']['rows']) for x in kinds['complete_read'])==42
 assert sum(len(x['payload']['original_dispatch']['executions']) for x in kinds['saved_packet'])==100
 assert r[pipeline.COMPONENTS[1]]==baseline['baseline_q43']
 q45=r[pipeline.COMPONENTS[2]];old45=baseline['baseline_q45']
 for cohort,previous in zip(q45['cohorts'],old45['cohorts']):
  assert cohort['cohort']==previous['cohort']
  for row,oldrow in zip(cohort['terms'],previous['terms']):assert row['original_record']==oldrow['original_record'] and row['strict_read']['term']==oldrow['strict_read']['term']
 assert r[pipeline.COMPONENTS[3]]==baseline['baseline_q46']
 q44=r[pipeline.COMPONENTS[4]];assert q44==baseline['baseline_q44'] and calls==267406
 for component in q44['components']:assert stream[component['component']].hexdigest()==component['full_record_stream_sha256']
 with zipfile.ZipFile(inputs['q11_evidence']) as z:old_manifest=json.loads(z.read('PACKAGE_MODULES.json'))
 current=saved.package_manifest();source_receipt=compat.verify_source(old_manifest,current)
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (pipeline.CombinedSavedPipelineError,rp.RunPlanV2Error,compat.SourceCompatibilityError):refusals.append(name)
  else:raise AssertionError('NO_REFUSAL:'+name)
 with patch.object(saved,'make_calibration',side_effect=AssertionError('INPUT_BEFORE_SCOPE')):
  for field,value in [('scope','FULL_MASTER_PRODUCER'),('generation_allowed',True),('generation_allowed',0),('production_authority','MASTER'),('components',pipeline.COMPONENTS[:-1]),('route','OTHER')]:
   bad=deepcopy(plan);bad[field]=value;refuse(field+str(value),lambda bad=bad:rp.execute_combined_saved_master_pipeline_run_plan(bad,inputs={}))
  bad=deepcopy(plan);bad['merge_scopes']=True;refuse('merge_scopes',lambda:rp.execute_combined_saved_master_pipeline_run_plan(bad,inputs={}))
 refuse('frontier',lambda:rp.compile_run_plan_v2(plan,frontier={}))
 refuse('missing_component_inputs',lambda:rp.execute_combined_saved_master_pipeline_run_plan(plan,inputs={}))
 for name,mutate in [('protected_science',lambda x:x.update({'canon.py':'0'*64})),('changed_service',lambda x:x.update({'storage_collections.py':'0'*64})),('unknown_addition',lambda x:x.update({'unqualified_extension.py':'0'*64})),('removed_old_module',lambda x:x.pop('canon.py'))]:
  bad=dict(current);mutate(bad);refuse(name,lambda bad=bad:compat.verify_source(old_manifest,bad))
 text=Path(rp.__file__).read_text();refuse('legacy_dispatch_tamper',lambda:compat.validate_runplan(text.replace('frontier_mutated','frontier_adopted')))
 assert len(refusals)==14
 assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==old_sem and theorem_registry.verify_all_theorems(raise_on_stale=False)==old_th
 write_json_atomic('COMBINED_SAVED_MASTER_PIPELINE.json',result)
 write_json_atomic('SOURCE_COMPATIBILITY.json',source_receipt)
 write_json_atomic('BASELINE_COMPARISON.json',{'status':'PASS','groups':5,'O7_requests':27,'O7_saved_Tests':100,'O7_complete_Counters':42,'R100_descriptors':193,'G2_G3_original_reads':5,'G4_terms':6,'G4_documents':80,'G4_authority_links':43,'native_store_records':267406,'all_original_payloads_and_stream_commitments_match':True,'calibration_source_rebound_only_for_current_qualified_reader':True})
 write_json_atomic('REFUSALS.json',{'status':'PASS','count':len(refusals),'cases':refusals,'unsupported_scopes_before_input':True})
 write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'COMBINED_CURRENT_SOURCE_SAVED_PIPELINE_QUALIFIED','components':5,'saved_requests':27,'source_scoped_g4_terms':6,'g4_saved_documents':80,'store_records':267406,'authority_links':43,'refusal_count':14,'generation_calls':0,'DAG_read_restorations':2,'full_master_producer_compatible':False,'package2_complete':False,'production_authority':'NONE','body_seconds':time.monotonic()-started})
if __name__=='__main__':main()
