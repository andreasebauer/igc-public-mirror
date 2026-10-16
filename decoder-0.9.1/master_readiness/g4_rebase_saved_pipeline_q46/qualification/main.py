"""Qualify all saved G4 rebase records and reconciled authority links natively."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import json,os,zipfile,hashlib
from infinity_grid import g4_rebase_saved_pipeline as reader
from infinity_grid.run_plan_v2 import compile_run_plan_v2,execute_g4_rebase_saved_pipeline_run_plan,RunPlanV2Error
from infinity_grid import theorem_registry,semantic_sentinel
from infinity_grid.g4_term_state import G3TermState
from infinity_grid.canon import write_json_atomic

def main():
 inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text());pins=json.loads(reader.MANIFEST.read_text());paths={k:inputs[k] for k in pins}
 plan={'schema_id':reader.SCHEMA,'route':reader.ROUTE,'scope':'ORIGINAL_SAVED_RECORDS_AND_AUTHORITY_LINKS','r6_authority':'DRIVE_BRANCH','generation_allowed':False,'production_authority':'NONE'}
 sem=semantic_sentinel.verify_native_semantics(raise_on_change=False);th=theorem_registry.verify_all_theorems(raise_on_stale=False)
 assert sem['status']=='REOPEN_REQUIRED' and len(th['stale_theorems'])==4
 expected={}
 # Independent full record comparator uses raw ZIP/JSON, no stage finalizers.
 for name,path in paths.items():
  with zipfile.ZipFile(path) as z:expected[name]={n:json.loads(z.read(n)) for n in pins[name]['members']}
 with patch.object(G3TermState,'from_homogeneous_seed',side_effect=AssertionError('REBUILD')),patch.object(G3TermState,'reserve_external_relation',side_effect=AssertionError('GENERATE')):
  result=execute_g4_rebase_saved_pipeline_run_plan(plan,archives=paths)
 assert result['compiled']==compile_run_plan_v2(plan) and result['saved_records']==80 and result['rebase_stages']==15
 streams={}
 for name,stage in result['stages'].items():
  assert stage['records']==expected[name]
  streams[name]=hashlib.sha256(json.dumps(expected[name],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (reader.G4SavedPipelineError,RunPlanV2Error):refusals.append(name)
  else:raise AssertionError('NO_REFUSAL:'+name)
 with patch.object(reader,'_snapshot',side_effect=AssertionError('READ_BEFORE_SCOPE_REFUSAL')):
  for field,value in [('scope','FULL_MASTER'),('scope','GENERATE_MISSING_OUTPUTS'),('r6_authority','PARALLEL_LOCAL_BRANCH'),('generation_allowed',True),('generation_allowed',0),('production_authority','MASTER'),('route','OTHER')]:
   bad=deepcopy(plan);bad[field]=value;refuse(field+str(value),lambda bad=bad:execute_g4_rebase_saved_pipeline_run_plan(bad,archives={}))
  bad=deepcopy(plan);bad['merge_branches']=True;refuse('merge_override',lambda:execute_g4_rebase_saved_pipeline_run_plan(bad,archives={}))
 refuse('frontier',lambda:compile_run_plan_v2(plan,frontier={}))
 refuse('missing_input',lambda:execute_g4_rebase_saved_pipeline_run_plan(plan,archives={}))
 badpaths=dict(paths);badpaths['R6']=paths['R6_PARALLEL'];refuse('same_named_wrong_r6_archive',lambda:execute_g4_rebase_saved_pipeline_run_plan(plan,archives=badpaths))
 bad=deepcopy(result['stages']);bad['R6']=bad['R6_PARALLEL'];refuse('wrong_r6_record_branch',lambda:reader._branch(bad))
 bad=deepcopy(result['stages']);bad['S2']['main']['authority']['s1_science_sha256']='0'*64;refuse('broken_chain_link',lambda:reader._links(bad))
 bad=deepcopy(result['stages']);bad['S0']['main']['certified_term_corpus']['terms']['C']['term']['term_ref']='0'*64;refuse('wrong_saved_C_reference',lambda:reader._branch(bad))
 assert len(refusals)==14
 assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==sem and theorem_registry.verify_all_theorems(raise_on_stale=False)==th
 write_json_atomic('G4_REBASE_SAVED_PIPELINE.json',result)
 write_json_atomic('INDEPENDENT_RECORD_COMPARISON.json',{'status':'PASS','saved_records':80,'complete_record_equality':True,'streams':streams})
 write_json_atomic('REFUSALS.json',{'status':'PASS','count':len(refusals),'cases':refusals,'unsupported_scope_before_input':True})
 write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'G4_REBASE_SAVED_PIPELINE_QUALIFIED','installed_runplan_dispatch':True,'saved_records':80,'rebase_stages':15,'parallel_supporting_branches':1,'authority_links':len(result['authority_links']),'R6_authority':'DRIVE_BRANCH','saved_C_D_terms_bound':True,'refusal_count':14,'generation_calls':0,'complete_g4_carrier_population':False,'full_master_producer_compatible':False,'production_authority':'NONE','package2_complete':False})
if __name__=='__main__':main()
