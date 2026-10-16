"""Native qualification of exact original saved G4 terms; no scientific regeneration."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import os,json,hashlib,zipfile
from infinity_grid import historical_g4_saved_terms as reader
from infinity_grid.run_plan_v2 import compile_run_plan_v2,execute_historical_g4_saved_term_run_plan,RunPlanV2Error
from infinity_grid import semantic_sentinel,theorem_registry
from infinity_grid.g4_term_state import G3TermState
from infinity_grid.canon import write_json_atomic

def main():
 inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text());paths={k:inputs[k] for k in reader.PINS}
 plan={'schema_id':reader.SCHEMA,'route':reader.ROUTE,'cohorts':['original','rebase'],'scope':'SAVED_CAPS7_ATOMIC_TERMS_ONLY','generation_allowed':False,'production_authority':'NONE'}
 old_sem=semantic_sentinel.verify_native_semantics(raise_on_change=False);old_th=theorem_registry.verify_all_theorems(raise_on_stale=False)
 assert old_sem['status']=='REOPEN_REQUIRED' and len(old_th['stale_theorems'])==4
 # Independent raw JSON comparison: no decoder carrier or corpus constructor.
 expected={}
 for cohort,path in paths.items():
  with zipfile.ZipFile(path) as z:
   candidates=[n for n in z.namelist() if '/primary/' in n and n.endswith('.json')]
   results=[json.loads(z.read(n)) for n in candidates]
   v=[v for v in results if 'certified_term_corpus' in v];assert len(v)==1
   expected[cohort]=v[0]['certified_term_corpus']['terms']
 with patch.object(G3TermState,'from_homogeneous_seed',side_effect=AssertionError('CORPUS_REBUILD')),patch.object(G3TermState,'reserve_external_relation',side_effect=AssertionError('RELATION_GENERATION')):
  result=execute_historical_g4_saved_term_run_plan(plan,archives=paths)
 assert result['compiled']==compile_run_plan_v2(plan)
 streams={};count=0
 for c in result['cohorts']:
  observed={r['key']:r['original_record'] for r in c['terms']};assert observed==expected[c['cohort']]
  for row in c['terms']:
   assert row['strict_read']['term']==row['original_record']['term'];count+=1
  streams[c['cohort']]=hashlib.sha256(json.dumps(observed,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 assert count==6 and result['cohorts_merged'] is False
 refusals=[]
 def refuse(name,fn):
  try:fn()
  except (reader.HistoricalG4Error,RunPlanV2Error):refusals.append(name)
  else:raise AssertionError('NO_REFUSAL:'+name)
 with patch.object(reader,'_snapshot',side_effect=AssertionError('READ_BEFORE_REFUSAL')):
  for field,value in [('scope','FULL_MASTER'),('scope','REBUILD_FROM_SUMMARY'),('generation_allowed',True),('generation_allowed',0),('production_authority','MASTER'),('cohorts',['rebase']),('route','OTHER')]:
   bad=deepcopy(plan);bad[field]=value;refuse(field+str(value),lambda bad=bad:execute_historical_g4_saved_term_run_plan(bad,archives={}))
  bad=deepcopy(plan);bad['merge_cohorts']=True;refuse('cohort_merge_override',lambda:execute_historical_g4_saved_term_run_plan(bad,archives={}))
 refuse('frontier',lambda:compile_run_plan_v2(plan,frontier={}))
 refuse('missing_input',lambda:execute_historical_g4_saved_term_run_plan(plan,archives={}))
 badpaths=dict(paths);badpaths['original']=paths['rebase'];refuse('wrong_archive',lambda:execute_historical_g4_saved_term_run_plan(plan,archives=badpaths))
 assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==old_sem and theorem_registry.verify_all_theorems(raise_on_stale=False)==old_th
 write_json_atomic('HISTORICAL_G4_SAVED_TERMS.json',result)
 write_json_atomic('INDEPENDENT_RECORD_COMPARISON.json',{'status':'PASS','source_scoped_terms':count,'complete_original_record_equality':True,'streams':streams})
 write_json_atomic('REFUSALS.json',{'status':'PASS','count':len(refusals),'cases':refusals,'unsupported_scopes_refused_before_input':True})
 write_json_atomic('QUALIFICATION_RESULT.json',{'status':'PASS','scientific_outcome':'HISTORICAL_G4_SAVED_TERM_READ_QUALIFIED','installed_runplan_dispatch':True,'source_scoped_terms':6,'original_terms':2,'rebase_terms':4,'refusal_count':11,'generation_calls':0,'corpus_rebuilds':0,'cohorts_merged':False,'full_master_producer_compatible':False,'production_authority':'NONE','package2_complete':False})
if __name__=='__main__':main()
