"""Native diagnostic: persist observed population before exact comparison report."""
import hashlib,gzip
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound

def differences(a,b,path='',counts=None,examples=None):
 if counts is None:counts={};examples=[]
 if type(a)!=type(b):kind='TYPE'
 elif isinstance(a,dict):
  for k in sorted(set(a)|set(b)):
   p=path+'/'+str(k)
   if k not in a or k not in b:
    counts['MISSING_KEY']=counts.get('MISSING_KEY',0)+1
    if len(examples)<100:examples.append(dict(path=p,kind='MISSING_KEY',observed=a.get(k),reference=b.get(k)))
   else:differences(a[k],b[k],p,counts,examples)
  return counts,examples
 elif isinstance(a,list):
  if len(a)!=len(b):
   counts['LIST_LENGTH']=counts.get('LIST_LENGTH',0)+1
   if len(examples)<100:examples.append(dict(path=path,kind='LIST_LENGTH',observed=len(a),reference=len(b)))
  for i,(x,y) in enumerate(zip(a,b)):differences(x,y,path+'/'+str(i),counts,examples)
  return counts,examples
 elif a==b:return counts,examples
 else:kind='VALUE'
 counts[kind]=counts.get(kind,0)+1
 if len(examples)<100:examples.append(dict(path=path,kind=kind,observed=a,reference=b))
 return counts,examples

def handler(stage,runtime):
 i=stage['input_artifacts'];c=stage['execution']['parameters'];h=c['bindings']
 for k in h:read_bound(i[k],h[k])
 if hashlib.sha256(gzip.decompress(Path(i['bootstrap']).read_bytes())).hexdigest()!=c['bootstrap_raw_sha256']:raise ValueError('BOOTSTRAP_RAW_IDENTITY')
 common={k+'_path':i[k] for k in ('pairs','recipes','reference','anchor')};common.update({k+'_sha256':h[k] for k in ('pairs','recipes','reference','anchor')})
 payload=dict(common,mode='LEVEL',level=100,parent_path=i['bootstrap'],parent_sha256=h['bootstrap']);task=TaskSpec(task_id='terminal100_diagnostic',task_kind='HISTORICAL_G1_QUALIFICATION',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id='terminal100_diagnostic',tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=c['result_transport_budget_bytes'],max_work_seconds=c['task_work_budget_seconds'])
 rows=list(runtime.iter_generated_states(phase_id='terminal100_diagnostic'));assert len(rows)==1;data=rows[0]['state'];assert data['candidate_count']==193 and data['selected_count']==193
 observed=data['terminal_interface_population'];p=runtime.publish_json('terminal100_observed_population',observed)
 counts,examples=differences(observed,read_bound(i['reference'],h['reference']))
 reference=read_bound(i['reference'],h['reference']);row_equal=observed['interfaces']==reference['interfaces']
 report=dict(diagnostic_only=True,terminal_comparison=data['terminal_comparison'],historical_depth100_beam_anchor=data['historical_depth100_beam_anchor'],difference_counts=counts,examples=examples,interface_rows_exact_equal=row_equal,top_level_differing_keys=[k for k in sorted(set(observed)|set(reference)) if observed.get(k)!=reference.get(k)],observed_population=dict(path=p['path'],sha256=hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()),accepted_historical_depth=98,new_admissions=0)
 rp=runtime.publish_json('terminal100_diagnostic_report',report)
 runtime.publish_json('terminal100_diagnostic_state',data)
 return ChainExecutionResult(result=dict(outcome='TERMINAL_DIAGNOSTIC_RECORDED',diagnostic_only=True,observed_population_persisted=True,candidate_build_calls=193,master_scientific_slices=151,new_admissions=0,terminal_comparison=data['terminal_comparison'],diagnostic_report=rp))
