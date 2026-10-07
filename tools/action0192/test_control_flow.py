"""Exercise real TaskSpec and every handler branch using a non-scientific runtime double."""
import sys,json,tempfile,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine');sys.path.insert(0,str(B))
from project.handler import handler
s=json.loads((B/'SPEC.json').read_text());stage={'execution':{'parameters':s['execution']['parameters']},'input_artifacts':{x['logical_name']:x['path'] for x in s['inputs']}}
class Runtime:
 def __init__(self,d):self.root=Path(d);self.phases={};self.reused=0
 def run_content_indexed_generation(self,**kw):
  task=list(kw['tasks'])[0];assert task.task_kind and kw['evaluator_ref']=='project.worker:evaluate'
  if kw['phase_id'] in self.phases:self.reused+=1;return
  p=task.payload;n=193 if p['level']==100 else 24
  state={'level':p['level'],'dag':{'roots':list(range(n))},'selected_count':n}
  if p['mode']=='RESTORE':state={'interfaces_checked':193}
  self.phases[kw['phase_id']]=state
 def iter_generated_states(self,**kw):return iter([{'state':self.phases[kw['phase_id']]}])
 def publish_json(self,name,obj):
  p=self.root/(name+'.json');p.write_text(json.dumps(obj));return {'path':str(p)}
with tempfile.TemporaryDirectory() as d:
 r=Runtime(d);a=handler(stage,r).result;b=handler(stage,r).result;assert a==b and r.reused==95 and len(r.phases)==95
out={'status':'PASS_HANDLER_CONTROL_FLOW','native_TaskSpec_constructor':'REAL','phases_exercised':95,'repeat_reused_phases':95,'scientific_evaluator_calls':0,'limit':'Runtime double only; scientific/bootstrap execution still required'};(B/'CONTROL_FLOW_TEST.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
