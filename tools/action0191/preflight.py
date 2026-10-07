import sys,json,hashlib,tempfile
from pathlib import Path
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine');sys.path.insert(0,str(B))
from infinity_grid.workflow_guard import preflight
from infinity_grid.result_contracts import normalize
from infinity_grid.project_stage import bind_callables
from infinity_grid.canon import canonical_sha256
s=json.loads((B/'SPEC.json').read_text());checks=[]
for x in s['inputs']+s['environment']['artifacts']:
 assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256'];checks.append(x['logical_name'])
gate=preflight(B,[B/'project/handler.py',B/'project/worker.py']);assert gate['status']=='PASS'
normalize(s['output_contract'],s['execution'],s['question'])
from project.worker import read_bound
with tempfile.TemporaryDirectory() as t:
 p=Path(t)/'input.json';p.write_text('{"x":1}');h=hashlib.sha256(p.read_bytes()).hexdigest();assert read_bound(p,h)=={'x':1};p.write_text('{"x":2}')
 try:read_bound(p,h)
 except ValueError as e:assert str(e)=='INPUT_IDENTITY'
 else:raise AssertionError('modified parent accepted')
result={'status':'PASS_STATIC_NATIVE_PREFLIGHT','architecture':gate,'input_bindings_checked':checks,'modified_parent_rejected':True,'result_contract_normalized':True,'python':sys.version.split()[0],'handler_called':False,'evaluator_called':False,'scientific_generator_calls':0,'resource_limits':{'native_task_payload_bytes':4194304,'encoded_result_bytes':8388608,'max_phase_tasks':1,'workers':1},'remaining_resource_gate':'No real DAG growth or bootstrap resource pilot has run; limits fail closed, never auto-raised'}
(B/'PREFLIGHT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='architecture'}))
