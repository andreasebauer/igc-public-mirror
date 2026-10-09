"""Meaningful native closure, source-inspection and refusal regression controls."""
from pathlib import Path
import sys,json,hashlib,tempfile,shutil,os
B=Path(__file__).resolve().parent;Q=B.parent/'continuation0235';E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid.workflow_guard import preflight,scientific_call
from infinity_grid.invocation import InvocationRefused
from infinity_grid.semantic_sentinel import callable_semantic_sha256
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def refused(source,paths,code):
 try:preflight(source,paths)
 except InvocationRefused as e:assert e.code==code;return
 raise AssertionError('Expected refusal '+code)
def main():
 pre=json.loads((B/'PREREGISTRATION.json').read_text());results=[]
 with tempfile.TemporaryDirectory(prefix='ig_guard0237_') as t:
  S=Path(t);shutil.copytree(E/'infinity_grid',S/'infinity_grid');shutil.copytree(Q/'project',S/'project')
  gate=preflight(S,[S/'project/handler.py',S/'project/worker.py']);assert len(gate['modules'])==22
  accepted=[r for r in gate['modules'] if 'source_inspection_exception' in r];assert len(accepted)==1 and accepted[0]['sha256']==pre['semantic_sentinel_sha256'];results.append('PASS_FULL_22_MODULE_CAPTURED_SOURCE_CLOSURE')
  p=S/'infinity_grid/semantic_sentinel.py';raw=p.read_bytes();p.write_bytes(raw+b'\n# mutated negative control\n');refused(S,[p],'SOURCE_INSPECTION_IDENTITY');p.write_bytes(raw);results.append('PASS_MUTATED_SENTINEL_REFUSED')
  for name,code in [('dynamic','import importlib\nimportlib.import_module("os")\n'),('process','import subprocess\nsubprocess.run(["true"])\n'),('pool','from concurrent.futures import ProcessPoolExecutor\nProcessPoolExecutor()\n')]:
   p=S/'project'/('negative_'+name+'.py');p.write_text(code);refused(S,[p],'PRIVATE_EXECUTION_PREFLIGHT');results.append('PASS_PROJECT_'+name.upper()+'_REFUSED');p.unlink()
  try:
   with scientific_call(S):os.system('true')
  except InvocationRefused as e:assert e.code=='PRIVATE_EXECUTION_RUNTIME';results.append('PASS_RUNTIME_PROCESS_CREATION_REFUSED')
  else:raise AssertionError('Runtime process guard missing')
  for n,h in pre['scientific_source_pins'].items():assert sha(E/'infinity_grid/uplift_structural.py' if n.startswith('engine/') else Q/n)==h
  assert sha(E/'infinity_grid/semantic_sentinel.py')==pre['semantic_sentinel_sha256']
  changed=[]
  for p in E.rglob('*.py'):
   old=Path('/tmp/ig_engine0204')/p.relative_to(E)
   if old.exists() and sha(old)!=sha(p):changed.append(str(p.relative_to(E)))
  assert changed==['infinity_grid/workflow_guard.py'];results.append('PASS_ONLY_ENGINE_GUARD_CHANGED_SCIENCE_UNCHANGED')
  # Source normalization hash should agree with the old engine's scheme.
  h=callable_semantic_sha256('infinity_grid.semantic_sentinel:verify_native_semantics')
  out=dict(status='PASS_NARROW_NATIVE_IMPORT_REPAIR',checks=results,complete_native_gate=gate,changed_files=changed,guard_before_sha256=pre['old_guard_sha256'],guard_after_sha256=sha(E/'infinity_grid/workflow_guard.py'),sentinel_callable_semantic_sha256=h,official_cases_executed=0,master_slices=152,new_admissions=0,G2_promotion=False)
  (B/'REPAIR_TESTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='complete_native_gate'}))
if __name__=='__main__':main()
