from pathlib import Path
import subprocess,json,shutil,sys
h=Path(__file__).resolve().parent.parent
runtime='/tmp/ig_decoder09_v5_runtime/python-bundled-host'
modules=['test_decoder09_implementation','test_bulk_mapping_store','test_checkpoint_project_closure','test_completion_publication','test_terminal_json_equivalence','test_resource_limit_policy','test_v05_memory_housekeeping','test_preservation_batch','test_decoder06_results_preservation','test_paused_checkpoint_recovery','test_v05_controller_namespace_portability','test_v05_stage_runtime_exact_byte_handoff','test_execution_start_method_portability','test_08lib_backend']
rows=[]
(h/'verification_tmp').mkdir(exist_ok=True)
for module in modules:
 temp=h/'verification_tmp'/module
 cmd=[runtime,str(h/'scripts/pytest_run.py'),'-q',str(h/'source/tests'/f'{module}.py'),f'--basetemp={temp}',f'--junitxml={h/"results"/(module+".xml")}']
 run=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 (h/'results'/(module+'.txt')).write_text(run.stdout)
 rows.append({'module':module,'returncode':run.returncode});print(module,run.returncode,flush=True)
 # Only this orchestrator's disposable fixture directory, after evidence saved.
 if temp.is_dir() and not temp.is_symlink() and temp.parent.resolve()==(h/'verification_tmp').resolve():shutil.rmtree(temp)
 if run.returncode:break
(h/'results/LOCAL_SUITES.json').write_text(json.dumps(rows,indent=2)+'\n')
raise SystemExit(max(x['returncode'] for x in rows))
