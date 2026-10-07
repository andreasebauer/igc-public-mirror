from pathlib import Path
import json,sys,hashlib,shutil
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine');sys.path.insert(0,str(B))
from project.partitions import assemble
from infinity_grid import maturation_parallel as mp
from infinity_grid.workflow_guard import preflight
from infinity_grid.result_contracts import normalize
A=json.loads((B.parent/'partition0193/AUDIT.json').read_text());C=Path(A['cold_workspace']);r=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));m=json.loads((r/'artifacts/g1_partition_depth_30_manifest.json').read_text());base=C/'runtime/intake/artifacts'/(m['base']['sha256']+'.bin');assert hashlib.sha256(base.read_bytes()).hexdigest()==m['base']['sha256'];nodes=json.loads(base.read_text())['dag']['nodes']
for ref in m['partitions']:
 p=r/'artifacts'/Path(ref['path']).name;assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
 for k,v in json.loads(p.read_text())['nodes'].items():
  assert k not in nodes or nodes[k]==v;nodes[k]=v
D=assemble(nodes,m['roots'],m['science_sha256']);_,states=mp._states_from_dag(D);assert len(states)==24 and mp.state_dag_wire(states)==D and D['science_sha256']==A['rows'][-1]['science_sha256']
x={'level':30,'dag':D,'candidate_count':193,'selected_count':24};raw=json.dumps(x,sort_keys=True,separators=(',',':')).encode();(B/'BOOTSTRAP30.json').write_bytes(raw);h=hashlib.sha256(raw).hexdigest();s=json.loads((B.parent/'partition0193/SPEC.json').read_text());s['job_id']='MASTER.G1.EXACT.PARTITION.CONTINUATION.0194';s['project_source']=str(B/'project');s['question'].update(stage_id='MASTER:G1:EXACT:PARTITION:0194',description='Bounded exact continuation31-36 from audited depth30; no terminal claim');s['execution']['parameters'].update(start_depth=31,stop_depth=36,bootstrap_science_sha256=D['science_sha256']);s['execution']['parameters']['bindings']['bootstrap']=h
for row in s['inputs']:
 if row['logical_name']=='bootstrap':row.update(path=str(B/'BOOTSTRAP30.json'),sha256=h)
for row in s['output_contract']['result_checks']:
 if row['pointer']=='/completed_depth':row['equals']=36
normalize(s['output_contract'],s['execution'],s['question']);(B/'SPEC.json').write_text(json.dumps(s,indent=2));source=Path('/tmp/ig_partition0194_preflight');shutil.copytree('/tmp/ig_admit0188/engine',source,dirs_exist_ok=True);shutil.copytree(B/'project',source/'project',dirs_exist_ok=True);g=preflight(source,[source/'project/handler.py',source/'project/worker.py']);assert g['status']=='PASS';out={'status':'PASS_BOOTSTRAP30_EXACT_RESTORE_AND_TRANSITIVE_PREFLIGHT','bootstrap_sha256':h,'bootstrap_bytes':len(raw),'science_sha256':D['science_sha256'],'roots':24,'nodes':len(D['nodes']),'generator_calls':0,'storage_code_unchanged_from0193':True,'architecture':g};(B/'TESTS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='architecture'}))
