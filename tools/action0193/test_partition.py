from pathlib import Path
import sys,json,tempfile,hashlib,shutil
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_admit0188/engine');sys.path.insert(0,str(B))
from project.partitions import delta,load_parent,assemble
from infinity_grid import maturation_parallel as mp
from infinity_grid.workflow_guard import preflight
from infinity_grid.result_contracts import normalize
cold=Path(json.loads((B.parent/'execution0192/AUDIT.json').read_text())['cold_workspace']);r=next(cold.glob('runtime/runs/*/chain/decoder_stage_runtime/*/artifacts'));old=json.loads((r/'g1_exact_depth_23.json').read_text());cur=json.loads((B/'BOOTSTRAP24.json').read_text());d=delta(old,cur)
with tempfile.TemporaryDirectory() as t:
 t=Path(t)
 def save(name,x):
  p=t/name;p.write_text(json.dumps(x));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 base=save('base.json',old);part=save('part.json',d);m={'schema_id':'IG_G1_PARTITION_MANIFEST_V1','base':base,'partitions':[part],'level':24,'roots':d['roots'],'science_sha256':d['science_sha256'],'selected_count':24,'candidate_count':193};manifest=save('manifest.json',m);got=load_parent(manifest['path'],manifest['sha256']);assert got==cur;_,states=mp._states_from_dag(got['dag']);assert mp.state_dag_wire(states)==cur['dag']
 def reject(fn):
  try:fn()
  except (ValueError,KeyError):return
  raise AssertionError('corruption accepted')
 missing=dict(cur['dag']['nodes']);missing.pop(cur['dag']['roots'][0]);reject(lambda:assemble(missing,d['roots'],d['science_sha256']))
 conflict=json.loads(json.dumps(cur));k=next(k for k in old['dag']['nodes'] if k in conflict['dag']['nodes']);conflict['dag']['nodes'][k]['lane']='CORRUPTED';reject(lambda:delta(old,conflict))
 Path(part['path']).write_text('{}');reject(lambda:load_parent(manifest['path'],manifest['sha256']))
source=Path('/tmp/ig_partition0193_preflight');shutil.copytree('/tmp/ig_admit0188/engine',source,dirs_exist_ok=True);shutil.copytree(B/'project',source/'project',dirs_exist_ok=True);g=preflight(source,[source/'project/handler.py',source/'project/worker.py']);assert g['status']=='PASS'
s=json.loads((B/'SPEC.json').read_text());normalize(s['output_contract'],s['execution'],s['question'])
from infinity_grid.structural_encoding import structural_canonical_bytes
from infinity_grid.canon import canonical_text
size=len(structural_canonical_bytes(d))+len(canonical_text(d).encode());assert size<8*1024*1024
out={'status':'PASS_EXACT_DELTA_ROUNDTRIP_AND_TRANSITIVE_PREFLIGHT','reused_old_nodes':len(cur['dag']['nodes'])-len(d['nodes']),'new_nodes':len(d['nodes']),'encoded_delta_bytes':size,'missing_node_rejected':True,'full_node_collision_rejected':True,'modified_partition_rejected':True,'independent_state_restore':'PASS','generator_calls':0,'architecture':g};(B/'TESTS.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='architecture'}))
