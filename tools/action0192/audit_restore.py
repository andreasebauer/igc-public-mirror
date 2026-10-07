from pathlib import Path
import json,sys,hashlib
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p,maturation_parallel as mp
from infinity_grid.canon import canonical_sha256
p.confirm_batch(J,B/'ACK_BATCH.json');s=p.status(J);assert not s['pending_objects'];(B/'PRESERVATION_FINAL.json').write_text(json.dumps(s,indent=2))
e=p.export_checkpoint(J,B/'NATIVE_CHECKPOINT_SLIM.zip',slim=True);(B/'CHECKPOINT_EXPORT.json').write_text(json.dumps(e,indent=2))
# Stage only verified dependencies from durable readbacks.
m=json.loads((B/'CHECKPOINT_READBACKS.json').read_text());o=B/'restore_objects';o.mkdir(exist_ok=True)
for x in e['dependencies']:
 r=m[x['sha256']];src=Path(r['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==x['sha256'];dest=o/(x['sha256']+'.bin');dest.symlink_to(src)
C=Path('/tmp/ig_native0192/cold_checkpoint');p.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,e['sha256'],o)
r=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));rows=[]
for n in range(7,25):
 a=r/'artifacts'/f'g1_exact_depth_{n}.json';x=json.loads(a.read_text());_,states=mp._states_from_dag(x['dag']);assert len(states)==24;roundtrip=mp.state_dag_wire(states);assert roundtrip==x['dag']
 rows.append({'level':n,'root_count':len(states),'node_count':len(x['dag']['nodes']),'science_sha256':x['dag']['science_sha256'],'snapshot_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'canonical_data_bytes':len(json.dumps(x,sort_keys=True,separators=(',',':')).encode()),'phase_status':json.loads((r/'phases'/f'g1_exact_depth_{n}'/'RUNTIME_STATUS.json').read_text())})
out={'status':'PASS_COLD_CHECKPOINT_AND_EXACT_DAG_ROUNDTRIP','capture_id':e['dependencies'][0].get('capture_id',json.loads((B/'POINTER.json').read_text()).get('capture_id')),'cold_workspace':str(C),'completed_depths':[7,24],'committed_phases':18,'level_generation_attempts':18,'candidate_build_calls':18*193,'terminal_comparison':'NOT_RUN','master_slices':151,'new_admissions':0,'stop':'DEPTH25_GENERATION_TASK_BYTES_LIMIT','rows':rows};(B/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
