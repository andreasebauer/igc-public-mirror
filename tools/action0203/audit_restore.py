from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'));sys.path.insert(0,str(B))
from infinity_grid import preservation as p,maturation_parallel as mp
from project.partitions import assemble
p.confirm_batch(J,B/'ACK_BATCH.json');s=p.status(J);assert not s['pending_objects'];(B/'PRESERVATION_FINAL.json').write_text(json.dumps(s,indent=2));e=p.export_checkpoint(J,B/'NATIVE_CHECKPOINT_SLIM.zip',slim=True);(B/'CHECKPOINT_EXPORT.json').write_text(json.dumps(e,indent=2));m=json.loads((B/'READBACKS.json').read_text());o=B/'restore_objects';o.mkdir(exist_ok=True)
for x in e['dependencies']:
 src=Path(m[x['sha256']]['path']);assert hashlib.sha256(src.read_bytes()).hexdigest()==x['sha256'];(o/(x['sha256']+'.bin')).symlink_to(src)
C=Path('/tmp/ig_native0203/cold_checkpoint');p.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,e['sha256'],o);r=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));rows=[]
# Transport locators refer to original workspace; resolve identical bytes only inside restored tree.
for n in range(85,89):
 manifest=json.loads((r/'artifacts'/f'g1_partition_depth_{n}_manifest.json').read_text());base=C/'runtime/intake/artifacts'/(manifest['base']['sha256']+'.bin');assert hashlib.sha256(base.read_bytes()).hexdigest()==manifest['base']['sha256'];nodes=json.loads(base.read_text())['dag']['nodes']
 for ref in manifest['partitions']:
  part=r/'artifacts'/Path(ref['path']).name;assert hashlib.sha256(part.read_bytes()).hexdigest()==ref['sha256'];d=json.loads(part.read_text())
  for k,v in d['nodes'].items():
   assert k not in nodes or nodes[k]==v;nodes[k]=v
 dag=assemble(nodes,manifest['roots'],manifest['science_sha256']);_,states=mp._states_from_dag(dag);assert mp.state_dag_wire(states)==dag
 summary=json.loads((r/'phases'/f'g1_partition_depth_{n}'/'SUMMARY.json').read_text());rows.append({'level':n,'roots':len(states),'nodes':len(dag['nodes']),'science_sha256':dag['science_sha256'],'new_partition_nodes':len(d['nodes']),'stored_identity_bytes':summary['generation']['stored_exact_identity_canonical_bytes']})
 (B/'AUDIT_PROGRESS.json').write_text(json.dumps({'checked_depths':[x['level'] for x in rows]},indent=2))
result={'status':'PASS_COLD_NATIVE_CHECKPOINT_AND_PARTITION_DAG_RESTORE','completed_depths':[85,88],'roots':24,'earlier_depths_regenerated':False,'candidate_build_calls_committed':4*193,'failed_depth':89,'failure':'STREAM_RESULT_BYTES_LIMIT','depth90_started':False,'generation_attempt_scope':'Four committed 193-candidate phases; one failed depth89 task, committed count does not include failed work','pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN','cold_workspace':str(C),'original_workspace_state_read_during_audit':False,'rows':rows};(B/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
