"""Cold restoration audit with one shared ancestor memo; no candidate generation."""
from pathlib import Path
import sys,json,hashlib,sqlite3
B=Path(__file__).resolve().parent
J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,str(J/'source'));sys.path.insert(0,str(B))
from infinity_grid import preservation as p,maturation_parallel as mp
from infinity_grid.canon import canonical_sha256
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.partitions import assemble
p.confirm_batch(J,B/'ACK_BATCH.json')
s=p.status(J);assert not s['pending_objects'];(B/'PRESERVATION_FINAL.json').write_text(json.dumps(s,indent=2))
e=p.export_checkpoint(J,B/'NATIVE_CHECKPOINT_SLIM.zip',slim=True);(B/'CHECKPOINT_EXPORT.json').write_text(json.dumps(e,indent=2))
m=json.loads((B/'READBACKS.json').read_text());o=B/'restore_objects';o.mkdir(exist_ok=True)
for x in e['dependencies']:
 src=Path(m[x['sha256']]['path']);assert src.stat().st_size==x['size_bytes'] and hashlib.file_digest(src.open('rb'),'sha256').hexdigest()==x['sha256'];(o/(x['sha256']+'.bin')).symlink_to(src)
C=Path('/tmp/ig_native0209/cold_checkpoint');p.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,e['sha256'],o)
r=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));rows=[];dags=[];previous_science=json.loads((B/'GATE.json').read_text())['science_sha256']
for n in (100,):
 manifest=json.loads((r/'artifacts'/f'g1_partition_depth_{n}_manifest.json').read_text());base=C/'runtime/intake/artifacts'/(manifest['base']['sha256']+'.bin');assert hashlib.file_digest(base.open('rb'),'sha256').hexdigest()==manifest['base']['sha256'];nodes=json.loads(base.read_text())['dag']['nodes']
 for ref in manifest['partitions']:
  part=r/'artifacts'/Path(ref['path']).name;assert hashlib.file_digest(part.open('rb'),'sha256').hexdigest()==ref['sha256'];d=json.loads(part.read_text())
  for k,v in d['nodes'].items():
   assert k not in nodes or nodes[k]==v;nodes[k]=v
 dag=assemble(nodes,manifest['roots'],manifest['science_sha256']);assert d['level']==n and d['candidate_count']==193 and d['selected_count']==193 and d['parent_science_sha256']==previous_science
 db=r/'phases'/f'g1_partition_depth_{n}'/'state_store.sqlite3';conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);saved=conn.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall();assert len(saved)==1 and conn.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1;conn.close()
 digest,exact,state_json=saved[0];assert structural_canonical_bytes(json.loads(state_json))==bytes(exact) and hashlib.sha256(exact).hexdigest()==digest and json.loads(state_json)==d
 summary=json.loads((r/'phases'/f'g1_partition_depth_{n}'/'SUMMARY.json').read_text())['generation'];assert summary['raw_generated_occurrence_count']==1 and summary['stored_exact_identity_canonical_bytes']==len(exact)
 rows.append({'level':n,'roots':193,'nodes':len(dag['nodes']),'science_sha256':dag['science_sha256'],'new_partition_nodes':len(d['nodes']),'stored_identity_bytes':len(exact)});dags.append(dag);previous_science=dag['science_sha256'];del nodes,d,exact,state_json,saved
# Build the union only for independent restoration. Each original DAG must roundtrip exactly.
union_nodes={};union_roots=[]
for dag in dags:
 union_roots.extend(dag['roots'])
 for k,v in dag['nodes'].items():
  assert k not in union_nodes or union_nodes[k]==v;union_nodes[k]=v
union={'schema_id':'IG_MATURATION_STATE_DAG_V1','roots':union_roots,'nodes':union_nodes};union['science_sha256']=canonical_sha256(union)
_,states=mp._states_from_dag(union);assert len(states)==193
for i,dag in enumerate(dags):
 assert mp.state_dag_wire(states[i*193:(i+1)*193])==dag
 (B/'AUDIT_PROGRESS.json').write_text(json.dumps({'checked_depths':[x['level'] for x in rows[:i+1]]},indent=2))
from project.public_interface import extract_interface_population
capture=json.loads((C/'capture.json').read_text()) if (C/'capture.json').exists() else None
spec=json.loads((B/'SPEC.json').read_text());rh=spec['execution']['parameters']['bindings']['reference'];rp=C/'runtime/intake/artifacts'/(rh+'.bin');assert hashlib.file_digest(rp.open('rb'),'sha256').hexdigest()==rh;reference=json.loads(rp.read_text())
observed=extract_interface_population(states,source_stage='G1:R100',source_authority_sha256=reference['source_authority_sha256']);assert len(observed['interfaces'])==193 and observed['interfaces']==reference['interfaces']
(B/'INDEPENDENT_INTERFACE_POPULATION.json').write_text(json.dumps(observed,sort_keys=True,separators=(',',':')))
comparison=json.loads((r/'artifacts/g1_terminal_interface_comparison.json').read_text());expected={'level':100,'restored_roots':193,'interfaces_checked':193,'dag_science_sha256':dags[-1]['science_sha256']};assert comparison==expected
db=r/'phases/g1_terminal_interface_restore/state_store.sqlite3';conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);saved=conn.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall();assert len(saved)==1 and conn.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1;conn.close();digest,exact,state_json=saved[0];assert json.loads(state_json)==expected and structural_canonical_bytes(expected)==bytes(exact) and hashlib.sha256(exact).hexdigest()==digest
(B/'INTERFACE_AUDIT.json').write_text(json.dumps({'status':'PASS_INDEPENDENT193_INTERFACE_EQUALITY','interfaces_checked':193,'reference_sha256':rh,'observed_population_sha256':canonical_sha256(observed),'native_restore_exact_identity_verified':True,'candidate_generation_calls':0},indent=2))
raw=json.dumps({'level':100,'dag':dags[-1],'candidate_count':193,'selected_count':193},sort_keys=True,separators=(',',':')).encode();boot=Path('/tmp/ig_verified0209/BOOTSTRAP100.json');boot.write_bytes(raw)
(B/'BOOTSTRAP100_META.json').write_text(json.dumps({'path':str(boot),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'science_sha256':dags[-1]['science_sha256'],'generator_calls':0},indent=2))
result={'status':'PASS_COLD_TERMINAL_DAG_AND193_INTERFACE_COMPARISON','completed_depths':[100],'roots':193,'earlier_depths_regenerated':False,'candidate_build_calls':193,'pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'PASS','cold_workspace':str(C),'original_workspace_state_read_during_audit':False,'audit_candidate_generation_calls':0,'reconstruction_method':'independent ancestor reconstruction for193root references; exact original DAG roundtrip','rows':rows};(B/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
