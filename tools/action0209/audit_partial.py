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
# Frozen native checkpoint refused the oversized published artifact. Restore only
# byte-verified forensic runtime archive; this is not a native terminal completion.
meta=json.loads((B/'FORENSIC_READBACK.json').read_text());archive=Path(meta['raw_readback_path']);assert meta['raw_readback_verified'] and hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()==meta['sha256'] and archive.stat().st_size==meta['bytes']
import zipfile
C=Path('/tmp/ig_native0209/forensic_cold');assert not C.exists();C.mkdir()
with zipfile.ZipFile(archive) as z:
 manifest=json.loads(z.read('FORENSIC_MANIFEST.json'))
 for name,record in manifest.items():
  target=C/name;assert target.resolve().is_relative_to(C.resolve());target.parent.mkdir(parents=True,exist_ok=True)
  with z.open(name) as src,target.open('wb') as dest:
   h=hashlib.sha256();n=0
   while chunk:=src.read(1048576):dest.write(chunk);h.update(chunk);n+=len(chunk)
  assert h.hexdigest()==record['sha256'] and n==record['bytes']
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
assert not (r/'artifacts/g1_terminal_interface_comparison.json').exists()
db=r/'phases/g1_terminal_interface_restore/state_store.sqlite3';conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True);restore_states=conn.execute('SELECT COUNT(*) FROM states').fetchone()[0];restore_tasks=conn.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0];conn.close();assert restore_states==0 and restore_tasks==0
(B/'INTERFACE_AUDIT.json').write_text(json.dumps({'status':'PASS_INDEPENDENT193_INTERFACE_EQUALITY','interfaces_checked':193,'reference_sha256':rh,'observed_population_sha256':canonical_sha256(observed),'native_restore_committed':False,'native_restore_states':restore_states,'native_restore_tasks':restore_tasks,'candidate_generation_calls':0},indent=2))
raw=json.dumps({'level':100,'dag':dags[-1],'candidate_count':193,'selected_count':193},sort_keys=True,separators=(',',':')).encode();boot=Path('/tmp/ig_verified0209/BOOTSTRAP100.json');boot.write_bytes(raw)
(B/'BOOTSTRAP100_META.json').write_text(json.dumps({'path':str(boot),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'science_sha256':dags[-1]['science_sha256'],'generator_calls':0},indent=2))
result={'status':'PASS_FORENSIC_COLD_TERMINAL100_DAG_AND193_INTERFACE_COMPARISON','native_registered_scope_completed':False,'native_interface_restore_committed':False,'stop_reason':'CHECKPOINT_STATE_RAW_LIMIT:1190082971','preservation_method':'Raw byte-verified forensic archive; frozen native checkpoint refusal retained','completed_depths':[100],'roots':193,'earlier_depths_regenerated':False,'candidate_build_calls':193,'native_pending_bytes':json.loads((B/'PRESERVATION_FINAL.json').read_text())['pending_bytes'],'master_slices':151,'new_admissions':0,'terminal_comparison':'PASS','cold_workspace':str(C),'original_workspace_state_read_during_audit':False,'audit_candidate_generation_calls':0,'reconstruction_method':'independent ancestor reconstruction for193root references; exact original DAG roundtrip','rows':rows};(B/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
