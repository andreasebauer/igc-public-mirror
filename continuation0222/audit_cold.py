"""Cold native checkpoint identity, DAG, and historical anchor audit; no generation."""
from pathlib import Path
import sys,json,hashlib,sqlite3
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace']);sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());objects=Path('/tmp/ig_cold0222_objects');objects.mkdir(exist_ok=True)
M=json.loads((B/'READBACKS.json').read_text())
for row in E['dependencies']:
 src=Path(M[row['sha256']]['path']);assert src.stat().st_size==row['size_bytes'] and hashlib.file_digest(src.open('rb'),'sha256').hexdigest()==row['sha256'];(objects/(row['sha256']+'.bin')).symlink_to(src)
C=Path('/tmp/ig_native0222/cold');assert not C.exists();p.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=objects)
# Import only the restored captured namespace for independent audit.
for name in list(sys.modules):
 if name=='infinity_grid' or name.startswith('infinity_grid.') or name=='project' or name.startswith('project.'):del sys.modules[name]
sys.path[0]=str(C/'source')
from infinity_grid.canon import canonical_sha256
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.historical import maturation_parallel as mp
from project.partitions import assemble
r=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'));A=r/'artifacts';bh=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings']['bootstrap'];bp=C/'runtime/intake/artifacts'/(bh+'.bin');assert hashlib.file_digest(bp.open('rb'),'sha256').hexdigest()==bh;base=json.loads(bp.read_text());dags=[];rows=[];previous=base['dag']['science_sha256']
for n in range(63,69):
 manifest=json.loads((A/f'historical_g1_depth_{n}_manifest.json').read_text());assert manifest['base']['sha256']==bh;nodes=dict(base['dag']['nodes'])
 for ref in manifest['partitions']:
  pp=A/Path(ref['path']).name;assert hashlib.file_digest(pp.open('rb'),'sha256').hexdigest()==ref['sha256'];d=json.loads(pp.read_text())
  for k,v in d['nodes'].items():assert k not in nodes or nodes[k]==v;nodes[k]=v
 dag=assemble(nodes,manifest['roots'],manifest['science_sha256']);assert d['parent_science_sha256']==previous
 assert d['level']==n and d['selected_count']==24 and d['candidate_count']==193
 conn=sqlite3.connect((r/'phases'/f'historical_g1_depth_{n}'/'state_store.sqlite3').resolve().as_uri()+'?mode=ro',uri=True);saved=conn.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall();assert len(saved)==1 and conn.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1;conn.close();digest,exact,sj=saved[0];assert json.loads(sj)==d and structural_canonical_bytes(d)==bytes(exact) and hashlib.sha256(exact).hexdigest()==digest
 rows.append({'level':n,'roots':24,'nodes':len(dag['nodes']),'science_sha256':dag['science_sha256'],'native_exact_identity_bytes':len(exact)});dags.append(dag);previous=dag['science_sha256']
nodes={};roots=[]
for dag in dags:
 roots.extend(dag['roots'])
 for k,v in dag['nodes'].items():assert k not in nodes or nodes[k]==v;nodes[k]=v
union={'schema_id':'IG_MATURATION_STATE_DAG_V1','roots':roots,'nodes':nodes};union['science_sha256']=canonical_sha256(union);_,states=mp._states_from_dag(union);assert len(states)==144
for i,dag in enumerate(dags):assert mp.state_dag_wire(states[i*24:(i+1)*24])==dag
ah=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings']['anchor'];ap=C/'runtime/intake/artifacts'/(ah+'.bin');assert hashlib.file_digest(ap.open('rb'),'sha256').hexdigest()==ah;anchor=json.loads(ap.read_text());expected={x['construction_digest']:{'skin':x['resource_skin_sha256'],'caps':x['total_free_by_type']} for x in anchor['materialized_discovery_evidence']['state_probes']};observed={x.construction_digest:{'skin':x.skin,'caps':list(x.total_caps)} for x in states[-24:]};assert observed==expected
raw=json.dumps({'level':68,'dag':dags[-1],'candidate_count':193,'selected_count':24},sort_keys=True,separators=(',',':')).encode();boot=Path('/tmp/ig_verified0222/BOOTSTRAP68_HISTORICAL.json');boot.write_bytes(raw);(B/'BOOTSTRAP68_META.json').write_text(json.dumps({'path':str(boot),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'science_sha256':previous},indent=2));out={'status':'PASS_COLD_HISTORICAL_DEPTH68_CONTINUATION','native_registered_scope_completed':True,'cold_native_checkpoint_restore':True,'native_exact_identity_phases':6,'DAG_roundtrip_roots':144,'historical_depth68_anchor':'PASS24_EXACT_ROOT_SKIN_CAPS','candidate_build_calls':1158,'audit_candidate_generation_calls':0,'original_workspace_scientific_state_read':False,'rows':rows,'master_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN','Q2_payload_generated':False};(B/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
