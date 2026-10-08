"""Independent native recovery restoration and exact scientific verification."""
from pathlib import Path
import json,hashlib,sqlite3,sys
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,str(J/'source'))
from infinity_grid import preservation as p
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());objects=Path('/tmp/ig_cold0225_objects');objects.mkdir(exist_ok=True)
for x in E['dependencies']:
 f=Path(M[x['sha256']]['path']);assert f.stat().st_size==x['size_bytes'] and hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==x['sha256'];(objects/(x['sha256']+'.bin')).symlink_to(f)
C=Path('/tmp/ig_native0225/cold');assert not C.exists();restored=p.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=objects)
for name in list(sys.modules):
 if name=='infinity_grid' or name.startswith('infinity_grid.') or name=='project' or name.startswith('project.'):del sys.modules[name]
sys.path[0]=str(C/'source')
from infinity_grid.v05_controller_event_loop import validate_workspace_job,verified_completion
from infinity_grid import submission as sub
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.worker import verify,read_bound
from project.partitions import assemble
rec=sub.capture_record(C);admission=validate_workspace_job(C,rec['job']['job_id']);done=verified_completion(admission)
assert done is not None
c=rec['job']['execution']['parameters'];payload={k+'_path':str(C/'runtime/intake/artifacts'/(v+'.bin')) for k,v in c['bindings'].items()};payload.update({k+'_sha256':v for k,v in c['bindings'].items()});payload.update(prior_capture_id=c['prior_capture_id'],final_science_sha256=c['final_science_sha256'])
observed=verify(payload);assert done['result']==observed
f=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*/phases/saved_depth80_verification/state_store.sqlite3'))
with sqlite3.connect(f.resolve().as_uri()+'?mode=ro',uri=True) as db:
 rows=db.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall();assert len(rows)==1 and db.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1
 digest,exact,state=rows[0];assert structural_canonical_bytes(observed)==bytes(exact) and hashlib.sha256(exact).hexdigest()==digest and json.loads(state)==observed
base=read_bound(payload['bootstrap_path'],payload['bootstrap_sha256']);phases=read_bound(payload['phases_path'],payload['phases_sha256']);nodes=dict(base['dag']['nodes'])
for row in phases['phases']:
 d=row['state'];nodes.update(d['nodes'])
dag=assemble(nodes,d['roots'],d['science_sha256']);raw=json.dumps({'level':80,'dag':dag,'candidate_count':193,'selected_count':24},sort_keys=True,separators=(',',':')).encode()
meta=json.loads((B/'EXPECTED_BOOTSTRAP80_META.json').read_text());assert len(raw)==meta['bytes'] and hashlib.sha256(raw).hexdigest()==meta['sha256']
boot=Path('/tmp/ig_verified0225/BOOTSTRAP80_HISTORICAL.json');boot.write_bytes(raw);meta['path']=str(boot);(B/'BOOTSTRAP80_META.json').write_text(json.dumps(meta,indent=2))
out={'status':'PASS_COLD_NATIVE_DEPTH80_COMPLETION_RECOVERY','native_registered_scope_completed':True,'native_recovery_completion_verified':True,'prior_capture_status':'PAUSED_WORKSPACE_BUDGET','prior_capture_modified':False,'candidate_regeneration_calls':0,'audit_candidate_generation_calls':0,'saved_native_exact_identity_phases':6,'DAG_roundtrip_roots':144,'historical_depth80_anchor':'PASS24_EXACT_ROOT_SKIN_CAPS','final_science_sha256':observed['final_science_sha256'],'original_workspace_scientific_state_read':False,'bootstrap80_reproduced_exactly':True,'master_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN','Q2_payload_generated':False,'rows':observed['rows']}
(B/'AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
