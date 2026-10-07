"""Cold handoff proof: exact stored identities and DAG closure, no generation."""
from pathlib import Path
import sys,json,hashlib,sqlite3
B=Path(__file__).resolve().parent;J=Path(json.loads((B/'POINTER.json').read_text())['workspace'])
sys.path.insert(0,str(J/'source'));sys.path.insert(0,str(B))
from infinity_grid import preservation as pr
from infinity_grid.structural_encoding import structural_canonical_bytes
from project.partitions import assemble
pr.confirm_batch(J,B/'ACK_BATCH.json')
status=pr.status(J);assert not status['pending_objects']
(B/'PRESERVATION_FINAL.json').write_text(json.dumps(status,indent=2))
export=pr.export_checkpoint(J,B/'NATIVE_CHECKPOINT_SLIM.zip',slim=True)
(B/'CHECKPOINT_EXPORT.json').write_text(json.dumps(export,indent=2))
mapping=json.loads((B/'READBACKS.json').read_text());objects=B/'restore_objects';objects.mkdir(exist_ok=True)
for row in export['dependencies']:
    source=Path(mapping[row['sha256']]['path'])
    assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==row['sha256']
    (objects/(row['sha256']+'.bin')).symlink_to(source)
C=Path('/tmp/ig_native0205/cold_handoff')
pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,export['sha256'],objects=objects)
runtime=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'))
assert (C/'runtime/PAUSE_ACK.json').exists()
binding=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings']['bootstrap']
base=C/'runtime/intake/artifacts'/(binding+'.bin')
assert hashlib.file_digest(base.open('rb'),'sha256').hexdigest()==binding
initial=json.loads(base.read_text());previous=initial['dag'];nodes=dict(previous['nodes']);rows=[]
assert initial['level']==90
for n in range(91,95):
    phase=runtime/'phases'/f'g1_partition_depth_{n}'
    conn=sqlite3.connect((phase/'state_store.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
    saved=conn.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchall()
    assert len(saved)==1 and conn.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1
    digest,exact,state_json=saved[0];conn.close();data=json.loads(state_json)
    assert structural_canonical_bytes(data)==bytes(exact) and hashlib.sha256(exact).hexdigest()==digest
    assert data['level']==n and data['candidate_count']==193 and data['selected_count']==24 and len(data['roots'])==24
    assert data['parent_science_sha256']==previous['science_sha256']
    for key,value in data['nodes'].items():
        assert key not in nodes or nodes[key]==value
        nodes[key]=value
    dag=assemble(nodes,data['roots'],data['science_sha256'])
    manifest=runtime/'artifacts'/f'g1_partition_depth_{n}_manifest.json'
    if n<=93:
        m=json.loads(manifest.read_text());part=m['partitions'][-1]
        path=runtime/'artifacts'/Path(part['path']).name
        assert hashlib.file_digest(path.open('rb'),'sha256').hexdigest()==part['sha256']
        assert json.loads(path.read_text())==data and m['science_sha256']==dag['science_sha256'] and m['roots']==dag['roots']
    else:assert not manifest.exists()
    summary=json.loads((phase/'SUMMARY.json').read_text())['generation']
    assert summary['task_count']==1 and summary['stored_exact_identity_canonical_bytes']==len(exact)
    rows.append({'level':n,'roots':24,'nodes':len(dag['nodes']),'science_sha256':dag['science_sha256'],'stored_identity_bytes':len(exact),'native_manifest_published':manifest.exists()})
    previous=dag
    (B/'AUDIT_PROGRESS.json').write_text(json.dumps({'checked_depths':[r['level'] for r in rows]},indent=2))
assert not (runtime/'phases/g1_partition_depth_95').exists() and not (runtime/'phases/g1_partition_depth_96').exists()
result={'status':'PASS_COLD_CHECKPOINT_EXACT_NATIVE_IDENTITIES_AND_DAG_CLOSURE','generation_completed_depths':[91,94],'native_published_manifest_depths':[91,93],'candidate_build_calls_committed':772,'native_registered_scope_completed':False,'native_pause':'USER_REQUESTED_BEFORE_DEPTH94_PUBLICATION','depths95_96_started':False,'earlier_depths_regenerated':False,'pending_bytes':0,'master_slices':151,'new_admissions':0,'terminal_comparison':'NOT_RUN','cold_workspace':str(C),'original_workspace_state_read_during_audit':False,'independent_state_object_reconstruction':'NOT_RUN_FOR91_94; required for saved94 bootstrap before further generation','relocated_controller_resume':'NOT_TESTED','rows':rows}
(B/'AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
