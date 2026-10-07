"""Independent restoration gate; no generation of completed depths."""
from pathlib import Path
import sys,json,hashlib,sqlite3
B=Path(__file__).resolve().parent
if len(sys.argv)!=3:raise SystemExit('Usage: python-fixed-host reconstruct_saved94.py RESTORED_CHECKPOINT_ROOT OUTPUT_JSON')
C=Path(sys.argv[1]).resolve();output=Path(sys.argv[2]).resolve()
sys.path.insert(0,str(C/'source'));sys.path.insert(0,str(B))
from infinity_grid.structural_encoding import structural_canonical_bytes
from infinity_grid import maturation_parallel as mp
from project.partitions import assemble
audit=json.loads((B/'AUDIT.json').read_text());assert audit['generation_completed_depths']==[91,94]
runtime=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'))
bound=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings']['bootstrap']
base=C/'runtime/intake/artifacts'/(bound+'.bin')
assert hashlib.file_digest(base.open('rb'),'sha256').hexdigest()==bound
previous=json.loads(base.read_text())['dag'];nodes=dict(previous['nodes'])
for n in range(91,95):
    db=runtime/'phases'/f'g1_partition_depth_{n}'/'state_store.sqlite3'
    conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    rows=conn.execute('SELECT canonical_bytes,state_json FROM states').fetchall();conn.close();assert len(rows)==1
    exact,data_json=rows[0];data=json.loads(data_json)
    assert structural_canonical_bytes(data)==bytes(exact)
    assert data['level']==n and data['candidate_count']==193 and data['selected_count']==24 and data['parent_science_sha256']==previous['science_sha256']
    for key,value in data['nodes'].items():
        assert key not in nodes or nodes[key]==value
        nodes[key]=value
    previous=assemble(nodes,data['roots'],data['science_sha256'])
assert previous['science_sha256']==audit['rows'][-1]['science_sha256']
_,states=mp._states_from_dag(previous)
assert len(states)==24 and mp.state_dag_wire(states)==previous
raw=json.dumps({'level':94,'dag':previous,'candidate_count':193,'selected_count':24},sort_keys=True,separators=(',',':')).encode()
output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(raw)
print(json.dumps({'status':'PASS_INDEPENDENT_SAVED94_STATE_RECONSTRUCTION_AND_EXACT_DAG_ROUNDTRIP','bootstrap_sha256':hashlib.sha256(raw).hexdigest(),'bootstrap_bytes':len(raw),'science_sha256':previous['science_sha256'],'roots':24,'generator_calls':0,'next_missing_depths':[95,96]}))
