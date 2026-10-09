"""Derive an exact terminal input only from independent cold-restored 0230 bytes."""
from pathlib import Path
import sys,json,sqlite3,gzip,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,str(B));sys.path.insert(0,'/tmp/ig_engine0204')
from project.partitions import read,assemble
from infinity_grid.structural_encoding import structural_canonical_bytes
P=Path('/workspace/scratch/75d85ae95659/continuation0230');C=Path('/tmp/ig_native0230/cold_diagnostic')
assert json.loads((P/'AUDIT.json').read_text())['status']=='PASS_COLD_DIAGNOSTIC_SAVED_BYTES'
S=json.loads((P/'SPEC.json').read_text());h=S['execution']['parameters']['bindings'];base=read(C/'runtime/intake/artifacts'/(h['bootstrap']+'.bin'),h['bootstrap'])
R=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*'))
with sqlite3.connect((R/'phases/terminal100_diagnostic/state_store.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
 assert db.execute('SELECT COUNT(*) FROM generation_tasks').fetchone()[0]==1 and db.execute('SELECT COUNT(*) FROM states').fetchone()[0]==1
 digest,exact,sj=db.execute('SELECT index_digest,canonical_bytes,state_json FROM states').fetchone()
 data=json.loads(sj);assert bytes(exact)==structural_canonical_bytes(data) and hashlib.sha256(exact).hexdigest()==digest
nodes=dict(base['dag']['nodes'])
for k,v in data['nodes'].items():assert k not in nodes or nodes[k]==v;nodes[k]=v
dag=assemble(nodes,data['roots'],data['science_sha256']);assert len(dag['roots'])==193
obj=dict(level=100,dag=dag,candidate_count=193,selected_count=193,beam_roots=data['beam_roots'])
raw=json.dumps(obj,sort_keys=True,separators=(',',':')).encode();p=B/'BOOTSTRAP100_HISTORICAL.json.gz';p.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
observed=data['terminal_interface_population'];op=B/'DIAGNOSTIC_V2_POPULATION.json';op.write_text(json.dumps(observed,sort_keys=True,separators=(',',':')))
r=dict(status='PASS_COLD_SAVED_TERMINAL_INPUT_DERIVED',source_checkpoint=230,source_capture_id='c765b8d629bbd348182a4563623f589f4482a20f57f9452d9906469d7e9ccf1c',source_native_index_digest=digest,dag_science_sha256=dag['science_sha256'],bootstrap_raw_sha256=hashlib.sha256(raw).hexdigest(),bootstrap_transport_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bootstrap_bytes=p.stat().st_size,dag_roots=193,dag_nodes=len(dag['nodes']),diagnostic_v2_sha256=hashlib.sha256(op.read_bytes()).hexdigest(),candidate_generation_calls=0,original_live_workspace_read=False)
(B/'SOURCE_INPUT_RECOVERY.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
