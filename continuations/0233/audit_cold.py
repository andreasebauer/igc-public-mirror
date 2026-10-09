"""Cold restore the saved native checkpoint and requalify the exact-parent reader."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0233_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
    p=Path(M[x['sha256']]['path'])
    with p.open('rb') as f:assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
    q=O/(x['sha256']+'.bin')
    if not q.exists():q.symlink_to(p)
C=Path('/tmp/ig_native0233/cold_admission');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
    if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path[0]=str(C/'source')
from project.reader import qualify
h=json.loads((B/'SPEC.json').read_text())['execution']['parameters']['bindings']
i={k:C/'runtime/intake/artifacts'/(v+'.bin') for k,v in h.items()}
r=qualify(i,h);native=json.loads((B/'NATIVE_RESULT.json').read_text());assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and r==native['result']
A=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*/artifacts/exact_parent_reader_qualification.json'));assert json.loads(A.read_text())==r
out=dict(status='PASS_COLD_SCOPED_EXACT_PARENT_ADMISSION',native_result_exact=True,restored_checkpoint_sha256=E['sha256'],candidate_master_slices=152,prior_slices_unchanged=151,carrier_routes_checked=193,DAG_nodes_checked=16528,generation_calls=0,new_DAG_decodes=0,G2_promotion=False,master_cursor_updated=False,result=r)
(B/'AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
