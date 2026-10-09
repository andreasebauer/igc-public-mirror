"""Cold restore the idle refusal snapshot; verify identical preflight refusal."""
from pathlib import Path
import sys,json,hashlib
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0204')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0236_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
 p=Path(M[x['sha256']]['path'])
 with p.open('rb') as f:assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
 q=O/(x['sha256']+'.bin')
 if not q.exists():q.symlink_to(p)
C=Path('/tmp/ig_native0236/cold_stop');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
 if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path.insert(0,str(C/'source'))
from infinity_grid.workflow_guard import preflight_job
from infinity_grid.invocation import InvocationRefused
from infinity_grid import submission,preservation
try:
 preflight_job({'source':C/'source','job':submission.capture_record(C)['job']})
 raise AssertionError('EXPECTED_IDENTICAL_PREFLIGHT_REFUSAL')
except InvocationRefused as e:
 expected=json.loads((B/'PREFLIGHT_DIAGNOSIS.json').read_text());assert e.code==expected['code'] and e.checks==expected['checks']
pre=json.loads((B.parent/'continuation0235/PREREGISTRATION.json').read_text())
for n,h in pre['source_sha256'].items():
 p=C/'source/infinity_grid/uplift_structural.py' if n.startswith('engine/') else C/'source'/n
 assert hashlib.sha256(p.read_bytes()).hexdigest()==h
p=preservation.status(C);assert not p['pending_objects']
out=dict(status='PASS_COLD_RESTORED_NATIVE_REFUSAL',refusal_exact=True,scientific_source_pins_unchanged=True,checkpoint_sha256=E['sha256'],pending_bytes=0,official_cases_executed=0,master_slices=152,new_admissions=0,G2_promotion=False)
(B/'COLD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
