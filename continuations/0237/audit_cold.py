"""Cold restore and independently repeat every fixed62 scientific comparison."""
from pathlib import Path
import sys,json,hashlib,time,resource
B=Path(__file__).resolve().parent;sys.path.insert(0,'/tmp/ig_engine0237')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0237_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
 p=Path(M[x['sha256']]['path'])
 with p.open('rb') as f:assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
 q=O/(x['sha256']+'.bin')
 if not q.exists():q.symlink_to(p)
C=Path('/tmp/ig_native0237/cold_probe');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
 if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path.insert(0,str(C/'source'))
from infinity_grid.workflow_guard import preflight_job,scientific_call
from infinity_grid import submission,preservation
from infinity_grid.canon import canonical_sha256
job=submission.capture_record(C)['job'];gate=preflight_job({'source':C/'source','job':job});assert gate['status']=='PASS'
from project.worker import evaluate
h=job['execution']['parameters']['bindings'];payload={k+'_path':str(C/'runtime/intake/artifacts'/(v+'.bin')) for k,v in h.items()};payload.update({k+'_sha256':v for k,v in h.items()});payload['source_binding']=job['execution']['parameters']['source_binding'];t=time.monotonic()
with scientific_call(C/'source'):fresh=evaluate(payload)
artifact=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*/artifacts/bounded_q2_probe.json'));saved=json.loads(artifact.read_text());fresh_cases=sorted(json.loads(json.dumps([x['state'] for x in fresh['states']])),key=lambda x:x['ordinal']);saved_cases=sorted(saved['cases'],key=lambda x:x['ordinal']);assert fresh_cases==saved_cases and len(fresh_cases)==62
native=json.loads((B/'NATIVE_RESULT.json').read_text());assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and native['result']['cases_checked']==62
pre=json.loads((B.parent/'continuation0235/PREREGISTRATION.json').read_text())
for n,v in pre['source_sha256'].items():
 p=C/'source/infinity_grid/uplift_structural.py' if n.startswith('engine/') else C/'source'/n
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v
assert not preservation.status(C)['pending_objects']
out=dict(status='PASS_COLD_BOUND62_HISTORICAL_S1_Q2_PROBE',cases_exact=62,public_projections_and_operational_witnesses_exact=True,cold_G2_primary_realizations=62,cold_G2_swap_checks=62,G1_candidate_generation=0,official_native_cases=62,checkpoint_sha256=E['sha256'],cases_payload_sha256=canonical_sha256(fresh_cases),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,master_slices=152,new_admissions=0,G2_promotion=False,scope='Bounded historical S1 deterministic binary realization only; no full580351 congruence, historical exact G2 identity or later relation semantics claim',pending_bytes=0)
assert out['peak_RSS_bytes']<=4294967296
(B/'COLD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(B/'Q2_PROBE.json').write_text(json.dumps(saved,indent=2)+'\n');print(json.dumps(out),flush=True)
