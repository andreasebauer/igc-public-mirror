"""Independent cold replay of the62 fixed complete-class Q2 cases."""
from pathlib import Path
import sys,json,hashlib,time,resource
B=Path(__file__).resolve().parent;Q=Path('/workspace/scratch/2a87972b51af/continuation0240');sys.path.insert(0,'/tmp/ig_engine0237')
from infinity_grid import preservation as pr
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path('/tmp/ig_cold0265_067_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
 p=Path(M[x['sha256']]['path'])
 with p.open('rb') as f:assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
 q=O/(x['sha256']+'.bin')
 if not q.exists():q.symlink_to(p)
C=Path('/tmp/ig_native0265_067/cold_probe');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
for n in list(sys.modules):
 if n=='infinity_grid' or n.startswith('infinity_grid.') or n=='project' or n.startswith('project.'):del sys.modules[n]
sys.path.insert(0,str(C/'source'))
from infinity_grid.workflow_guard import preflight_job,scientific_call
from infinity_grid import submission,preservation
from infinity_grid.canon import canonical_sha256
job=submission.capture_record(C)['job'];assert preflight_job({'source':C/'source','job':job})['status']=='PASS'
from project.worker import evaluate,read_bound
h=job['execution']['parameters']['bindings'];payload={k+'_path':str(C/'runtime/intake/artifacts'/(v+'.bin')) for k,v in h.items()};payload.update({k+'_sha256':v for k,v in h.items()});payload['source_binding']=job['execution']['parameters']['source_binding'];t=time.monotonic()
with scientific_call(C/'source'):fresh=evaluate(payload)
artifact=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*/artifacts/complete_class_Q2_067.json'));saved=json.loads(artifact.read_text());data=sorted(json.loads(json.dumps([x['state'] for x in fresh['states']])),key=lambda x:x['ordinal']);assert data==sorted(saved['cases'],key=lambda x:x['ordinal']) and len(data)==62
cases=read_bound(payload['cases_path'],h['cases']);by={x['ordinal']:x for x in cases['cases']};groups={}
for row in data:
 key=by[row['ordinal']]['record']['outcome_science_sha256'];q=row['public_projection']
 if key in groups:assert groups[key]==q
 groups[key]=q
assert set(groups)==set(cases['class_keys']) and len(groups)==31
pre=json.loads((Q/'PREREGISTRATION.json').read_text())
for n,v in pre['source_sha256'].items():
 p=C/'source/infinity_grid'/n.split('/',1)[1] if n.startswith('engine/') else C/'source'/n
 assert hashlib.sha256(p.read_bytes()).hexdigest()==v
native=json.loads((B/'NATIVE_RESULT.json').read_text());assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and native['result']['complete_Q2_classes_checked']==31
assert not preservation.status(C)['pending_objects']
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<=pre['memory_budget_bytes']
out=dict(status='PASS_COLD_COMPLETE_CLASS_Q2_PILOT',cases_exact= 62,complete_Q2_classes_exact= 31,public_projections_and_operational_witnesses_exact=True,cold_G2_primary_realizations= 62,cold_G2_swap_checks= 62,G1_candidate_generation=0,checkpoint_sha256=E['sha256'],cases_payload_sha256=canonical_sha256(data),native_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=peak,master_slices=152,new_admissions=0,G2_promotion=False,full_historical_public_observer_compared=False,pending_bytes=0)
(B/'COLD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(B/'Q2_PROBE.json').write_text(json.dumps(saved,indent=2)+'\n');print(json.dumps(out),flush=True)
