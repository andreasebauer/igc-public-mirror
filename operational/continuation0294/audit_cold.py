"""Independent preserved-checkpoint replay of one unchanged official observer chunk."""
from pathlib import Path
import sys,json,hashlib,time,resource
B=Path(__file__).resolve().parent;idx=int(B.name);Q=B.parent.parent/'continuation0285';sys.path.insert(0,'/tmp/ig_engine0237')
from infinity_grid import preservation as pr
reg=json.loads((Q/'PREREGISTRATION.json').read_text());row=reg['specifications'][idx];ncase=row['case_count'];nclass=row['class_count']
E=json.loads((B/'CHECKPOINT_EXPORT.json').read_text());M=json.loads((B/'READBACKS.json').read_text());O=Path(f'/tmp/ig_cold0294_{idx:03d}_objects');O.mkdir(exist_ok=True)
for x in E['dependencies']:
    p=Path(M[x['sha256']]['path'])
    with p.open('rb') as f:assert p.stat().st_size==x['size_bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
    q=O/(x['sha256']+'.bin')
    if not q.exists():q.symlink_to(p)
C=Path(f'/tmp/ig_native0294_{idx:03d}/cold_probe');assert not C.exists();pr.restore_checkpoint(B/'NATIVE_CHECKPOINT_SLIM.zip',C,E['sha256'],objects=O)
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
chunk=job['execution']['parameters']['chunk_id'];artifact=next(C.glob('runtime/runs/*/chain/decoder_stage_runtime/*/artifacts/'+chunk+'.json'));saved=json.loads(artifact.read_text());data=sorted(json.loads(json.dumps([x['state'] for x in fresh['states']])),key=lambda x:x['ordinal']);assert data==sorted(saved['cases'],key=lambda x:x['ordinal']) and len(data)==ncase
cases=read_bound(payload['cases_path'],h['cases']);by={x['ordinal']:x for x in cases['cases']};groups={}
for row in data:
    key=by[row['ordinal']]['record']['outcome_science_sha256'];observer=row['public_observer']
    assert row['historical_observer_hash_exact']
    if key in groups:assert groups[key]==observer
    groups[key]=observer
assert set(groups)==set(cases['class_keys']) and len(groups)==nclass
reg=json.loads((Q/'PREREGISTRATION.json').read_text())
for n,v in reg['source_sha256'].items():assert hashlib.sha256((C/'source'/n).read_bytes()).hexdigest()==v
for n,v in reg['required_engine_files'].items():assert hashlib.sha256((C/'source/infinity_grid'/n.split('/',1)[1]).read_bytes()).hexdigest()==v
native=json.loads((B/'NATIVE_RESULT.json').read_text());assert native['status']=='COMPLETED' and native['evidence_status']=='VERIFIED' and native['result']['complete_observer_classes_checked']==nclass
assert not preservation.status(C)['pending_objects']
peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;assert peak<=reg['memory_budget_bytes']
out=dict(status='PASS_COLD_OFFICIAL_HISTORICAL_OBSERVER_CHUNK',chunk_index=idx,cases_exact=ncase,complete_observer_classes_exact=nclass,full_public_observer_payloads_hashes_Q2_and_operational_witnesses_exact=True,checkpoint_sha256=E['sha256'],cases_payload_sha256=canonical_sha256(data),native_artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-t,peak_RSS_bytes=peak,master_slices=152,new_admissions=0,G2_promotion=False,full_historical_public_observer_compared=False,coverage_credit_pending_final_saved_batch=True,pending_bytes=0)
(B/'COLD_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');(B/'OBSERVER_PROBE.json').write_text(json.dumps(saved,indent=2)+'\n');print(json.dumps(out),flush=True)
