"""Official engineering probe: real saved evidence, no scientific execution."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import tempfile
import traceback
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'candidate/source'))
sys.path.insert(0,str(HERE/'candidate/operational'))
from infinity_grid import project_checkpoint as pc, portable_registry as reg
from infinity_grid import preservation as pr, result_contracts as rc, submission as sub
from infinity_grid import v05_controller_event_loop as loop
from p0_control import write,read,sha,require
from tests.test_decoder06_capture import _spec

ROOT=Path('/tmp/ig_p0_project_closure154_r2')
RESULT=HERE/'results/NATIVE_MATRIX.json'
rows=[]

def record(name,**fields):
    rows.append({'operation':name,'status':'PASS',**fields})
    write(RESULT,{'status':'RUNNING','rows':rows,'scientific_dispatches':0})

def main():
    require(not ROOT.exists(),'DISPOSABLE_ROOT_EXISTS')
    require(shutil.disk_usage('/tmp').free >= 10*1024**3,'MINIMUM_FREE_DISK')
    ROOT.mkdir()
    prereg=read(HERE/'PREREGISTRATION.json')
    sys.path.insert(0,str(HERE/'candidate/overlay_tools'))
    from restore_candidate import inventory
    inv=inventory(HERE/'candidate/source')
    require(hashlib.sha256(json.dumps(inv,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            == prereg['candidate_inventory_sha256'],'CANDIDATE_SOURCE_PIN')
    original=Path(prereg['fixture_source'])
    saved=HERE.parent/'p0_recovery_gate/PREREGISTRATION_REVISION4.json'
    old=read(saved)
    with zipfile.ZipFile(Path(old['saved_unit'])/'NATIVE_CHECKPOINT_SLIM.zip') as z:
        packet=json.loads(z.read('CHECKPOINT.json'))
    baseline_sha=packet['checkpoint']['project']['base']['sha256']
    baseline=Path(next(x['path'] for x in old['dependencies'] if x['sha256']==baseline_sha)).read_bytes()
    limits=rc.policy({'preservation':{'project_evidence_externalization':1}})
    with tempfile.TemporaryDirectory(dir=ROOT) as d:
        closure=pc.prepare(original,baseline,limits,d)
        full=pc.hydrate(closure['metadata'],closure['evidence_path'].read_bytes(),
                        dict(closure,schema_id=pc.SCHEMA),limits)
        require(full==reg.snapshot(original),'SAVED_FAILURE_FULL_PROJECT_MISMATCH')
        record('EXACT_SAVED_FAILURE_SPLIT_AND_HYDRATE',full_project_sha256=sub._sha(full),
               metadata_raw_bytes=closure['metadata_raw_bytes'],evidence_raw_bytes=closure['evidence_raw_bytes'],
               evidence_file_count=closure['evidence_file_count'],evidence_zip_bytes=closure['evidence_path'].stat().st_size,
               historical_capture_changed=False)
    spec=_spec(ROOT,project=False)
    spec['output_contract']={'schema_id':rc.SCHEMA,'claim':'EXECUTION_ONLY','required_artifacts':[],
                            'result_checks':[],'prerequisites':[],
                            'preservation':{'project_evidence_externalization':1}}
    spec['job_id']='P0.DEV154.PROJECT.CLOSURE.ENGINEERING.ONLY'
    store=ROOT/'store';reg.initialize(store,'Dev154 bounded closure engineering capture',spec['engine_source'])
    project=store/'coordination'
    diagnostic=read(HERE.parent/'p0_recovery_gate/results/PROJECT_RAW_BOUND_DIAGNOSTIC.json')
    capsule_sha=diagnostic['completion_capsules'][0]['capsule_sha256']
    capsule=json.loads(reg.blob(original,capsule_sha))
    for digest in sorted(set(capsule['files'].values())|{capsule_sha}):
        raw=reg.blob(original,digest)
        require(reg.put(project,raw)==digest,'HISTORICAL_EVIDENCE_COPY_CHANGED')
    with reg.lock(project):
        reg.append(project,'evidence:historical-completion',[],{'capsule':capsule_sha},
                   'Exact saved historical evidence as data; no new completion claim')
    capture=sub.capture(store,spec);work=Path(capture['workspace'])
    current=pr.make_checkpoint(work,'P0_BOUNDED_PROJECT_EVIDENCE_ENGINEERING')
    require(current['status']=='SAVE_REQUIRED','NEW_CAPTURE_CHECKPOINT_STATUS')
    checkpoint=read(pr._root(work)/'commits'/(current['latest_checkpoint']+'.json'))
    require(checkpoint['project']['schema_id']==pc.SCHEMA,'NATIVE_V2_PROTOCOL')
    require(pr.make_checkpoint(work,'P0_REPEAT_SAME_CLOSURE')['latest_checkpoint']==current['latest_checkpoint'],
            'REPEATED_CHECKPOINT_NOT_REUSED')
    plan=pr.drain_plan(work)
    require(any(r['role']=='checkpoint_project_evidence' for r in plan['obligations']),
            'EXTERNAL_EVIDENCE_ACK_ROLE_MISSING')
    exported=ROOT/'before-ack.zip';pr.export_checkpoint(work,exported)
    cold=ROOT/'cold-before-ack';pr.restore_checkpoint(exported,cold,sha(exported))
    require(reg.snapshot(cold/'coordination')==reg.snapshot(project),'NATIVE_COLD_PROJECT_BYTES_CHANGED')
    require(reg.verify_capsule(cold/'coordination',capsule_sha)['completion_sha256']==capsule['completion_sha256'],
            'HISTORICAL_COMPLETION_CHANGED')
    record('NEW_CAPTURE_NATIVE_V2_CHECKPOINT_AND_COLD_RESTORE',capture_id=capture['capture_id'],
           checkpoint_sha256=current['latest_checkpoint'],project_sha256=checkpoint['project']['full_sha256'],
           native_obligations=len(plan['obligations']),physical_objects=len(plan['physical_objects']),
           readback_bytes=plan['readback_bytes'],historical_capsule_sha256=capsule_sha,
           exact_cold_project=True,repeated_snapshot_reused=True,provider_acknowledged=False,baseline_already_contains_completion_evidence=True)
    write(HERE/'results/LIVE_PENDING_OBJECTS.json',{'workspace':str(work),'native_plan':plan,
              'checkpoint':current['latest_checkpoint'],'root':str(ROOT),'capsule_sha256':capsule_sha})
    write(RESULT,{'status':'AWAITING_EXACT_PROVIDER_READBACKS','rows':rows,'scientific_dispatches':0,
                  'full_native_fault_matrix_qualified':False})

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write(RESULT,{'status':'STOPPED_AT_FIRST_NATIVE_MISMATCH','rows':rows,
                      'error':str(exc),'traceback':traceback.format_exc(),'scientific_dispatches':0,
                      'full_native_fault_matrix_qualified':False})
        raise
