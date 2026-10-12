"""Registered orchestration UNIT tests using genuine, immutable saved output bytes.
Admission/save/prerequisite services and project RunClaim are explicitly stubbed
only here, inside registered change validation. No workload is dispatched, no
receipt is fabricated, and this suite is not end-to-end native qualification.
The actual verified_completion checker and actual saved completion/evidence are
used. Historical source, results, and evidence are never rewritten in place.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,zipfile
import pytest
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid import submission as sub
from infinity_grid import result_contracts as rc
from infinity_grid import preservation as pr
from infinity_grid import portable_registry as project

FIXTURE_SHA='a63ba44ddd0bd485262414694e10a77fd996209d1c5c9923ba6bee28482c49c5'
SAVED_COMPLETION='095c850877dd2c675a4fd033a7763c31b6c1557a6e8c43af3d5d9783ac0e2a58'
STUB={'unit_test_only':'SAVED_CAPSULE_CACHE_STUB_NO_EXECUTION'}


def case(tmp_path,monkeypatch,cached=True,local=True):
    raw=(Path(__file__).parent/'fixtures/repair7r_completed_state.zip').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==FIXTURE_SHA
    root=tmp_path/'unit_workspace';root.mkdir()
    with zipfile.ZipFile(Path(__file__).parent/'fixtures/repair7r_completed_state.zip') as z:
        for n in z.namelist():
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
            if n=='CAPTURE.json' or n.startswith(('runtime/runs/','runtime/intake/completed/','runtime/intake/pending/')):
                dst=root/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(n))
    capture=json.loads((root/'CAPTURE.json').read_text());job=capture['job']
    pending=list((root/'runtime/intake/pending').glob('*.json'));assert len(pending)==1
    req=json.loads(pending[0].read_text())
    completed=list((root/'runtime/intake/completed').glob('*.json'));assert len(completed)==1
    completion=completed[0];done=json.loads(completion.read_text())
    assert done['completion_sha256']==SAVED_COMPLETION
    admission={'workspace':root,'job':job,'request':req,'source_sha256':job['source_sha256']}
    # Before any mutation, the actual checker verifies the genuine saved bytes.
    verify=loop.verified_completion;assert verify(admission)==done
    if not local:completion.unlink()
    trace=[]
    def verification(a, **kwargs):
        # Forward the dev82 pending-checkpoint option to the real checker.
        # The genuine historical completion/evidence and refusal checks remain.
        trace.append('VERIFY_LOCAL');return verify(a, **kwargs)
    monkeypatch.setattr(loop,'verified_completion',verification)
    monkeypatch.setattr(loop,'validate_workspace_job',lambda *a,**k:admission)
    monkeypatch.setattr(sub,'require_saved',lambda *a,**k:None)
    monkeypatch.setattr(rc,'prerequisites',lambda *a,**k:None)
    monkeypatch.setattr(pr,'backlog',lambda *a,**k:None)
    class Claim:
        reused=STUB if cached else None
        def __init__(self,a, *, reuse_verification=None):trace.append('CLAIM_INIT')
        def completed_reuse(self):trace.append('CLAIM_REUSE');return self.reused
        def __enter__(self):trace.append('CLAIM_ENTER');return self
        def __exit__(self,*args):trace.append('CLAIM_EXIT')
        def complete(self,d):
            assert d==done;trace.append('CLAIM_COMPLETE')
    monkeypatch.setattr(project,'RunClaim',Claim)
    def never(*a,**k):raise AssertionError('UNIT_TEST_MUST_NOT_DISPATCH_WORKLOAD')
    monkeypatch.setattr(loop,'_dispatch_workspace_job',never)
    return root,job['job_id'],completion,done,trace


@pytest.mark.parametrize('mutation',['whitespace','value','missing_output','extra_output','completion_binding'])
def test_bad_local_evidence_refuses_before_cached_return(tmp_path,monkeypatch,mutation):
    root,job,completion,done,trace=case(tmp_path,monkeypatch)
    outputs=list((root/'runtime/runs').glob('*/chain/decoder_stage_runtime/*/phases/*/SUMMARY.json'))
    assert len(outputs)==1
    summary=outputs[0];original_completion=completion.read_bytes()
    if mutation=='whitespace':summary.write_bytes(summary.read_bytes()+b'\n ')
    elif mutation=='value':summary.write_text(json.dumps({'deliberately_changed_value':999})+'\n')
    elif mutation=='missing_output':summary.unlink()
    elif mutation=='extra_output':(summary.parent/'extra.txt').write_bytes(b'unexpected extra evidence')
    else:
        body=json.loads(completion.read_text());body['source_sha256']='0'*64
        completion.write_text(json.dumps(body))
    with pytest.raises(loop.ControllerLoopError,match='COMPLETION_EVIDENCE_MISMATCH'):
        loop.run_workspace_job(root,job)
    assert 'CLAIM_ENTER' not in trace
    assert not (root/'runtime/attempts').exists()
    if mutation!='completion_binding':assert completion.read_bytes()==original_completion
    print('IG_BENCHMARK_JSON:'+json.dumps({'case':mutation,'refused_before_cache':True,'workload_dispatched':False}))


def test_good_local_evidence_is_checked_before_cached_return(tmp_path,monkeypatch):
    root,job,completion,done,trace=case(tmp_path,monkeypatch)
    before=completion.read_bytes()
    assert loop.run_workspace_job(root,job)==STUB
    assert trace.index('VERIFY_LOCAL')<trace.index('CLAIM_REUSE')
    assert completion.read_bytes()==before and not (root/'runtime/attempts').exists()


def test_clean_capture_without_local_completion_keeps_cache_reuse(tmp_path,monkeypatch):
    root,job,completion,done,trace=case(tmp_path,monkeypatch,local=False)
    assert loop.run_workspace_job(root,job)==STUB
    assert not completion.exists() and not (root/'runtime/attempts').exists()


def test_valid_local_completion_without_cache_does_not_execute(tmp_path,monkeypatch):
    root,job,completion,done,trace=case(tmp_path,monkeypatch,cached=False)
    before=completion.read_bytes();result=loop.run_workspace_job(root,job)
    assert {k:v for k,v in result.items() if k!='preservation'}==dict(done,reused=True)
    assert result['preservation']['status']=='NO_CHECKPOINT'
    assert result['preservation']['latest_checkpoint'] is None  # Historical fixture is not retroactively checkpointed.
    assert 'CLAIM_COMPLETE' in trace
    assert completion.read_bytes()==before and not (root/'runtime/attempts').exists()
