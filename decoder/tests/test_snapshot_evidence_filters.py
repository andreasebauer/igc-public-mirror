from pathlib import Path
import pytest
from infinity_grid.v05_controller_event_loop import _snapshot_files, ControllerLoopError

@pytest.mark.parametrize('tree',['runs','sealed'])
def test_runtime_evidence_keeps_source_like_names(tmp_path,tree):
    names=['__pycache__/old.cpython-313.pyc','file.pyo','build/result.json','dist/output','x/.pytest_cache/result','.runner.lock','.engineering_tmp/result','.git/data']
    prefix=f'runtime/{tree}/intent-one/script/work/'
    for n in names:
        p=tmp_path/(prefix+n);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(n.encode())
    files={n:p.read_bytes() for p,n in _snapshot_files(tmp_path)}
    assert files=={prefix+n:n.encode() for n in names}

def test_non_evidence_caches_still_excluded(tmp_path):
    for n in ['source/__pycache__/module.pyc','source/build/output','project/file.pyo','runtime/execution_leases/lease','durability/outbox/object','.runner.lock','project/keep.txt']:
        p=tmp_path/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x')
    assert [n for p,n in _snapshot_files(tmp_path)]==['project/keep.txt']

@pytest.mark.parametrize('tree',['runs','sealed'])
def test_evidence_cache_symlink_is_rejected(tmp_path,tree):
    target=tmp_path/'target';target.write_text('bytes')
    p=tmp_path/f'runtime/{tree}/intent-one/script/__pycache__/file.pyc';p.parent.mkdir(parents=True);p.symlink_to(target)
    with pytest.raises(ControllerLoopError,match='WORKSPACE_SNAPSHOT_SYMLINK'):list(_snapshot_files(tmp_path))

def test_preserved_input_completion_and_corruption(tmp_path):
    import json,zipfile
    from infinity_grid import preservation as pr, submission as sub
    workspace=Path(__file__).resolve().parents[2]
    captured=json.loads((workspace/'CAPTURE.json').read_text())
    obj=next(r for r in captured['objects'] if r['role']=='input:failed_evidence')
    archive=workspace/'runtime/intake/artifacts'/(obj['sha256']+'.bin')
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            rel=Path(n);assert not rel.is_absolute() and '..' not in rel.parts
            p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
    pr._verify_terminal_state_evidence(pr._state_files(tmp_path))
    bytecode=next(tmp_path.glob('runtime/sealed/*/script/work/mature/code/__pycache__/*.pyc'))
    raw=bytecode.read_bytes();bytecode.write_bytes(raw+b'CORRUPTION_CONTROL')
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):pr._state_files(tmp_path)
    bytecode.write_bytes(raw)
    pr._verify_terminal_state_evidence(pr._state_files(tmp_path))
