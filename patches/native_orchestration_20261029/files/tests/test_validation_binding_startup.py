"""Binding scans retain exact identity and become visible before scanning."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_streamed_binding_matches_original_concatenation(tmp_path,monkeypatch):
    from infinity_grid import validation_frontier as frontier
    source=tmp_path/'source';source.mkdir()
    (source/'a.py').write_bytes(b'x=1\n')
    (source/'operand').write_bytes(b'z'*(2*1048576+37))
    (source/'__pycache__').mkdir()
    (source/'__pycache__/skip.pyc').write_bytes(b'ignored')
    prefix=tmp_path/'runtime/base';prefix.mkdir(parents=True)
    for name in ('base/a','native/b','host/c'):
        path=prefix.parent/name;path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode()*20)
    expected_source=hashlib.sha256();expected_runtime=hashlib.sha256()
    for p in sorted(source.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        expected_source.update(p.relative_to(source).as_posix().encode());expected_source.update(b'\0');expected_source.update(p.read_bytes())
    for directory in (prefix,prefix.parent/'native',prefix.parent/'host'):
        for p in sorted(directory.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                expected_runtime.update(str(p.relative_to(prefix.parent)).encode());expected_runtime.update(b'\0');expected_runtime.update(p.read_bytes())
    monkeypatch.setattr(frontier,'__file__',str(source/'__init__.py'))
    monkeypatch.setattr(frontier.sys,'prefix',str(prefix))
    def no_whole_reads(path):raise AssertionError('whole-file binding read')
    monkeypatch.setattr(Path,'read_bytes',no_whole_reads)
    actual=frontier.implementation_binding()
    assert actual['source_sha256']==expected_source.hexdigest()
    assert actual['runtime_sha256']==expected_runtime.hexdigest()


def test_binding_failure_is_reported_by_started_observer(tmp_path,monkeypatch):
    from infinity_grid import accepted_generation_runner as runner
    from infinity_grid import acceptance_observer as observer
    from infinity_grid import validation_frontier as frontier
    runtime=SimpleNamespace(runtime_root=tmp_path)
    status=tmp_path/'full_generation/telemetry/ACCEPTANCE_STATUS.json'
    class BindingFailed(RuntimeError):pass
    def fail_binding():
        data=json.loads(status.read_text())
        assert data['status']=='RUNNING'
        assert data['identity']['source_sha256']=='fixture-source'
        assert observer._ACTIVE.get() is not None
        raise BindingFailed('scan failure')
    monkeypatch.setattr(frontier,'implementation_binding',fail_binding)
    with pytest.raises(BindingFailed):
        runner.run(runtime,{}, {}, {'source_sha256':'fixture-source'})
    data=json.loads(status.read_text())
    assert data['status']=='FAILED'
    assert data['termination']['exception_type']=='BindingFailed'
    assert not (tmp_path/'full_generation/validation_authority').exists()
