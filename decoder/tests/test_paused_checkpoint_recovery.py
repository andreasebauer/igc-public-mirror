"""Data-only recovery regression fixtures; no genuine save or workload authority."""
from pathlib import Path
import io,json,stat,zipfile
import pytest
from infinity_grid import preservation as pr, submission as sub, portable_registry as project
from infinity_grid import v05_controller_event_loop as loop
from tests.test_decoder06_capture import _spec


def fixture(tmp_path):
    spec=_spec(tmp_path,project=False)
    spec['inputs']=[{'logical_name':'mode_input','path':str(tmp_path/'input.bin'),'sha256':sub._sha(b'input')}]
    (tmp_path/'input.bin').write_bytes(b'input')
    store=tmp_path/'store';project.initialize(store,'UNIT RECOVERY NO WORKLOAD',spec['engine_source'])
    c=sub.capture(store,spec);return Path(c['workspace']),c


def inv(root):
    return {str(p.relative_to(root)):(p.read_bytes(),stat.S_IMODE(p.stat().st_mode)) for p in root.rglob('*') if p.is_file() and p.name!='.runner.lock'}


def export(root,tmp_path):
    p=tmp_path/'checkpoint.zip';pr.export_checkpoint(root,p);return p


def rewrite_packet(path,edit):
    with zipfile.ZipFile(path) as z:files={n:z.read(n) for n in z.namelist()}
    packet=json.loads(files['CHECKPOINT.json']);edit(packet['checkpoint'])
    packet['checkpoint_sha256']=sub._sha(sub._json_bytes(packet['checkpoint']))
    files['CHECKPOINT.json']=sub._json_bytes(packet);path.write_bytes(sub._archive(files))


def test_modes_and_mode_only_checkpoint_roundtrip(tmp_path):
    root,c=fixture(tmp_path)
    for name,mode in [('private',0o600),('readonly',0o444),('executable',0o755)]:
        p=root/'runtime'/name;p.parent.mkdir(exist_ok=True);p.write_bytes(name.encode());p.chmod(mode)
    inp=next((root/'runtime/intake/artifacts').glob('*.bin'));inp.chmod(0o400)
    first=pr.make_checkpoint(root,'UNIT');(root/'runtime/private').chmod(0o640)
    second=pr.make_checkpoint(root,'UNIT_MODE_ONLY');assert first['latest_checkpoint']!=second['latest_checkpoint']
    expected=inv(root/'runtime');p=export(root,tmp_path);dest=tmp_path/'restored'
    result=pr.restore_checkpoint(p,dest,sub._sha(p.read_bytes()))
    assert result['state_file_modes_restored'] is True and inv(dest/'runtime')==expected


@pytest.mark.parametrize('kind',['missing','extra','boolean','special','null'])
def test_bad_mode_manifest_refuses_without_destination(tmp_path,kind):
    root,c=fixture(tmp_path);pr.make_checkpoint(root,'UNIT');p=export(root,tmp_path)
    def edit(row):
        modes=row['state_file_modes'];key=next(iter(modes))
        if kind=='missing':modes.pop(key)
        elif kind=='extra':modes['../../outside']=0o644
        elif kind=='boolean':modes[key]=True
        elif kind=='special':modes[key]=0o4755
        else:row['state_file_modes']=None
    rewrite_packet(p,edit)
    with pytest.raises(sub.SubmissionError,match='RESTORE_FILE_MODE_'):
        pr.restore_checkpoint(p,tmp_path/'restored',sub._sha(p.read_bytes()))
    assert not (tmp_path/'restored').exists()


def test_legacy_restore_reports_missing_mode_guarantee(tmp_path):
    root,c=fixture(tmp_path);pr.make_checkpoint(root,'UNIT');p=export(root,tmp_path)
    rewrite_packet(p,lambda row:row.pop('state_file_modes'))
    result=pr.restore_checkpoint(p,tmp_path/'restored',sub._sha(p.read_bytes()))
    assert result['state_file_modes_restored'] is False
    assert result['mode_scope']=='LEGACY_MODES_NOT_RECORDED'


def interrupt_fixture(tmp_path,monkeypatch,refusal_failure=False):
    root,c=fixture(tmp_path)
    # Explicit unit-only admission stub. No saved receipt is fabricated.
    monkeypatch.setattr(sub,'require_saved',lambda *a,**k:None)
    def interrupt(*args):raise KeyboardInterrupt('UNIT_DISPATCH_INTERRUPT')
    monkeypatch.setattr(loop,'_dispatch_workspace_job',interrupt)
    if refusal_failure:
        from infinity_grid import invocation
        def fail(*args):raise OSError('UNIT_REFUSAL_WRITE_FAILURE')
        monkeypatch.setattr(invocation,'retain_refusal',fail)
    with pytest.raises(KeyboardInterrupt,match='UNIT_DISPATCH_INTERRUPT'):
        loop.run_workspace_job(root,c['job_id'])
    return root,c


def test_controller_pause_refusal_is_in_exact_restoration(tmp_path,monkeypatch):
    root,c=interrupt_fixture(tmp_path,monkeypatch)
    assert list((root/'runtime/refusals').glob('*.json'))
    expected=inv(root/'runtime');p=export(root,tmp_path);dest=tmp_path/'restored'
    pr.restore_checkpoint(p,dest,sub._sha(p.read_bytes()))
    assert inv(dest/'runtime')==expected
    attempts=[json.loads(p.read_text()) for p in (dest/'runtime/attempts').rglob('*.json')]
    assert len(attempts)==1 and attempts[0]['status']=='PAUSED'
    assert not list((dest/'runtime/intake/completed').glob('*.json'))


def test_refusal_save_error_preserves_interrupt_and_marks_checkpoint_failure(tmp_path,monkeypatch):
    root,c=interrupt_fixture(tmp_path,monkeypatch,True)
    failure=json.loads((root/'durability/CHECKPOINT_FAILURE.json').read_text())
    assert failure['reason']=='UNIT_REFUSAL_WRITE_FAILURE'
    assert failure['execution_reason']=='UNIT_DISPATCH_INTERRUPT'
    assert failure['status']=='CHECKPOINT_UNAVAILABLE'
