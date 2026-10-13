"""Frozen engineering checks; invoke only via Decoder registered validation."""
import json
from pathlib import Path
import shutil
import pytest

from infinity_grid import portable_registry as reg, submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.canon import canonical_sha256
from infinity_grid.invocation import InvocationRefused
from tests.test_decoder06_capture import _spec


def project(tmp_path):
    source=Path(loop.__file__).resolve().parents[1]
    store=tmp_path/'store';reg.initialize(store,'Fixture project',source)
    return store,store/'coordination'


def test_capture_automatically_binds_project_and_requires_saved_history(tmp_path):
    store,root=project(tmp_path)
    state=sub.capture(store,_spec(tmp_path,project=False));ws=Path(state['workspace'])
    _,binding=reg.locate(ws)
    assert binding['project_id']==reg.read(root/'PROJECT.json')['project_id']
    roles={o['role'] for o in sub.required_objects(ws)}
    assert {'engine_source','project_binding','project_baseline'}<=roles
    with pytest.raises(sub.SubmissionError,match='SAVE_REQUIRED'):loop.run_workspace_job(ws,state['job_id'])
    assert not (ws/'runtime/attempts').exists()


def test_engine_identity_excludes_experiment_and_display_labels(tmp_path):
    store,root=project(tmp_path);spec=_spec(tmp_path)
    first=sub.capture(store,spec);a=reg.read(Path(first['workspace'])/'PROJECT_BINDING.json')
    spec['job_id']='SECOND.LABEL';spec['question']['description']='Another display description'
    second=sub.capture(store,spec);b=reg.read(Path(second['workspace'])/'PROJECT_BINDING.json')
    assert a['work_id']==b['work_id'] and a['capture_id']!=b['capture_id']
    Path(spec['project_source'],'helper.py').write_text('value = 9\n')
    third=sub.capture(store,spec);c=reg.read(Path(third['workspace'])/'PROJECT_BINDING.json')
    assert c['identity']['engine']==a['identity']['engine']
    assert c['work_id']!=a['work_id']


def test_repeat_is_explicit_and_preserved(tmp_path):
    store,root=project(tmp_path);spec=_spec(tmp_path)
    a=sub.capture(store,spec);spec['repeat_id']='REPEAT.1'
    b=sub.capture(store,spec)
    assert reg.read(Path(a['workspace'])/'PROJECT_BINDING.json')['work_id']!=reg.read(Path(b['workspace'])/'PROJECT_BINDING.json')['work_id']
    # Capture ID must also include intentional repetition, otherwise immutable
    # bindings for the same saved experiment could be accidentally overwritten.
    assert a['capture_id']!=b['capture_id']


def test_semantic_inputs_contract_resources_and_environment_change_identity(tmp_path):
    store,root=project(tmp_path);spec=_spec(tmp_path)
    a=sub.capture(store,spec);ws=Path(a['workspace']);base=reg.identity(root,ws,None)
    for key in ('execution','question','output_contract','environment','actual_environment','resources','providers'):
        altered=dict(base);altered[key]={**base[key],'different':1}
        assert canonical_sha256(altered)!=canonical_sha256(base)


def test_modified_binding_refuses(tmp_path):
    store,root=project(tmp_path);state=sub.capture(store,_spec(tmp_path));ws=Path(state['workspace'])
    path=ws/'PROJECT_BINDING.json';b=reg.read(path);b['work_id']='a'*64;path.write_text(json.dumps(b))
    with pytest.raises(InvocationRefused,match='PROJECT_BINDING_MISMATCH'):sub.required_objects(ws)


def test_missing_project_cannot_start_registered_route(tmp_path):
    state=sub.capture(tmp_path/'unbound',_spec(tmp_path,project=False));ws=Path(state['workspace'])
    with pytest.raises(InvocationRefused,match='PROJECT_BINDING_REQUIRED'):reg.locate(ws)
    assert not (ws/'runtime/attempts').exists()


def test_stale_release_compare_and_swap_and_history_retention(tmp_path):
    store,root=project(tmp_path);head,value=reg.current(root,'release')
    with reg.lock(root):
        new=reg.append(root,'release',[head],value,'First controlled fixture change')
        with pytest.raises(InvocationRefused,match='PROJECT_STALE_PARENT'):
            reg.append(root,'release',[head],value,'Stale second fixture change')
    assert reg.current(root,'release')[0]==new and head in reg.events(root)


def test_branch_import_retains_conflict_and_requires_explicit_selection(tmp_path):
    store,root=project(tmp_path);base=reg.snapshot(root);archive=tmp_path/'base.zip';archive.write_bytes(base)
    peer=tmp_path/'peer';reg.import_history(peer,archive,sub._sha(base));peerroot=peer/'coordination'
    head,value=reg.current(root,'release')
    with reg.lock(root):a=reg.append(root,'release',[head],value,'Chat A fixture')
    with reg.lock(peerroot):b=reg.append(peerroot,'release',[head],value,'Chat B fixture')
    archive.write_bytes(reg.snapshot(peerroot));result=reg.import_history(store,archive,sub._sha(archive.read_bytes()))
    assert result['conflicts']['release']==sorted([a,b])
    with pytest.raises(InvocationRefused,match='PROJECT_HISTORY_CONFLICT'):reg.current(root,'release')
    with pytest.raises(InvocationRefused,match='REASON_REQUIRED'):reg.resolve(root,'release',a,'')
    resolved=reg.resolve(root,'release',a,'Preserve both; choose A after review')
    assert set(resolved['preserved_heads'])=={a,b}
    assert {head,a,b,resolved['head']}<=set(reg.events(root))


def test_unrelated_work_histories_merge_without_project_wide_conflict(tmp_path):
    store,root=project(tmp_path);archive=tmp_path/'base.zip';archive.write_bytes(reg.snapshot(root))
    peer=tmp_path/'peer';reg.import_history(peer,archive,sub._sha(archive.read_bytes()))
    with reg.lock(root):reg.append(root,'work:A',[],{'state':'RUNNING'},'Fixture A')
    with reg.lock(peer/'coordination'):reg.append(peer/'coordination','work:B',[],{'state':'RUNNING'},'Fixture B')
    archive.write_bytes(reg.snapshot(peer/'coordination'))
    assert reg.import_history(store,archive,sub._sha(archive.read_bytes()))['status']=='READY'
    assert reg.current(root,'work:A')[1]['state']==reg.current(root,'work:B')[1]['state']=='RUNNING'


def test_provider_requires_current_parent_and_real_check_capsules(tmp_path):
    store,root=project(tmp_path);_,release=reg.current(root,'release')
    request={'engine':release['engine'],'role':'composition','path':'infinity_grid/controller.py',
             'expected_head':'a'*64,'correctness_capsule':'b'*64,'performance_capsule':'c'*64,'reason':'Fixture replacement'}
    with pytest.raises(InvocationRefused,match='PROVIDER_CONFLICT'):reg.provider(root,request)
    request['expected_head']=None
    with pytest.raises(FileNotFoundError):reg.provider(root,request)
    assert reg.current(root,'provider:'+release['engine']+':composition')==(None,None)


def test_archive_hash_and_missing_parent_refuse(tmp_path):
    store,root=project(tmp_path);archive=tmp_path/'base.zip';archive.write_bytes(reg.snapshot(root))
    with pytest.raises(InvocationRefused,match='ARCHIVE_HASH'):reg.import_history(tmp_path/'peer',archive,'0'*64)
    head,value=reg.current(root,'release')
    with reg.lock(root):reg.append(root,'release',[head],value,'Fixture successor')
    (root/'events'/(head+'.json')).unlink()
    with pytest.raises(InvocationRefused,match='PARENT_MISSING'):reg.status(root)


def test_direct_release_activation_has_no_authority(tmp_path):
    with pytest.raises(InvocationRefused):reg.promote(tmp_path,'a'*64,tmp_path,tmp_path,'direct')
    with pytest.raises(InvocationRefused):reg.rollback_release(tmp_path,'a'*64,'b'*64,'direct')


def test_current_validation_has_native_project_claim():
    ws=Path(loop.__file__).resolve().parents[2];root,binding=reg.locate(ws)
    _,work=reg.current(root,'work:'+binding['work_id'])
    assert work['state']=='RUNNING' and work['capture_id']==sub.capture_record(ws)['capture_id']


def test_refusal_provides_usable_store_and_explicit_missing_arguments(tmp_path):
    store,root=project(tmp_path)
    with pytest.raises(InvocationRefused) as caught:reg.refuse('PROJECT_PROVIDER_CHECK_BINDING_REQUIRED',root)
    args=caught.value.as_dict()['required_arguments']
    assert reg.status(Path(args['store'])/'coordination')['status']=='READY'
    with pytest.raises(InvocationRefused) as caught:reg.refuse('PROJECT_HISTORY_CONFLICT:release',root,'project resolve')
    row=caught.value.as_dict()
    assert row['required_arguments']['key']=='release'
    assert set(row['missing_decisions'])=={'selected_head','reason'}


def test_recapture_refreshes_release_and_provider_context(tmp_path):
    store,root=project(tmp_path);spec=_spec(tmp_path,project=False)
    a=sub.capture(store,spec);ab=reg.read(Path(a['workspace'])/'PROJECT_BINDING.json')
    head,value=reg.current(root,'release')
    with reg.lock(root):reg.append(root,'release',[head],value,'Same engine restored at a newer history head')
    b=sub.capture(store,spec);bb=reg.read(Path(b['workspace'])/'PROJECT_BINDING.json')
    assert a['capture_id']!=b['capture_id'] and ab['work_id']==bb['work_id']
    with reg.lock(root):reg.append(root,'provider:'+value['engine']+':FIXTURE',[],{'provider_sha256':'a'*64},'Preparation fixture only')
    c=sub.capture(store,spec);cb=reg.read(Path(c['workspace'])/'PROJECT_BINDING.json')
    assert b['capture_id']!=c['capture_id'] and bb['work_id']!=cb['work_id']
    assert cb['identity']['engine']==bb['identity']['engine']


def test_existing_work_keeps_its_provider_binding(tmp_path):
    store,root=project(tmp_path);state=sub.capture(store,_spec(tmp_path,project=False));ws=Path(state['workspace'])
    binding=reg.read(ws/'PROJECT_BINDING.json');key='work:'+binding['work_id']
    with reg.lock(root):
        reg.append(root,key,[],{'state':'PAUSED','capture_id':binding['capture_id'],'identity':binding['identity']},'Prepared interrupted-claim fixture')
        reg.append(root,'provider:'+binding['identity']['engine']+':FIXTURE',[],{'provider_sha256':'a'*64},'Later provider fixture')
    admission=loop.validate_workspace_job(ws,state['job_id'],check_loaded=False)
    claim=reg.RunClaim(admission)
    assert claim.binding['identity']['providers']=={}


def test_reuse_lookup_uses_captured_host_but_does_not_create_a_claim(tmp_path,monkeypatch):
    store,root=project(tmp_path);state=sub.capture(store,_spec(tmp_path,project=False));ws=Path(state['workspace'])
    binding=reg.read(ws/'PROJECT_BINDING.json');key='work:'+binding['work_id']
    with reg.lock(root):
        reg.append(root,key,[],{'state':'PAUSED','capture_id':binding['capture_id'],
            'identity':binding['identity']},'Prepared non-completed compatibility fixture')
    changed=dict(binding['identity']['actual_environment'],python='0.0 changed-host fixture')
    monkeypatch.setattr(reg,'actual_environment',lambda:changed)
    admission=loop.validate_workspace_job(ws,state['job_id'],check_loaded=False)
    with pytest.raises(InvocationRefused,match='PROJECT_EXECUTION_IDENTITY_CHANGED'):
        reg.RunClaim(admission)
    probe=reg.RunClaim(admission,reuse_verification=True)
    assert probe.completed_reuse() is None
    assert reg.current(root,key)[1]['state']=='PAUSED'
