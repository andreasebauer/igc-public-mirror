"""Data/fixture checks run only inside the registered native validation job."""
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import zipfile

import pytest
from infinity_grid import result_contracts as rc
from infinity_grid import preservation as pr
from infinity_grid import validation_reports as vr
from infinity_grid import submission as sub
from infinity_grid.canon import canonical_sha256, write_json_atomic


def contract(**kw):
    row={'schema_id':rc.SCHEMA,'claim':'SCIENCE','required_artifacts':[],
         'result_checks':[{'pointer':'/outcome','equals':'NO_EFFECT'}], 'prerequisites':[],'preservation':{}}
    row.update(kw);return row


def test_negative_scientific_result_can_be_verified(tmp_path):
    c=contract(required_artifacts=[{'path':'result.json','json_checks':[{'pointer':'/rows','count':0}]}])
    write_json_atomic(tmp_path/'result.json',{'rows':[]})
    n=rc.normalize(c,{'kind':'SCRIPT'},{'outcomes':['NO_EFFECT']})
    v=rc.verify(n,{'outcome':'NO_EFFECT'},tmp_path)
    assert v['status']=='VERIFIED' and v['scientific_outcome']=='NO_EFFECT'


def test_descriptive_contract_does_not_certify_science(tmp_path):
    c=rc.normalize({'answer':'interesting'}, {'kind':'SCRIPT'}, {})
    r=rc.verify(c,{'status':'PASS','outcome':'PROCESS_COMPLETED'},tmp_path)
    assert r['status']=='NOT_DECLARED' and r['scientific_outcome'] is None


def test_script_scientific_outcome_is_read_from_declared_artifact(tmp_path):
    write_json_atomic(tmp_path/'answer.json',{'outcome':'NO_EFFECT'})
    c=contract(required_artifacts=[{'path':'answer.json'}],result_checks=[{'pointer':'/status','equals':'PASS'}],
               outcome={'artifact':'answer.json','pointer':'/outcome'})
    c=rc.normalize(c,{'kind':'SCRIPT'},{'outcomes':['NO_EFFECT','EFFECT']})
    r=rc.verify(c,{'status':'PASS','outcome':'PROCESS_COMPLETED'},tmp_path)
    assert r['status']=='VERIFIED' and r['scientific_outcome']=='NO_EFFECT'


def test_missing_output_is_rejected(tmp_path):
    r=rc.verify(contract(required_artifacts=[{'path':'missing.json'}]),{'outcome':'NO_EFFECT'},tmp_path)
    assert r['status']=='REJECTED'
    assert r['failures'][0]['reason']=='ARTIFACT_MISSING_OR_UNSAFE'


def test_wrong_hash_count_and_typed_value_are_rejected(tmp_path):
    write_json_atomic(tmp_path/'result.json',{'rows':[1]})
    c=contract(required_artifacts=[{'path':'result.json','sha256':'0'*64,'json_checks':[{'pointer':'/rows','count':0}]}])
    r=rc.verify(c,{'outcome':'NO_EFFECT'},tmp_path)
    assert len(r['failures'])==2 and r['status']=='REJECTED'
    assert rc.evaluate_checks({'x':True},[{'pointer':'/x','equals':1}])


def test_bad_contract_paths_schema_and_limits_refuse():
    for c in [contract(schema_id='UNKNOWN'),contract(required_artifacts=[{'path':'../outside'}]),
              contract(result_checks=[]),contract(preservation={'max_commit_bytes':20,'max_pending_bytes':10})]:
        with pytest.raises(sub.SubmissionError):rc.normalize(c,{'kind':'SCRIPT'},{})


def test_output_symlink_is_rejected(tmp_path):
    outside=tmp_path/'outside.json';outside.write_text('{}')
    out=tmp_path/'run';out.mkdir();(out/'result.json').symlink_to(outside)
    assert rc.verify(contract(required_artifacts=[{'path':'result.json'}]),{'outcome':'NO_EFFECT'},out)['status']=='REJECTED'


def test_missing_prerequisite_refuses_before_execution(tmp_path):
    from tests.test_decoder06_capture import _spec
    from infinity_grid import portable_registry as project
    spec=_spec(tmp_path,project=False);spec['output_contract']=contract(prerequisites=[{'capsule_sha256':'0'*64,'completion_sha256':'1'*64}])
    store=tmp_path/'store';project.initialize(store,'PREREQUISITE_REFUSAL_FIXTURE',spec['engine_source'])
    capture=sub.capture(store,spec)
    with pytest.raises(sub.SubmissionError,match='PREREQUISITE_EVIDENCE_UNRESOLVED'):
        rc.prerequisites({'workspace':Path(capture['workspace'])})
    assert not (Path(capture['workspace'])/'runtime/attempts').exists()


def report(root,node,outcome,finished=True,binding='bound'):
    row={'binding':binding,'node':node,'finished':finished,'phases':{
        'setup':{'outcome':'passed'},'call':{'outcome':outcome},'teardown':{'outcome':'passed'}}}
    write_json_atomic(root/'nodes'/(canonical_sha256(node)+'.json'),row)


def test_mixed_worker_does_not_relabel_passed_node(tmp_path):
    selectors=['tests/fixture.py'];report(tmp_path,'tests/fixture.py::test_ok','passed');report(tmp_path,'tests/fixture.py::test_bad','failed')
    rows,status,errors=vr.reduce_reports(tmp_path,'bound',selectors,[{'return_code':1}])
    assert {r['node']:r['status'] for r in rows}=={'tests/fixture.py::test_ok':'PASS','tests/fixture.py::test_bad':'FAIL'}
    assert status=='FAIL' and errors==[]


def test_resume_selects_only_unfinished_nodes(tmp_path):
    a,b='tests/fixture.py::test_a','tests/fixture.py::test_b'
    report(tmp_path,a,'passed');report(tmp_path,b,'passed',finished=False)
    write_json_atomic(tmp_path/'collections/selection.json',{'binding':'bound','selectors':['tests/fixture.py'],'nodes':[a,b]})
    assert vr.pending_selectors(tmp_path,'bound',['tests/fixture.py'])==[b]
    with pytest.raises(RuntimeError,match='BINDING'):vr.pending_selectors(tmp_path,'different',['tests/fixture.py'])


def test_node_results_do_not_depend_on_worker_grouping(tmp_path):
    report(tmp_path,'tests/fixture.py::test_a','passed');report(tmp_path,'tests/fixture.py::test_b','failed')
    a=vr.reduce_reports(tmp_path,'bound',['tests/fixture.py'],[{'return_code':1}])
    b=vr.reduce_reports(tmp_path,'bound',['tests/fixture.py'],[{'return_code':0},{'return_code':1}])
    assert a==b


def test_checkpoint_copies_committed_sqlite_state_only(tmp_path):
    db=tmp_path/'tasks.sqlite';conn=sqlite3.connect(db);conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE tasks(id INTEGER)');conn.execute('INSERT INTO tasks VALUES(1)');conn.commit()
    conn.execute('INSERT INTO tasks VALUES(2)')
    raw=pr._stable_file(db);copy=tmp_path/'copy.sqlite';copy.write_bytes(raw)
    with sqlite3.connect(copy) as restored:assert restored.execute('SELECT id FROM tasks').fetchall()==[(1,)]
    conn.rollback();conn.close()


def queued_fixture(root, *, previous=None):
    obj=pr._put(root,b'unit fixture bytes: no source execution and no Drive acknowledgment')
    # Unit-only checkpoint fixture: all required 0.6.1 logical roles, one physical payload.
    # It is never a production capture or a genuine Drive acknowledgment.
    row={'schema_id':pr.SCHEMA,'previous':previous,'objects':[obj],'created_unix':time.time(),
         'capture_id':'UNIT_FIXTURE_CAPTURE','source':obj,'state':obj,
         'project':{'base':obj,'delta':obj},'base_objects':[],
         'reason':'DATA_ONLY_UNIT_FIXTURE'}
    raw=sub._json_bytes(row);entry=pr._put(root,raw);path=pr._root(root)/'commits'/(entry['sha256']+'.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    write_json_atomic(pr._root(root)/'CURRENT.json',{'sha256':entry['sha256']})
    return obj,entry


def test_outbox_deduplicates_objects_and_never_invents_ack(tmp_path):
    shared,a=queued_fixture(tmp_path);_,b=queued_fixture(tmp_path,previous=a['sha256'])
    status=pr.status(tmp_path)
    assert status['pending_bytes']==shared['size_bytes']+a['size_bytes']+b['size_bytes']
    assert {row['sha256'] for row in status['pending_objects']} == {shared['sha256'], a['sha256'], b['sha256']}
    assert len({row['obligation_id'] for row in status['pending_objects']}) == 10
    assert len(status['pending_checkpoints']) == 2
    assert status['status']=='SAVE_REQUIRED'


def test_wrong_outbox_readback_cannot_mark_saved(tmp_path):
    obj,_=queued_fixture(tmp_path);bad=tmp_path/'bad';bad.write_bytes(b'wrong')
    with pytest.raises(sub.SubmissionError,match='READBACK_MISMATCH'):
        pr.confirm(tmp_path,obj['sha256'],bad,'fixture_refusal_only',role='checkpoint_source')
    assert pr.status(tmp_path)['status']=='SAVE_REQUIRED'
    assert not list((pr._root(tmp_path)/'receipts').rglob('*.json'))


def test_missing_outbox_object_is_visible(tmp_path):
    obj,_=queued_fixture(tmp_path);(pr._root(tmp_path)/'objects'/(obj['sha256']+'.bin')).unlink()
    with pytest.raises(sub.SubmissionError,match='OUTBOX_OBJECT_MISSING'):pr.status(tmp_path)


def test_slim_restore_requires_exact_dependencies(tmp_path):
    obj={'sha256':hashlib.sha256(b'needed').hexdigest(),'size_bytes':6}
    row={'schema_id':pr.SCHEMA,'objects':[obj]}
    packet={'schema_id':'IG_DECODER_CHECKPOINT_EXPORT_V1','checkpoint_sha256':sub._sha(sub._json_bytes(row)),'checkpoint':row}
    raw=sub._archive({'CHECKPOINT.json':sub._json_bytes(packet)});archive=tmp_path/'slim.zip';archive.write_bytes(raw)
    with pytest.raises(sub.SubmissionError,match='RECOVERY_DEPENDENCIES_UNRESOLVED'):
        pr.restore_checkpoint(archive,tmp_path/'restored',sub._sha(raw))
    assert not (tmp_path/'restored').exists()


def test_interrupted_node_is_never_reported_as_pass(tmp_path):
    report(tmp_path,'tests/fixture.py::test_a','passed',finished=False)
    rows,status,_=vr.reduce_reports(tmp_path,'bound',['tests/fixture.py'],[{'return_code':124}])
    assert rows[0]['status']=='INTERRUPTED' and status=='FAIL'


def test_project_checkpoint_delta_reuses_baseline_and_reconstructs_exactly():
    base=sub._archive({'PROJECT.json':b'{}','objects/a':b'saved before execution'})
    full=sub._archive({'PROJECT.json':b'{}','objects/a':b'saved before execution','events/new.json':b'{}'})
    delta=pr._project_delta(base,full)
    assert set(pr._zip_files(delta))=={'events/new.json'}
    assert pr._restore_project(base,delta,sub._sha(full))==full
    changed=sub._archive({'PROJECT.json':b'{"changed":true}'})
    with pytest.raises(sub.SubmissionError,match='BASELINE_NOT_RETAINED'):pr._project_delta(base,changed)
    with pytest.raises(sub.SubmissionError,match='DELTA_OVERWRITE'):pr._restore_project(base,changed,sub._sha(full))


def test_stage_budget_counts_publication_and_all_phases(tmp_path):
    from infinity_grid.v05_stage_runtime import _stage_workspace_bytes, StageRuntimeError
    for name,raw in [('artifacts/answer.json',b'{}'),('phases/A/SUMMARY.json',b'123'),
                     ('phases/B/partition.sqlite3',b'12345')]:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    assert _stage_workspace_bytes(tmp_path)==10
    (tmp_path/'phases/B/unknown.bin').write_bytes(b'bad')
    with pytest.raises(StageRuntimeError,match='unexpected runtime workspace file'):
        _stage_workspace_bytes(tmp_path)


def test_stage_budget_counts_closed_replay_stores_and_rejects_unknown_entries(tmp_path):
    from infinity_grid.v05_stage_runtime import _stage_workspace_bytes, StageRuntimeError
    for name, raw in [
        ('replay_runner/runner_state.json', b'123'),
        ('replay_runner/checkpoints/a.json', b'1234'),
        ('replay_reference_data/MANIFEST.json', b'12345'),
        ('replay_reference_data/records/a.json', b'123456'),
        ('replay_reference_data/commits/a.json', b'1234567'),
        ('replay_reference_data/transactions/a.json', b'12345678'),
    ]:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    assert _stage_workspace_bytes(tmp_path) == 33
    (tmp_path / 'replay_runner/unknown.bin').write_bytes(b'bad')
    with pytest.raises(StageRuntimeError, match='unexpected replay workspace entry'):
        _stage_workspace_bytes(tmp_path)


def test_activation_packaging_retains_proof_without_duplicate_checkpoint_payloads(tmp_path):
    from infinity_grid.change_sessions import _activation_execution_files
    for name in ['runtime/runs/a/RESULT.json','durability/outbox/commits/checkpoint.json',
                 'durability/outbox/objects/payload.bin','durability/base_objects/input.bin',
                 'source/science.py','runtime/intake/artifacts/input.bin']:
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original bytes')
    files=_activation_execution_files(tmp_path)
    assert files['execution/runtime/runs/a/RESULT.json']==b'original bytes'
    assert files['execution/durability/outbox/commits/checkpoint.json']==b'original bytes'
    assert not any(name.endswith(('payload.bin','input.bin','science.py')) for name in files)
    assert json.loads(files['ACTIVATION_PACKAGE_SCOPE.json'])['standalone_recovery_archive'] is False
    assert (tmp_path/'durability/outbox/objects/payload.bin').read_bytes()==b'original bytes'


def test_physically_saved_capture_object_does_not_reconsume_backlog_reserve(tmp_path):
    obj,_=queued_fixture(tmp_path)
    before=pr.status(tmp_path)['pending_bytes']
    receipt={'schema_id':'IG_DECODER_EXPLICIT_SAVE_RECEIPT_V2','provider':'google_drive','sha256':obj['sha256'],
             'size_bytes':obj['size_bytes'],'drive_file_id':'fixture_readback_12345','raw_readback_verified':True}
    receipt['receipt_sha256']=canonical_sha256(receipt)
    path=tmp_path/'durability/receipts'/obj['sha256']/(receipt['receipt_sha256']+'.json');path.parent.mkdir(parents=True)
    write_json_atomic(path,receipt)
    status=pr.status(tmp_path)
    assert status['status']=='SAVE_REQUIRED' and status['pending_objects']
    assert status['pending_bytes']==before-obj['size_bytes']
