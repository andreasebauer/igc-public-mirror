"""Registered data-only fault tests; synthetic records grant no authority."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from infinity_grid import preservation as pr, result_contracts as rc, submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.canon import write_json_atomic

ROOT=Path(__file__).parents[1]


def old_contracts():
    path=ROOT/'tests/fixtures/dev81_publication_reproduction/result_contracts.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='0fd5396e23f0d463914a4756fa72eaf865ed315f0886aa462495ab9ba96e3f65'
    spec=importlib.util.spec_from_file_location('infinity_grid._dev81_result_contracts',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def artifact_contract(expected='ORIGINAL'):
    return rc.normalize({'schema_id':rc.SCIENCE_SCHEMA,'claim':'SCIENCE','scientific_content':['answer.json'],
        'required_artifacts':[{'path':'answer.json','json_checks':[{'pointer':'/outcome','equals':expected}]}],
        'outcome':{'artifact':'answer.json','pointer':'/outcome'},'result_checks':[],
        'prerequisites':[],'preservation':{}},{'kind':'SCRIPT'},{'outcomes':['ORIGINAL','CHANGED']})


def mutate_after_read(monkeypatch,path):
    original=Path.read_bytes;calls=[]
    def read(target):
        raw=original(target)
        if target==path:
            calls.append(raw)
            target.write_text('{"outcome":"CHANGED"}')
        return raw
    monkeypatch.setattr(Path,'read_bytes',read)
    return calls


def test_dev81_reproduces_hash_and_json_from_different_bytes(tmp_path,monkeypatch):
    path=tmp_path/'answer.json';path.write_text('{"outcome":"ORIGINAL"}')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    mutate_after_read(monkeypatch,path)
    result=old_contracts().verify(artifact_contract('CHANGED'),{},tmp_path)
    assert result['status']=='VERIFIED' and result['scientific_outcome']=='CHANGED'
    assert result['artifacts'][0]['sha256']==digest


def test_hash_json_checks_and_outcome_use_one_byte_snapshot(tmp_path,monkeypatch):
    path=tmp_path/'answer.json';path.write_text('{"outcome":"ORIGINAL"}')
    raw=path.read_bytes();calls=mutate_after_read(monkeypatch,path)
    result=rc.verify(artifact_contract(),{},tmp_path)
    assert result['status']=='VERIFIED' and result['scientific_outcome']=='ORIGINAL'
    assert result['artifacts'][0]['sha256']==hashlib.sha256(raw).hexdigest()
    assert calls==[raw]
    with pytest.raises(loop.ControllerLoopError,match='RESULT_ARTIFACT_CHANGED_AFTER_VERIFICATION'):
        loop._verified_artifact_evidence(tmp_path,result)


def test_outcome_only_artifact_is_also_read_once(tmp_path,monkeypatch):
    path=tmp_path/'answer.json';path.write_text('{"outcome":"ORIGINAL"}')
    c=artifact_contract();c['required_artifacts'][0].pop('json_checks');c['required_artifacts'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    calls=mutate_after_read(monkeypatch,path)
    result=rc.verify(c,{},tmp_path)
    assert result['scientific_outcome']=='ORIGINAL' and len(calls)==1


@pytest.mark.parametrize('raw',[b'not json',b'\xff'],ids=['malformed','invalid_utf8'])
def test_invalid_artifact_bytes_cannot_supply_outcome(tmp_path,raw):
    (tmp_path/'answer.json').write_bytes(raw)
    result=rc.verify(artifact_contract(),{},tmp_path)
    assert result['status']=='REJECTED' and result['scientific_outcome'] is None


def test_unchanged_artifact_evidence_binds_and_mutation_refuses(tmp_path):
    path=tmp_path/'answer.json';path.write_text('{"outcome":"ORIGINAL"}')
    result=rc.verify(artifact_contract(),{},tmp_path)
    assert loop._verified_artifact_evidence(tmp_path,result)==result['artifacts']
    path.write_text('{"outcome":"CHANGED"}')
    with pytest.raises(loop.ControllerLoopError,match='RESULT_ARTIFACT_CHANGED_AFTER_VERIFICATION'):
        loop._verified_artifact_evidence(tmp_path,result)


def test_exact_dev81_source_publishes_claim_before_terminal_checkpoint():
    path=ROOT/'tests/fixtures/dev81_publication_reproduction/v05_controller_event_loop.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='ed0bc2348f57d8d8169562acfa68951d8a89c3787ba9ba142383d220e2bcf82d'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_run_workspace_job')
    checkpoints=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Name) and n.func.id=='make_checkpoint'
        and len(n.args)>1 and isinstance(n.args[1],ast.Constant) and n.args[1].value=='TERMINAL_RESULT']
    published=[n.lineno for n in ast.walk(fn) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name)
        and n.func.value.id=='claim' and n.func.attr=='complete']
    assert max(published)<checkpoints[0]


def unit_done():
    return {'request_id':'UNIT_ONLY','source_sha256':'1'*64,'completion_sha256':'2'*64,
        'status':'COMPLETED','publication_protocol':pr.COMPLETION_PROTOCOL}


def unit_packet(done):
    row={'schema_id':pr.SCHEMA,'terminal':True,'capture_id':'UNIT_CAPTURE',
         'job_id':'UNIT.JOB','source_sha256':done['source_sha256'],
         'terminal_completions':{done['request_id']:done['completion_sha256']}}
    return {'checkpoint_sha256':sub._sha(sub._json_bytes(row)),'checkpoint':row}


def bind_unit_capture(monkeypatch):
    monkeypatch.setattr(sub,'capture_record',lambda root:{'capture_id':'UNIT_CAPTURE','job':{'job_id':'UNIT.JOB'}})


def test_checkpoint_failure_never_calls_claim_complete(tmp_path,monkeypatch):
    calls=[]
    prepared=tmp_path/'runtime/intake/prepared_completions/UNIT_ONLY.json'
    write_json_atomic(prepared,unit_done())
    original=prepared.read_bytes()
    def fail(root,done):
        calls.append('checkpoint');raise RuntimeError('CHECKPOINT_FAILURE')
    monkeypatch.setattr(pr,'ensure_terminal_completion',fail)
    claim=SimpleNamespace(complete=lambda done:calls.append('published'))
    with pytest.raises(RuntimeError,match='CHECKPOINT_FAILURE'):
        loop._publish_checkpointed_completion({'workspace':tmp_path},unit_done(),claim)
    assert calls==['checkpoint']
    assert prepared.read_bytes()==original
    assert not (tmp_path/'runtime/intake/completed/UNIT_ONLY.json').exists()


def test_publication_occurs_only_after_checkpoint_success(tmp_path,monkeypatch):
    calls=[]
    def checkpoint(root,done):calls.append('checkpoint');return {'status':'UNIT_TEST_ONLY'}
    monkeypatch.setattr(pr,'ensure_terminal_completion',checkpoint)
    claim=SimpleNamespace(complete=lambda done:calls.append('published'))
    result=loop._publish_checkpointed_completion({'workspace':tmp_path},unit_done(),claim)
    assert calls==['checkpoint','published'] and result['status']=='UNIT_TEST_ONLY'
    assert json.loads((tmp_path/'runtime/intake/completed/UNIT_ONLY.json').read_text())==unit_done()


def test_prepared_checkpoint_evidence_is_checked_and_divergence_refuses():
    evidence={'path':'answer.json','sha256':sub._sha(b'{}'),'size_bytes':2}
    done={'evidence':[evidence],'completion_sha256':'2'*64}
    raw=sub._json_bytes(done)
    files={'runtime/runs/unit/answer.json':b'{}',
           'runtime/intake/prepared_completions/unit.json':raw}
    pr._verify_terminal_state_evidence(files)
    files['runtime/runs/unit/answer.json']=b'changed'
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):
        pr._verify_terminal_state_evidence(files)
    files['runtime/runs/unit/answer.json']=b'{}'
    files['runtime/intake/completed/unit.json']=sub._json_bytes({**done,'other':'changed'})
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_PREPARED_COMPLETION_MISMATCH'):
        pr._verify_terminal_state_evidence(files)


def test_missing_checkpoint_proof_stays_unpublished(tmp_path):
    assert pr.terminal_completion_proof(tmp_path,unit_done()) is False


@pytest.mark.parametrize('fault',['hash','completion','source','capture','terminal'])
def test_invalid_checkpoint_proof_refuses(tmp_path,monkeypatch,fault):
    bind_unit_capture(monkeypatch);done=unit_done();packet=unit_packet(done)
    if fault=='hash':packet['checkpoint_sha256']='0'*64
    else:
        row=packet['checkpoint']
        if fault=='completion':row['terminal_completions'][done['request_id']]='0'*64
        elif fault=='source':row['source_sha256']='0'*64
        elif fault=='capture':row['capture_id']='OTHER'
        else:row['terminal']=False
        packet['checkpoint_sha256']=sub._sha(sub._json_bytes(row))
    write_json_atomic(tmp_path/'runtime/terminal_checkpoints/UNIT_ONLY.json',packet)
    with pytest.raises(sub.SubmissionError,match='TERMINAL_COMPLETION_CHECKPOINT_MISMATCH'):
        pr.terminal_completion_proof(tmp_path,done)


def test_terminal_restore_proof_is_portable(tmp_path,monkeypatch):
    bind_unit_capture(monkeypatch);done=unit_done()
    write_json_atomic(tmp_path/'durability/RESTORED_CHECKPOINT_PROVENANCE.json',unit_packet(done))
    assert pr.terminal_completion_proof(tmp_path,done)


def test_paused_restore_remains_pending(tmp_path,monkeypatch):
    bind_unit_capture(monkeypatch);packet=unit_packet(unit_done());packet['checkpoint']['terminal']=False
    packet['checkpoint_sha256']=sub._sha(sub._json_bytes(packet['checkpoint']))
    write_json_atomic(tmp_path/'durability/RESTORED_CHECKPOINT_PROVENANCE.json',packet)
    assert pr.terminal_completion_proof(tmp_path,unit_done()) is False


def test_historical_completion_rules_remain_and_unknown_protocol_refuses(tmp_path):
    assert pr.terminal_completion_proof(tmp_path,{'status':'HISTORICAL_FIXTURE'})
    with pytest.raises(sub.SubmissionError,match='UNKNOWN_COMPLETION_PUBLICATION_PROTOCOL'):
        pr.terminal_completion_proof(tmp_path,{'publication_protocol':'UNKNOWN'})
