"""Bounded registered engineering checks. No historical producer is executed."""
from pathlib import Path
import hashlib
import io
import json
import os
import sqlite3
import threading
import time
import zipfile

import pytest
from infinity_grid import evidence_diagnostics as ed, invocation, preservation as pr, submission as sub
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.canon import canonical_sha256, write_json_atomic
from infinity_grid.v05_passive_intake import request_claim_binding


def row(path, raw):
    return {'path':path,'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def fixture(tmp_path):
    # Deliberately synthetic admission, never submitted as execution authority.
    req={'request_id':'UNIT_ONLY','kind':'UNIT_ONLY'}
    a={'workspace':tmp_path,'source_sha256':'1'*64,'job':{'registration_sha256':'2'*64},'request':req}
    ident=request_claim_binding(req,'2'*64,'1'*64,'1'*64)
    out=tmp_path/'runtime/runs'/('intent-'+canonical_sha256(ident)[:32]);out.mkdir(parents=True)
    (out/'answer.json').write_bytes(b'original')
    done={'schema_id':'IG_DECODER_WORKSPACE_COMPLETION_V1','source_sha256':'1'*64,
          'registration_sha256':'2'*64,'request_sha256':ident['request_sha256'],
          'result':{},'result_sha256':canonical_sha256({}),'evidence':[row('answer.json',b'original')]}
    done['completion_sha256']=canonical_sha256(done)
    path=tmp_path/'runtime/intake/completed/UNIT_ONLY.json';write_json_atomic(path,done)
    return a,out,path,done


def tree(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_matching_completion_remains_read_only(tmp_path):
    a,out,p,done=fixture(tmp_path);before=tree(tmp_path)
    assert loop.verified_completion(a)==done
    assert tree(tmp_path)==before


def test_live_mismatch_inventory_is_single_read_and_read_only(tmp_path,monkeypatch):
    a,out,p,done=fixture(tmp_path);(out/'answer.json').write_bytes(b'changed');before=tree(tmp_path)
    original=loop._evidence_rows;calls=[]
    def read(root):calls.append(root);return original(root)
    monkeypatch.setattr(loop,'_evidence_rows',read)
    with pytest.raises(loop.ControllerLoopError,match='COMPLETION_EVIDENCE_MISMATCH') as caught:
        loop.verified_completion(a)
    d=caught.value.evidence_diagnostic;delta=next(iter(d['differences'].values()))
    assert len(calls)==1 and delta['changed']==[{'path':'answer.json','expected':done['evidence'][0],'observed':row('answer.json',b'changed')}]
    assert tree(tmp_path)==before and d['admitted_binding']['source_sha256']=='1'*64
    assert d['writer_attribution']=='UNKNOWN' and d['root_cause_established'] is False
    assert d['diagnostic_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='diagnostic_sha256'})


def test_missing_and_synthetic_sidecar_are_distinct(tmp_path):
    a,out,p,done=fixture(tmp_path);(out/'answer.json').unlink();(out/'answer.json-wal').write_bytes(b'SYNTHETIC')
    with pytest.raises(loop.ControllerLoopError) as caught:loop.verified_completion(a)
    delta=next(iter(caught.value.evidence_diagnostic['differences'].values()))
    assert delta['missing']==done['evidence'] and delta['unexpected']==[row('answer.json-wal',b'SYNTHETIC')] and not delta['changed']


def test_public_refusal_retains_same_diagnostic_outside_evidence(tmp_path,monkeypatch):
    a,out,p,done=fixture(tmp_path);(out/'answer.json').write_bytes(b'changed');before=tree(tmp_path)
    monkeypatch.setattr(loop,'_run_workspace_job',lambda *args:loop.verified_completion(a))
    with pytest.raises(loop.ControllerLoopError) as caught:loop.run_workspace_job(tmp_path,'UNIT_ONLY')
    exc=caught.value;assert exc.refusal['evidence_diagnostic']==exc.evidence_diagnostic
    paths=list((tmp_path/'runtime/refusals').glob('*.json'));assert len(paths)==1
    assert json.loads(paths[0].read_text())==exc.refusal
    assert {k:v for k,v in tree(tmp_path).items() if not k.startswith('runtime/refusals/')}==before


def test_metadata_refusal_does_not_claim_inventory_read(tmp_path,monkeypatch):
    a,out,p,done=fixture(tmp_path);done['source_sha256']='0'*64;write_json_atomic(p,done)
    def forbidden(*args):raise AssertionError('metadata-first order changed')
    monkeypatch.setattr(loop,'_evidence_rows',forbidden)
    with pytest.raises(loop.ControllerLoopError) as caught:loop.verified_completion(a)
    d=caught.value.evidence_diagnostic
    assert d['failed_checks']==['source_sha256'] and d['observed_roots']=={}
    assert d['observation_kind']=='NOT_READ_METADATA_REFUSAL'


def test_diagnostic_freezes_input_and_handles_duplicate_paths():
    rows=[row('x',b'1'),row('x',b'2')]
    e=ed.attach(RuntimeError('UNIT'),phase='UNIT',done={'evidence':rows},completion_path='UNIT',observed_roots={'ROOT':[]},failed_checks=['UNIT'],observation='SYNTHETIC')
    before=json.dumps(e.evidence_diagnostic,sort_keys=True);rows.clear()
    assert e.evidence_diagnostic['differences']['ROOT']=={'classification':'UNINDEXABLE_INVENTORY'}
    assert json.dumps(e.evidence_diagnostic,sort_keys=True)==before


def test_checkpoint_legacy_does_not_invent_unique_root():
    files={'runtime/runs/a/answer':b'A','runtime/runs/b/answer':b'B',
           'runtime/intake/completed/unit.json':sub._json_bytes({'evidence':[row('answer',b'C')]})}
    with pytest.raises(sub.SubmissionError) as caught:pr._verify_terminal_state_evidence(files)
    d=caught.value.evidence_diagnostic
    assert set(d['observed_roots'])=={'runtime/runs/a','runtime/runs/b'}
    assert d['observation_kind']=='EXACT_IN_MEMORY_CHECKPOINT_BYTES'


def test_exact_partition_incident_reports_retained_mismatch():
    base=Path(__file__).parent/'fixtures/a12_diagnostics'
    raw=(base/'partition_state.zip').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((base/'manifest.json').read_text())['partition_state.zip']
    with zipfile.ZipFile(io.BytesIO(raw)) as z:files={n:z.read(n) for n in z.namelist()}
    before={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH') as caught:pr._verify_terminal_state_evidence(files)
    d=caught.value.evidence_diagnostic;delta=next(iter(d['differences'].values()))
    assert len(delta['changed'])==1 and not delta['missing'] and not delta['unexpected']
    diff=delta['changed'][0];assert diff['path']=='script/work/REFINEMENT_PROGRESS.json'
    assert diff['expected']['size_bytes']==551 and diff['observed']['size_bytes']==422
    assert diff['expected']['sha256']=='1e82eb82a176663af778f2d8f781f2d44b3696d54a8ce6a0e28e543a6601a00d'
    assert diff['observed']['sha256']=='1ee428177083b193554f3a57c16f8813c543d0fe5431942257792a6281b3fef8'
    assert before=={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    print('A12_PARTITION_DIAGNOSTIC='+json.dumps(d,sort_keys=True))


def test_real_checkpoint_overlaps_held_writer_transaction(tmp_path):
    # Real checkpoint API on a captured DATA fixture, inside this saved validation.
    # No inner job is dispatched. Not a production periodic controller experiment.
    from tests.test_decoder06_capture import _spec
    from infinity_grid import portable_registry
    spec=_spec(tmp_path,project=False);store=tmp_path/'fixture_store'
    portable_registry.initialize(store,'A12 CHECKPOINT DATA FIXTURE',spec['engine_source'])
    captured=sub.capture(store,spec);w=Path(captured['workspace'])
    db=w/'runtime/tasks.sqlite3';db.parent.mkdir(exist_ok=True)
    with sqlite3.connect(db) as c:
        assert c.execute('PRAGMA journal_mode=WAL').fetchone()[0]=='wal'
        c.execute('CREATE TABLE tasks(id INTEGER PRIMARY KEY)');c.execute('INSERT INTO tasks VALUES(1)');c.commit()
    evidence=[]
    for pending in (2,3):
        ready=threading.Event();release=threading.Event();errors=[];marks={}
        def writer():
            try:
                with sqlite3.connect(db) as c:
                    c.execute('INSERT INTO tasks VALUES(?)',(pending,));marks['write_transaction_open_ns']=time.monotonic_ns();ready.set()
                    if not release.wait(45):raise RuntimeError('checkpoint handshake timeout')
                    c.commit();marks['write_transaction_commit_ns']=time.monotonic_ns()
            except BaseException as exc:errors.append(repr(exc));ready.set()
        thread=threading.Thread(target=writer);thread.start()
        try:
            assert ready.wait(10) and not errors
            marks['checkpoint_begin_ns']=time.monotonic_ns()
            result=pr.make_checkpoint(w,'A12_HELD_WRITER_TRANSACTION_'+str(pending))
            marks['checkpoint_end_ns']=time.monotonic_ns()
            current=json.loads((pr._root(w)/'CURRENT.json').read_text())['sha256']
            packet=json.loads((pr._root(w)/'commits'/(current+'.json')).read_text())
            state=pr._zip_files((pr._root(w)/'objects'/(packet['state']['sha256']+'.bin')).read_bytes())
            copy=tmp_path/('snapshot'+str(pending)+'.sqlite3');copy.write_bytes(state['runtime/tasks.sqlite3'])
            with sqlite3.connect(copy) as c:
                assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
                observed=[x[0] for x in c.execute('SELECT id FROM tasks ORDER BY id')]
            assert observed==list(range(1,pending))
            assert not any(n.endswith(('-wal','-shm')) for n in state)
        finally:
            release.set();thread.join(10)
        assert not thread.is_alive() and not errors
        assert marks['write_transaction_open_ns']<marks['checkpoint_begin_ns']<marks['checkpoint_end_ns']<marks['write_transaction_commit_ns']
        with sqlite3.connect(db) as c:assert [x[0] for x in c.execute('SELECT id FROM tasks ORDER BY id')]==list(range(1,pending+1))
        evidence.append({'checkpoint_sha256':current,'state_sha256':packet['state']['sha256'],'snapshot_rows':observed,'pending_id_excluded':pending,'events':marks})
    with sqlite3.connect(':memory:') as c:runtime={'sqlite_version':sqlite3.sqlite_version,'sqlite_source_id':c.execute('SELECT sqlite_source_id()').fetchone()[0]}
    print('A12_WRITER_CHECKPOINT_OVERLAP='+json.dumps({'scope':'REGISTERED_VALIDATION_REAL_CHECKPOINT_API_HELD_WRITE_TRANSACTION','runtime':runtime,'observations':evidence,'historical_fault_reproduced':False,'production_periodic_checkpoint_exercised':False},sort_keys=True))
