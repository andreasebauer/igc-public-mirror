"""Repair-7R: real SQLite lifecycle checks under native recorded-change validation.
No save receipts are invented. TEST_ONLY_EVIDENCE_FIXTURE dictionaries exercise
one pure snapshot consistency gate; they cannot be used as native completions.
Tracking references deliberately prevent GC from hiding missing close() calls.
"""
from contextlib import closing
from pathlib import Path
import hashlib,json,sqlite3,subprocess,sys
import pytest
from infinity_grid import preservation as pr
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.v05_stage_runtime import StageScienceRuntime


def emit(case, **kw):
    print('IG_BENCHMARK_JSON:'+json.dumps(dict(case=case,**kw),sort_keys=True))


def make_db(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    c=sqlite3.connect(path)
    assert c.execute('PRAGMA journal_mode=WAL').fetchone()[0]=='wal'
    c.execute('CREATE TABLE tasks(id INTEGER PRIMARY KEY, payload TEXT)')
    c.execute("INSERT INTO tasks VALUES(1,'first')");c.commit()
    return c


def track(monkeypatch, fail=None):
    real=sqlite3.connect; handles=[]
    class Tracked(sqlite3.Connection):
        was_closed=False
        def close(self):
            self.was_closed=True
            return super().close()
        def execute(self, sql, *args, **kw):
            if fail=='quick_check' and sql=='PRAGMA quick_check':
                class Bad:
                    def fetchone(self):return ('injected invalid check',)
                return Bad()
            return super().execute(sql,*args,**kw)
        def backup(self, target, **kw):
            if fail=='backup':raise sqlite3.OperationalError('injected backup failure')
            return super().backup(target,**kw)
    def connect(*args,**kw):
        if fail=='destination_open' and len(handles)==1:
            raise sqlite3.OperationalError('injected destination open failure')
        kw['factory']=Tracked
        c=real(*args,**kw);handles.append(c);return c
    monkeypatch.setattr(pr.sqlite3,'connect',connect)
    return handles,real


def clean(handles):
    for c in handles:
        if not c.was_closed:c.close()


def test_backup_closes_both_handles_and_preserves_committed_rows(tmp_path,monkeypatch):
    db=tmp_path/'tasks.sqlite3';writer=make_db(db)
    writer.execute("INSERT INTO tasks VALUES(2,'uncommitted')")
    hs,real=track(monkeypatch)
    try:
        raw=pr._stable_file(db)
        flags=[c.was_closed for c in hs]
        copy=tmp_path/'saved.sqlite3';copy.write_bytes(raw)
        with closing(real(copy.as_uri()+'?immutable=1',uri=True)) as c:
            rows=c.execute('SELECT * FROM tasks').fetchall()
            check=c.execute('PRAGMA quick_check').fetchone()[0]
        emit('backup_success',closed=flags,rows=rows,integrity=check)
        assert rows==[(1,'first')] and check=='ok'
        assert flags==[True,True], 'source and destination must be explicitly closed'
    finally:writer.rollback();writer.close();clean(hs)


@pytest.mark.parametrize('fail',['quick_check','backup','destination_open'])
def test_backup_closes_handles_on_failure(tmp_path,monkeypatch,fail):
    writer=make_db(tmp_path/'tasks.sqlite3');hs,real=track(monkeypatch,fail)
    try:
        with pytest.raises((sqlite3.OperationalError,pr.sub.SubmissionError)):
            pr._stable_file(tmp_path/'tasks.sqlite3')
        flags=[c.was_closed for c in hs]
        emit('backup_failure_'+fail,closed=flags)
        assert flags and all(flags), 'every opened handle must close even on failure'
    finally:writer.close();clean(hs)


def test_checkpoint_reader_cannot_keep_terminal_wal_alive(tmp_path,monkeypatch):
    db=tmp_path/'partition.sqlite3';writer=make_db(db);hs,real=track(monkeypatch)
    try:
        pr._stable_file(db)
        writer.execute("INSERT INTO tasks VALUES(2,'after checkpoint')");writer.commit();writer.close()
        journals=[p.name for p in tmp_path.iterdir() if p.name.endswith(('-wal','-shm'))]
        emit('terminal_after_backup',journals=journals,handles_closed=[c.was_closed for c in hs])
        assert journals==[], 'checkpoint-owned reader leaked into terminal evidence'
    finally:writer.close();clean(hs)


def phase_db(tmp_path):
    root=tmp_path/'runtime/runs/intent-test-only'
    return root,root/'chain/decoder_stage_runtime/R7R/phases/P/partition.sqlite3'


def test_completion_gate_refuses_open_engine_database_without_deleting_files(tmp_path):
    root,db=phase_db(tmp_path);writer=make_db(db)
    before={p.name:p.read_bytes() for p in db.parent.iterdir()}
    try:
        with pytest.raises(pr.sub.SubmissionError,match='TASK_DATABASE_NOT_QUIESCENT'):
            pr.require_quiescent_task_databases(root)
        assert before=={p.name:p.read_bytes() for p in db.parent.iterdir()}
    finally:writer.close()
    pr.require_quiescent_task_databases(root)
    emit('engine_database_quiescence',open_refused=True,closed_accepted=True,no_deletion=True)


def test_idle_partition_finalizer_uses_sqlite_to_retire_stale_wal(tmp_path):
    root,db=phase_db(tmp_path);db.parent.mkdir(parents=True)
    code=("import os,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
          "c.execute('PRAGMA journal_mode=WAL'); "
          "c.execute('CREATE TABLE items (value TEXT)'); "
          "c.execute(\"INSERT INTO items VALUES ('committed')\"); c.commit(); os._exit(0)")
    subprocess.run([sys.executable,'-c',code,str(db)],check=True)
    assert Path(str(db)+'-wal').exists()
    with pytest.raises(pr.sub.SubmissionError,match='TASK_DATABASE_NOT_QUIESCENT'):
        pr.require_quiescent_task_databases(root)
    StageScienceRuntime._finalize_idle_partition_database(db)
    pr.require_quiescent_task_databases(root)
    with closing(sqlite3.connect(db)) as reader:
        assert reader.execute('SELECT value FROM items').fetchone()==('committed',)


def test_completion_gate_does_not_rewrite_script_artifacts(tmp_path):
    root=tmp_path/'output';p=root/'script/work/notes-wal';p.parent.mkdir(parents=True);p.write_bytes(b'not a database')
    pr.require_quiescent_task_databases(root)
    assert p.read_bytes()==b'not a database'


def fixture_completion(root, evidence):
    p=root/'runtime/intake/completed/fixture.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'schema_id':'TEST_ONLY_EVIDENCE_FIXTURE','evidence':evidence}))


def test_terminal_snapshot_matches_exact_completed_evidence(tmp_path):
    out=tmp_path/'runtime/runs/intent-test-only';out.mkdir(parents=True)
    (out/'answer.txt').write_bytes(b'unchanged')
    fixture_completion(tmp_path,loop._evidence_rows(out))
    files=pr._state_files(tmp_path)
    assert files['runtime/runs/intent-test-only/answer.txt']==b'unchanged'


def test_terminal_snapshot_refuses_dropped_required_evidence(tmp_path):
    out=tmp_path/'runtime/runs/intent-test-only';out.mkdir(parents=True)
    (out/'answer.txt').write_bytes(b'unchanged')
    (out/'expected-wal').write_bytes(b'required bytes must never silently disappear')
    fixture_completion(tmp_path,loop._evidence_rows(out))
    with pytest.raises(pr.sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):
        pr._state_files(tmp_path)


def test_terminal_snapshot_refuses_changed_completed_evidence(tmp_path):
    out=tmp_path/'runtime/runs/intent-test-only';out.mkdir(parents=True)
    (out/'answer.txt').write_bytes(b'original')
    fixture_completion(tmp_path,loop._evidence_rows(out))
    (out/'answer.txt').write_bytes(b'changed')
    with pytest.raises(pr.sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):
        pr._state_files(tmp_path)


def test_completion_gate_is_before_native_completion_seal():
    import inspect
    src=inspect.getsource(loop._run_workspace_job)
    # Dev82 seals through the byte-binding helper after database quiescence.
    assert src.index('require_quiescent_task_databases(out)')<src.index("'evidence':_verified_artifact_evidence(sealed,verification)")


def test_plain_snapshot_bytes_are_unchanged(tmp_path):
    p=tmp_path/'plain.txt';p.write_bytes(b'raw stable data\n')
    assert pr._stable_file(p)==p.read_bytes()


def test_completed_generation_image_survives_readonly_reopen_after_midrun_backup(tmp_path):
    # The observed failure had a 32-row WAL view and a 49-row sealed main DB.
    # Exercise both sides of that checkpoint boundary with real SQLite.
    db=tmp_path/'state_store.sqlite3';writer=make_db(db)
    for i in range(2,33):writer.execute('INSERT INTO tasks VALUES(?,?)',(i,'x'*2048))
    writer.commit();snapshot=pr._stable_file(db)
    for i in range(33,50):writer.execute('INSERT INTO tasks VALUES(?,?)',(i,'y'*2048))
    writer.commit();writer.close()
    StageScienceRuntime._finalize_idle_partition_database(db)
    before=db.read_bytes()
    assert before[18:20]==bytes([1,1]), 'sealed image must not retain WAL mode'
    for _ in range(3):
        with closing(sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)) as c:
            assert c.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]==49
            assert c.execute('PRAGMA journal_mode').fetchone()[0]=='delete'
        assert db.read_bytes()==before
        assert not Path(str(db)+'-wal').exists()
        assert not Path(str(db)+'-shm').exists()
    saved=tmp_path/'midrun.sqlite3';saved.write_bytes(snapshot)
    with closing(sqlite3.connect(saved.as_uri()+'?immutable=1',uri=True)) as c:
        assert c.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]==32


def test_finalizer_refuses_active_reader_without_discarding_committed_rows(tmp_path):
    db=tmp_path/'state_store.sqlite3';writer=make_db(db)
    reader=sqlite3.connect(db);reader.execute('BEGIN');reader.execute('SELECT * FROM tasks').fetchall()
    writer.execute("INSERT INTO tasks VALUES(2,'later')");writer.commit();writer.close()
    try:
        with pytest.raises(sqlite3.OperationalError):
            StageScienceRuntime._finalize_idle_partition_database(db)
        assert Path(str(db)+'-wal').exists()
    finally:reader.close()
    StageScienceRuntime._finalize_idle_partition_database(db)
    with closing(sqlite3.connect(db)) as c:
        assert c.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]==2
