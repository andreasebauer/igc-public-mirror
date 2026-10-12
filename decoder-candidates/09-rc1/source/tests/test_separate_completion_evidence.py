"""Native registered engineering cases for the mutable/sealed boundary."""
from contextlib import closing
import json
import sqlite3
import pytest
from infinity_grid import completion_evidence as ce, preservation as pr
from infinity_grid import v05_controller_event_loop as loop


def fixture(tmp_path):
    work=tmp_path/'runtime/runs/intent-fixture';work.mkdir(parents=True)
    (work/'answer.json').write_text('{"answer":42}')
    db=work/'chain/decoder_stage_runtime/X/phases/P/state_store.sqlite3'
    db.parent.mkdir(parents=True)
    with closing(sqlite3.connect(db)) as c:
        c.execute('CREATE TABLE items(id INTEGER PRIMARY KEY, value TEXT)')
        c.executemany('INSERT INTO items VALUES(?,?)',[(1,'first'),(2,'second')]);c.commit()
    sealed=tmp_path/'runtime/sealed/intent-fixture'
    ce.stage_snapshot(work,sealed)
    done={'schema_id':'TEST_ONLY_EVIDENCE_FIXTURE','evidence_protocol':ce.PROTOCOL,
          'evidence_root':'runtime/sealed/intent-fixture','evidence':loop._evidence_rows(sealed)}
    p=tmp_path/'runtime/intake/prepared_completions/fixture.json';p.parent.mkdir(parents=True);p.write_text(json.dumps(done))
    return work,sealed,db,done


def test_working_wal_changes_do_not_change_sealed_snapshot(tmp_path):
    work,sealed,db,done=fixture(tmp_path)
    writer=sqlite3.connect(db)
    try:
        writer.execute('PRAGMA journal_mode=WAL')
        writer.execute("INSERT INTO items VALUES(3,'later')");writer.commit()
        assert db.with_name(db.name+'-wal').exists()
        assert loop._evidence_rows(sealed)==done['evidence']
        files=pr._state_files(tmp_path)
        pr._verify_terminal_state_evidence(files)
        sealed_db=sealed/db.relative_to(work)
        with closing(sqlite3.connect(sealed_db.as_uri()+'?immutable=1',uri=True)) as c:
            assert c.execute('SELECT * FROM items ORDER BY id').fetchall()==[(1,'first'),(2,'second')]
        assert sealed_db.read_bytes()[18:20]==bytes([1,1])
    finally:writer.close()


@pytest.mark.parametrize('mutation',['change','delete','extra','extra-wal'])
def test_any_sealed_tree_mutation_is_rejected_by_checkpoint(tmp_path,mutation):
    work,sealed,db,done=fixture(tmp_path)
    p=sealed/'answer.json'
    if mutation=='change':p.write_text('changed')
    elif mutation=='delete':p.unlink()
    else:(sealed/('unexpected-wal' if mutation=='extra-wal' else 'unexpected')).write_text('new')
    with pytest.raises(pr.sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):
        pr._state_files(tmp_path)


@pytest.mark.parametrize('location',['../runs/intent-fixture','runtime/sealed/other','runtime/runs/intent-fixture','/tmp/evidence'])
def test_completion_location_cannot_be_redirected(tmp_path,location):
    work,sealed,db,done=fixture(tmp_path);done['evidence_root']=location
    with pytest.raises(loop.ControllerLoopError,match='COMPLETION_EVIDENCE_LOCATION'):
        ce.evidence_root(tmp_path,work,done)


def test_legacy_evidence_still_binds_working_tree(tmp_path):
    work,sealed,db,done=fixture(tmp_path)
    legacy={'schema_id':'TEST_ONLY_EVIDENCE_FIXTURE','evidence':loop._evidence_rows(work)}
    p=tmp_path/'runtime/intake/prepared_completions/fixture.json';p.write_text(json.dumps(legacy))
    assert ce.evidence_root(tmp_path,work,legacy)==work
    (work/'answer.json').write_text('changed')
    with pytest.raises(pr.sub.SubmissionError,match='CHECKPOINT_COMPLETION_EVIDENCE_MISMATCH'):
        pr._state_files(tmp_path)


def test_open_writer_refused_and_its_committed_data_preserved(tmp_path):
    work,sealed,db,done=fixture(tmp_path);writer=sqlite3.connect(db)
    try:
        writer.execute('PRAGMA journal_mode=WAL');writer.execute("INSERT INTO items VALUES(3,'committed')");writer.commit()
        with pytest.raises(pr.sub.SubmissionError,match='TASK_DATABASE_NOT_QUIESCENT'):
            ce.stage_snapshot(work,tmp_path/'refused')
        assert writer.execute('SELECT COUNT(*) FROM items').fetchone()[0]==3
        assert db.with_name(db.name+'-wal').exists()
    finally:writer.close()


def test_symlink_artifact_refused(tmp_path):
    work,sealed,db,done=fixture(tmp_path)
    (work/'alias').symlink_to(work/'answer.json')
    with pytest.raises(loop.ControllerLoopError,match='EVIDENCE_SYMLINK'):
        ce.stage_snapshot(work,tmp_path/'refused')
