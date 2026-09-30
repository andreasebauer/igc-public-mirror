"""Four registered forensic experiments. Only disposable fixtures are opened."""
from contextlib import closing
from pathlib import Path
import hashlib
import json
import os
import sqlite3
import time
import zipfile
from infinity_grid import preservation as pr
from infinity_grid.v05_stage_runtime import StageScienceRuntime


def inventory(db):
    result={}
    for suffix in ('','-wal','-shm','-journal'):
        p=Path(str(db)+suffix)
        if p.exists():
            b=p.read_bytes();result[suffix]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'header18_19':list(b[18:20]) if suffix=='' else None}
    return result


def observe(events,label,db):
    events.append({'event':label,'unix_ns':time.time_ns(),'observer_pid':os.getpid(),'files':inventory(db)})


def emit(case,events,extra=None):
    row={'case':case,'events':events,'extra':extra or {},'historical_cause_established':False,'original_incident_modified':False}
    print('A12_SIDECAR_TRACE='+json.dumps(row,sort_keys=True))


def create(db,mode):
    with closing(sqlite3.connect(db)) as c:
        assert c.execute('PRAGMA journal_mode='+mode).fetchone()[0].upper()==mode
        c.execute('CREATE TABLE probe(id INTEGER)');c.execute('INSERT INTO probe VALUES(1)');c.commit()


def inspect_copy(raw,path,table):
    path.write_bytes(raw)
    with closing(sqlite3.connect(path)) as c:
        assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        return c.execute('SELECT COUNT(*) FROM '+table).fetchone()[0]


def test_closed_wal_backup_read_lifecycle(tmp_path):
    db=tmp_path/'probe.sqlite3';create(db,'WAL');events=[];observe(events,'AFTER_WRITER_CLOSE',db)
    assert set(events[-1]['files'])=={''}
    raw=pr._stable_file(db);observe(events,'AFTER_NATIVE_STABLE_FILE',db)
    assert inspect_copy(raw,tmp_path/'copy.sqlite3','probe')==1
    # A second identical read measures persistence; never manually remove journals.
    raw2=pr._stable_file(db);observe(events,'AFTER_SECOND_NATIVE_STABLE_FILE',db)
    assert inspect_copy(raw2,tmp_path/'copy2.sqlite3','probe')==1
    emit('CLOSED_WAL',events,{'first_read_created_sidecars':bool(set(events[1]['files'])-{''})})


def test_closed_delete_backup_read_control(tmp_path):
    db=tmp_path/'probe.sqlite3';create(db,'DELETE');events=[];observe(events,'AFTER_WRITER_CLOSE',db)
    raw=pr._stable_file(db);observe(events,'AFTER_NATIVE_STABLE_FILE',db)
    assert inspect_copy(raw,tmp_path/'copy.sqlite3','probe')==1
    emit('CLOSED_DELETE',events,{'read_created_sidecars':bool(set(events[1]['files'])-{''})})


def test_task_finalizer_then_backup_read(tmp_path):
    db=tmp_path/'partition.sqlite3';events=[]
    with closing(sqlite3.connect(db)) as c:
        assert c.execute('PRAGMA journal_mode=WAL').fetchone()[0]=='wal'
        c.execute('CREATE TABLE task_results(task_id TEXT)');c.executemany('INSERT INTO task_results VALUES(?)',[(str(i),) for i in range(96)]);c.commit()
        observe(events,'WRITER_OPEN_COMMITTED',db)
        raw=pr._stable_file(db);observe(events,'BACKUP_WITH_WRITER_OPEN',db)
        assert inspect_copy(raw,tmp_path/'open_copy.sqlite3','task_results')==96
    observe(events,'AFTER_WRITER_CLOSE',db)
    StageScienceRuntime._finalize_idle_partition_database(db);observe(events,'AFTER_NATIVE_FINALIZER',db)
    raw=pr._stable_file(db);observe(events,'AFTER_BACKUP_OF_FINALIZED_DB',db)
    assert inspect_copy(raw,tmp_path/'closed_copy.sqlite3','task_results')==96
    emit('TASK_FINALIZER',events)


def test_retained_incident_copies_before_after_native_reads(tmp_path):
    base=Path(__file__).parent/'fixtures/a12_sidecars';raw=(base/'incident.zip').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==json.loads((base/'manifest.json').read_text())['incident.zip']
    with zipfile.ZipFile(base/'incident.zip') as z:files={n:z.read(n) for n in z.namelist()}
    cases=[]
    for label,tail,table in [('CONTROL','/CONTROL/partition.sqlite3','task_results'),('HELD_WRITE','/HELD_WRITE/partition.sqlite3','task_results'),('PROJECT','/project_writer.sqlite3','probe')]:
        name=next(n for n in files if n.endswith(tail));db=tmp_path/label/'state.sqlite3';db.parent.mkdir();events=[]
        for suffix in ('','-wal','-shm'):
            if name+suffix in files:Path(str(db)+suffix).write_bytes(files[name+suffix])
        observe(events,'EXACT_RETAINED_COPY',db)
        copied=pr._stable_file(db);observe(events,'AFTER_NATIVE_BACKUP_READ',db)
        count=inspect_copy(copied,tmp_path/(label+'_copy.sqlite3'),table);assert count==(2 if label=='PROJECT' else 96)
        StageScienceRuntime._finalize_idle_partition_database(db);observe(events,'AFTER_NATIVE_FINALIZER',db)
        copied=pr._stable_file(db);observe(events,'AFTER_POST_FINALIZER_BACKUP',db)
        assert inspect_copy(copied,tmp_path/(label+'_final.sqlite3'),table)==count
        cases.append({'label':label,'events':events,'rows':count})
    assert hashlib.sha256((base/'incident.zip').read_bytes()).hexdigest()==hashlib.sha256(raw).hexdigest()
    emit('RETAINED_INCIDENT_COPIES',[],{'copies':cases})
