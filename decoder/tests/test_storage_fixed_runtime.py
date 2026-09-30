"""Bounded fixed-runtime engineering checks; invoked only by captured VALIDATION."""
from pathlib import Path
import hashlib
import json
import queue
import sqlite3
import subprocess
import sys
import threading
import pytest
from infinity_grid.sqlite_attestation import (
    UPSTREAM_FIXED_SOURCE, require_wal_fix, observe_runtime,
    SQLiteAdmissionError, verify_runtime_binding,
)

EXPECTED_LIBRARY = 'd5919c179c81d44ce155745caf49e4c50c6f37bd7af6460b7d6ce3a4a4e26c7e'
CHILD = Path(__file__).parent / 'fixtures/storage_fixed_runtime/child_probe.py'

def test_exact_fixed_library_loaded():
    o = require_wal_fix()
    assert o['sqlite_version'] == '3.51.3'
    assert o['sqlite_source_id'] == UPSTREAM_FIXED_SOURCE
    assert [x['sha256'] for x in o['loaded_sqlite_libraries']] == [EXPECTED_LIBRARY]
    assert o['sqlite_thread_safety'] == 3
    assert 'THREADSAFE=1' in o['sqlite_compile_options']

def test_fresh_python_inherits_fixed_library():
    p = subprocess.run([sys.executable, str(CHILD), 'probe'], capture_output=True, timeout=20)
    assert p.returncode == 0, p.stderr.decode(errors='replace')
    o = json.loads(p.stdout)
    assert o == {'version': '3.51.3', 'source_id': UPSTREAM_FIXED_SOURCE,
                 'library_hashes': [EXPECTED_LIBRARY]}

def test_disk_wal_reader_snapshot_and_commits(tmp_path):
    path = tmp_path/'live.sqlite'
    a = sqlite3.connect(path, isolation_level=None)
    b = sqlite3.connect(path, isolation_level=None)
    try:
        assert a.execute('PRAGMA journal_mode=WAL').fetchone()[0] == 'wal'
        a.execute('PRAGMA synchronous=FULL')
        a.execute('PRAGMA foreign_keys=ON')
        a.execute('PRAGMA wal_autocheckpoint=0')
        assert a.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert a.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        a.execute('CREATE TABLE t(k INTEGER PRIMARY KEY)')
        a.execute('INSERT INTO t VALUES(1)')
        b.execute('BEGIN')
        assert b.execute('SELECT count(*) FROM t').fetchone()[0] == 1
        a.execute('INSERT INTO t VALUES(2)')
        assert b.execute('SELECT count(*) FROM t').fetchone()[0] == 1
        b.execute('COMMIT')
        assert b.execute('SELECT count(*) FROM t').fetchone()[0] == 2
        assert a.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0
        assert a.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        b.close(); a.close()

def test_abrupt_process_exit_recovers_only_committed_rows(tmp_path):
    path = tmp_path/'crash.sqlite'
    p = subprocess.run([sys.executable, str(CHILD), 'crash', str(path)], capture_output=True, timeout=20)
    assert p.returncode == 73, p.stderr.decode(errors='replace')
    assert Path(str(path)+'-wal').exists()
    c = sqlite3.connect(path)
    try:
        assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert c.execute('SELECT * FROM t ORDER BY k').fetchall() == [(1, 'committed')]
        assert c.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] == 0
    finally:
        c.close()

def test_live_wal_backup_seals_readonly_without_mutation(tmp_path):
    live = tmp_path/'live.sqlite'; closed = tmp_path/'closed.sqlite'
    a = sqlite3.connect(live, isolation_level=None)
    a.execute('PRAGMA journal_mode=WAL'); a.execute('PRAGMA synchronous=FULL')
    a.execute('CREATE TABLE t(k INTEGER PRIMARY KEY, v TEXT NOT NULL)')
    a.executemany('INSERT INTO t VALUES(?,?)', [(i, str(i*i)) for i in range(128)])
    b = sqlite3.connect(closed)
    try:
        a.backup(b, pages=8)
        assert b.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        assert b.execute('PRAGMA journal_mode=DELETE').fetchone()[0] == 'delete'
    finally:
        b.close(); a.close()
    before = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in tmp_path.iterdir() if p.is_file()}
    c = sqlite3.connect(closed.as_uri()+'?mode=ro&immutable=1', uri=True)
    try:
        assert c.execute('SELECT count(*) FROM t').fetchone()[0] == 128
        with pytest.raises(sqlite3.OperationalError):
            c.execute("INSERT INTO t VALUES(999, 'forbidden')")
    finally:
        c.close()
    after = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
             for p in tmp_path.iterdir() if p.is_file()}
    assert before == after
    assert not any(Path(str(closed)+s).exists() for s in ('-wal','-shm','-journal'))

def test_exact_large_decimal_text_no_float_roundtrip(tmp_path):
    c = sqlite3.connect(tmp_path/'exact.sqlite')
    value = '9'*6000
    try:
        c.execute('CREATE TABLE exact(v TEXT NOT NULL) STRICT')
        c.execute('INSERT INTO exact VALUES(?)', (value,)); c.commit()
        actual, typeof = c.execute('SELECT v, typeof(v) FROM exact').fetchone()
        assert actual == value and typeof == 'text'
    finally:
        c.close()

def test_original_unfixed_binding_rejected():
    o = observe_runtime()
    o['sqlite_version'] = '3.46.1'
    o['sqlite_source_id'] = '2024-08-13 09:16:08 c9c2ab54ba1f5f46360f1b4f35d849cd3f080e6fc2b6c60e91b16c63f69aalt1'
    with pytest.raises(SQLiteAdmissionError, match='PINNED_RUNTIME_MISMATCH'):
        verify_runtime_binding(o)

def test_changed_library_hash_binding_rejected():
    o = observe_runtime()
    o['loaded_sqlite_libraries'][0]['sha256'] = '0'*64
    with pytest.raises(SQLiteAdmissionError, match='PINNED_SQLITE_LIBRARY_MISMATCH'):
        verify_runtime_binding(o)

def test_bounded_concurrent_writes_and_checkpoints(tmp_path):
    # Overlapping-connection smoke check, NOT a reproduction/proof of the rare upstream race.
    path = tmp_path/'concurrent.sqlite'
    c = sqlite3.connect(path, isolation_level=None)
    c.execute('PRAGMA journal_mode=WAL'); c.execute('CREATE TABLE t(k INTEGER PRIMARY KEY)'); c.close()
    barrier = threading.Barrier(2, timeout=20)
    errors = queue.Queue()
    def worker(kind):
        con = None
        try:
            con = sqlite3.connect(path, isolation_level=None, timeout=5)
            con.execute('PRAGMA synchronous=FULL'); con.execute('PRAGMA wal_autocheckpoint=0')
            for n in range(64):
                barrier.wait()
                if kind == 'writer':
                    con.execute('INSERT INTO t VALUES(?)', (n,))
                else:
                    assert con.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0] in (0, 1)
                barrier.wait()
        except BaseException as exc:
            errors.put(repr(exc)); barrier.abort()
        finally:
            if con is not None: con.close()
    threads = [threading.Thread(target=worker, args=(kind,), daemon=True)
               for kind in ('writer','checkpoint')]
    for t in threads: t.start()
    for t in threads: t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), 'WORKER_NOT_QUIESCENT'
    assert errors.empty(), list(errors.queue)
    c = sqlite3.connect(path)
    try:
        assert c.execute('SELECT k FROM t ORDER BY k').fetchall() == [(n,) for n in range(64)]
        assert c.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    finally:
        c.close()
