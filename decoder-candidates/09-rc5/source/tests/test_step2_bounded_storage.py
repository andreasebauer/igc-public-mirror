"""Synthetic fault and boundary tests; no remote receipt authority."""
from pathlib import Path
import copy
import json
import sqlite3

import pytest
from infinity_grid import preservation as pr, submission as sub
from infinity_grid.canon import canonical_sha256,write_json_atomic
from infinity_grid.replay_reference_data import ReplayReferenceDataStore,ReferenceDataError
from infinity_grid.result_contracts import policy
from infinity_grid.checkpoint_stream import snapshot_state,archive_paths
from tests.test_p3_replay_reference_data import record


def rows(n):
    out=[]
    for i in range(n):
        if i==0:
            out.append(record('CANONICAL_OBJECT',f'IGRD/STEP2/OBJECT/N{i:05d}',
                {'object_identity':str(i),'carrier_schema':'SYNTHETIC_STEP2',
                 'canonical_bytes_sha256':'0'*64,'formation_provenance':'ENGINEERING_FIXTURE'}))
        else:
            out.append(record('MECHANISM',f'IGRD/STEP2/OBJECT/N{i:05d}',
                {'mechanism_id':str(i),'input_record_ids':[out[-1]['record_id']],
                 'output_record_ids':[f'IGRD/STEP2/PRODUCED/N{i:05d}'],'determinism':'DETERMINISTIC',
                 'implementation_hashes':[{'ref':'synthetic','sha256':'0'*64}]},
                dependencies=[out[-1]['record_id']]))
    return out


def test_batch_matches_single_rows_and_reuses_identical_records(tmp_path):
    items=rows(30);one=ReplayReferenceDataStore.initialize(tmp_path/'one');batch=ReplayReferenceDataStore.initialize(tmp_path/'batch')
    for row in items:one.put(row)
    assert batch.put_many(items)==items
    assert batch.manifest==one.manifest
    assert batch.put_many(items)==items
    assert len(list((batch.root/'batches').glob('*.json')))==1


@pytest.mark.parametrize('cut',[1,3,7])
def test_batch_recovery_restores_only_the_declared_transaction(tmp_path,cut):
    store=ReplayReferenceDataStore.initialize(tmp_path/'batch');items=rows(7)
    with pytest.raises(ReferenceDataError,match='INJECTED'):store.put_many(items,_interrupt_after_records=cut)
    assert len(json.loads((store.root/'MANIFEST.json').read_text())['record_ids'])==0
    reopened=ReplayReferenceDataStore(store.root)
    assert reopened.manifest['record_ids']==[r['record_id'] for r in items]
    reopened.verify();reopened.put_many(items)
    assert len(list((store.root/'batch_commits').glob('*.json')))==1


def test_batch_dependency_and_collision_reject_before_journal(tmp_path):
    store=ReplayReferenceDataStore.initialize(tmp_path/'batch');items=rows(2)
    with pytest.raises(ReferenceDataError,match='dependency'):store.put_many(reversed(items))
    assert not list((store.root/'batches').glob('*.json'))
    store.put_many(items)
    wrong=copy.deepcopy(items[0]);wrong.pop('record_sha256');wrong['payload']['object_identity']='changed'
    with pytest.raises(ReferenceDataError,match='collision'):store.put_many([wrong])


def test_batch_committed_manifest_rollback_refuses(tmp_path):
    store=ReplayReferenceDataStore.initialize(tmp_path/'batch');before=(store.root/'MANIFEST.json').read_bytes()
    store.put_many(rows(4));(store.root/'MANIFEST.json').write_bytes(before)
    with pytest.raises(ReferenceDataError,match='REFERENCE_INDEX_ROLLBACK'):ReplayReferenceDataStore(store.root)


def test_stream_snapshot_backs_up_live_wal_and_excludes_journal(tmp_path):
    folder=tmp_path/'runtime/runs/test';folder.mkdir(parents=True)
    db=folder/'state_store.sqlite3';conn=sqlite3.connect(db);conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE t(x)');conn.execute('INSERT INTO t VALUES(42)');conn.commit()
    with snapshot_state(tmp_path,policy({})) as (files,modes,shards,temp):
        assert not any(n.endswith(('-wal','-shm')) for n in files)
        backup=files.paths['runtime/runs/test/state_store.sqlite3']
        with sqlite3.connect(backup) as read:assert read.execute('SELECT x FROM t').fetchall()==[(42,)]
        archive_paths(files.paths,temp/'state.zip')
    conn.close()


def test_state_size_refuses_before_reading_payload(tmp_path,monkeypatch):
    p=tmp_path/'runtime/large.bin';p.parent.mkdir();p.write_bytes(b'x'*4096)
    original=Path.open
    def guard(path,*args,**kw):
        if path==p:raise AssertionError('oversized state was opened')
        return original(path,*args,**kw)
    monkeypatch.setattr(Path,'open',guard)
    limits=policy({'preservation':{'max_state_bytes':1024,'max_state_file_bytes':1024}})
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_STATE_RAW_LIMIT'):
        with snapshot_state(tmp_path,limits):pass


def test_shard_reference_and_bytes_are_verified(tmp_path):
    import hashlib
    root=tmp_path/'runtime/runs/test';folder=root/'bulk_shards';folder.mkdir(parents=True)
    data=b'bounded immutable bulk';digest=hashlib.sha256(data).hexdigest();p=folder/(digest+'.bin');p.write_bytes(data)
    write_json_atomic(root/'BULK_SHARDS.json',{'schema_id':'IG_STAGE_BULK_SHARDS_V1','shards':{'one':
        {'sha256':digest,'size_bytes':len(data),'path':'bulk_shards/'+digest+'.bin'}}})
    with snapshot_state(tmp_path,policy({})) as (files,modes,shards,temp):
        assert len(shards)==1 and shards[0]['sha256']==digest
        assert all('/bulk_shards/' not in n for n in files)
    p.write_bytes(b'z'*len(data))
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_SHARD_HASH'):
        with snapshot_state(tmp_path,policy({})):pass
    p.unlink()
    with pytest.raises(sub.SubmissionError,match='CHECKPOINT_SHARD_REFERENCE_MISSING'):
        with snapshot_state(tmp_path,policy({})):pass


def test_capture_limit_reserves_final_pause_checkpoint():
    limits=policy({'preservation':{'max_retained_commits':3}})
    state={'pending_bytes':0,'pending_checkpoints':[],'oldest_pending_age_seconds':0,'retained_checkpoint_count':2}
    with pytest.raises(sub.SubmissionError,match='SAVE_BACKLOG_PAUSE'):pr._check_backlog(state,limits,reserve=True)
    assert pr._check_backlog(state,limits,reserve=False)==state


def test_lazy_task_input_and_generation_result_limits(tmp_path,monkeypatch):
    from infinity_grid import v05_stage_runtime as runtime
    from infinity_grid.execution import TaskSpec
    def endless():
        for i in range(1000):yield TaskSpec(str(i),'SYNTHETIC','0'*64,{'i':i})
    with pytest.raises(runtime.StageRuntimeError,match='PHASE_TASK_COUNT_LIMIT'):runtime._bounded_task_list(endless(),3)


def test_stage_accounting_includes_shards_and_batched_ledger(tmp_path):
    from infinity_grid.v05_stage_runtime import _stage_workspace_bytes,StageRuntimeError
    import hashlib
    root=tmp_path/'stage';root.mkdir()
    store=ReplayReferenceDataStore.initialize(root/'replay_reference_data');store.put_many(rows(4))
    data=b'bounded engineering shard';digest=hashlib.sha256(data).hexdigest()
    folder=root/'bulk_shards';folder.mkdir();(folder/(digest+'.bin')).write_bytes(data)
    write_json_atomic(root/'BULK_SHARDS.json',{'schema_id':'IG_STAGE_BULK_SHARDS_V1','shards':{
        'one':{'sha256':digest,'size_bytes':len(data),'path':'bulk_shards/'+digest+'.bin'}}})
    assert _stage_workspace_bytes(root)==sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    (folder/'untracked.json').write_text('{}')
    with pytest.raises(StageRuntimeError,match='invalid bounded bulk shard'):_stage_workspace_bytes(root)


def test_checkpoint_roundtrip_retains_shard_bytes_and_manifest(tmp_path):
    from tests.test_paused_checkpoint_recovery import fixture
    from infinity_grid.v05_stage_runtime import _stage_workspace_bytes
    import hashlib
    root,_=fixture(tmp_path)
    stage=root/'runtime/runs/unit/decoder_stage_runtime/step2';folder=stage/'bulk_shards';folder.mkdir(parents=True)
    data=b'unit-only immutable shard';digest=hashlib.sha256(data).hexdigest()
    shard=folder/(digest+'.bin');shard.write_bytes(data);shard.chmod(0o440)
    write_json_atomic(stage/'BULK_SHARDS.json',{'schema_id':'IG_STAGE_BULK_SHARDS_V1','shards':{
        'one':{'sha256':digest,'size_bytes':len(data),'path':'bulk_shards/'+digest+'.bin'}}})
    size=_stage_workspace_bytes(stage);pr.make_checkpoint(root,'UNIT_SHARD_ROUNDTRIP')
    target=tmp_path/'checkpoint.zip';pr.export_checkpoint(root,target)
    dest=tmp_path/'restored';pr.restore_checkpoint(target,dest,sub._sha(target.read_bytes()))
    restored=dest/shard.relative_to(root)
    assert restored.read_bytes()==data and restored.stat().st_mode&0o777==0o440
    assert _stage_workspace_bytes(dest/stage.relative_to(root))==size


def test_operator_observation_closes_sqlite_connection(tmp_path,monkeypatch):
    from tools.step2_capture_operator import committed
    from contextlib import closing
    root=tmp_path/'runtime/runs/unit';root.mkdir(parents=True)
    with closing(sqlite3.connect(root/'state_store.sqlite3')) as conn:
        conn.execute('CREATE TABLE generation_tasks(task_id,payload_sha256,state_occurrence_count,metrics_json)')
        conn.execute('CREATE TABLE occurrences(occurrence_id,task_id,state_token)');conn.commit()
    connect=sqlite3.connect;opened=[]
    def tracked(*args,**kwargs):
        connection=connect(*args,**kwargs);opened.append(connection);return connection
    monkeypatch.setattr(sqlite3,'connect',tracked)
    assert committed(tmp_path)=={'tasks':[],'occurrences':[],'worker_identities':[]}
    assert len(opened)==1
    with pytest.raises(sqlite3.ProgrammingError,match='closed'):opened[0].execute('SELECT 1')


def test_sqlite_streams_never_fork_live_controller_connection(monkeypatch):
    from infinity_grid.execution import ExecutionPolicy,ExecutionError,_choose_stream_start_method
    monkeypatch.delenv('IG_DECODER_START_METHOD',raising=False)
    assert _choose_stream_start_method(ExecutionPolicy())=='spawn'
    with pytest.raises(ExecutionError,match='SQLITE_STREAM_FORK_UNSAFE'):
        _choose_stream_start_method(ExecutionPolicy(start_method='fork'))
    monkeypatch.setenv('IG_DECODER_START_METHOD','fork')
    with pytest.raises(ExecutionError,match='SQLITE_STREAM_FORK_UNSAFE'):
        _choose_stream_start_method(ExecutionPolicy())
    assert _choose_stream_start_method(ExecutionPolicy(start_method='spawn'))=='spawn'


def test_task_database_rollback_policy_preserves_atomic_rows_and_backup(tmp_path):
    from contextlib import closing
    from infinity_grid.v05_stage_runtime import _configure_task_database
    db=tmp_path/'state.sqlite3'
    with closing(sqlite3.connect(db)) as conn:
        _configure_task_database(conn)
        assert conn.execute('PRAGMA journal_mode').fetchone()[0]=='delete'
        assert conn.execute('PRAGMA synchronous').fetchone()[0]==2
        conn.execute('CREATE TABLE t(id PRIMARY KEY)');conn.commit()
        conn.execute('INSERT INTO t VALUES(1)');conn.commit()
        conn.execute('INSERT INTO t VALUES(2)');conn.rollback()
        with closing(sqlite3.connect(tmp_path/'backup.sqlite3')) as dest:
            conn.backup(dest)
            assert dest.execute('SELECT id FROM t').fetchall()==[(1,)]
    assert not list(tmp_path.glob('state.sqlite3-*'))
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute('SELECT id FROM t').fetchall()==[(1,)]
