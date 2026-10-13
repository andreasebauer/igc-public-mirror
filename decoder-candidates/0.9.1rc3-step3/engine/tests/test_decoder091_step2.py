"""Step2 interruption, SQLite sealing and actual durability barrier checks."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import stat
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from infinity_grid import bulk_submission as bulk,submission as sub,portable_registry as reg
from infinity_grid import preservation as pr,completion_evidence as ce
from infinity_grid import immutable_publication as publication
from infinity_grid.canon import write_json_atomic


def inert_engine(tmp_path):
    engine=tmp_path/'engine';(engine/'infinity_grid').mkdir(parents=True)
    (engine/'infinity_grid/__init__.py').write_text('# inert capture; never executed\n')
    (engine/'infinity_grid/_version.py').write_text("__version__ = '0.9.1rc2'\n")
    (engine/'tests').mkdir();(engine/'tests/test_fixture.py').write_text('def test_inert():pass\n')
    return engine


def inert_capture(tmp_path):
    engine=inert_engine(tmp_path)
    spec={'schema_id':sub.SPEC_SCHEMA,'job_id':'ENG.STEP2.BARRIER',
          'engine_source':str(engine),'project_source':None,
          'question':{'stage_id':'ENG:STEP2','description':'Inert durability fixture',
                      'outcomes':['PASS'],'stopping_rule':'Never execute'},
          'execution':{'kind':'VALIDATION','nodes':['tests/test_fixture.py::test_inert']},
          'resources':{'workers':1,'start_method':'spawn','memory_budget_bytes':268435456,
                       'workspace_budget_bytes':268435456,'wall_seconds_max':30},
          'inputs':[],'environment':{'python':'3.13','requirements':[],'artifacts':[]},
          'output_contract':{'engineering_fixture':'NO_EXECUTION'}}
    reg.initialize(tmp_path/'store','Barrier fixture',engine)
    return Path(sub.capture(tmp_path/'store',spec)['workspace'])


@pytest.mark.parametrize('name',['mapping.sqlite3','partition.sqlite3','state_store.sqlite3'])
@pytest.mark.parametrize('suffix',['-journal','-wal','-shm'])
def test_every_engine_database_journal_refused_in_source_and_destination(tmp_path,name,suffix):
    work=tmp_path/'work';db=work/'chain/decoder_stage_runtime/p'/name;db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as c:c.execute('create table records(x)')
    Path(str(db)+suffix).write_bytes(b'keep for diagnosis')
    with pytest.raises(sub.SubmissionError,match='TASK_DATABASE_NOT_QUIESCENT'):
        pr.require_quiescent_task_databases(work)
    with pytest.raises(sub.SubmissionError,match='TASK_DATABASE_NOT_QUIESCENT'):
        ce.stage_snapshot(work,tmp_path/'sealed')
    assert Path(str(db)+suffix).read_bytes()==b'keep for diagnosis'


def test_orphan_mapping_journal_is_refused(tmp_path):
    journal=tmp_path/'chain/decoder_stage_runtime/p/mapping.sqlite3-journal';journal.parent.mkdir(parents=True)
    journal.write_bytes(b'orphan')
    with pytest.raises(sub.SubmissionError,match='TASK_DATABASE_NOT_QUIESCENT'):pr.require_quiescent_task_databases(tmp_path)


def test_mapping_backup_is_exact_semantically_and_closed(tmp_path):
    work=tmp_path/'work';db=work/'chain/decoder_stage_runtime/p/mapping.sqlite3';db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as c:
        c.execute('create table records(x)');c.executemany('insert into records values(?)',[(i,) for i in range(20)])
    before=db.read_bytes();sealed=tmp_path/'sealed';ce.stage_snapshot(work,sealed)
    copied=sealed/db.relative_to(work)
    with sqlite3.connect(copied) as c:
        assert c.execute('pragma quick_check').fetchone()==('ok',)
        assert c.execute('select x from records order by x').fetchall()==[(i,) for i in range(20)]
        assert c.execute('pragma journal_mode').fetchone()==('delete',)
    assert db.read_bytes()==before;pr.require_quiescent_task_databases(sealed)


def test_corrupt_mapping_not_published_and_ordinary_script_database_scope_unchanged(tmp_path):
    work=tmp_path/'work';db=work/'chain/decoder_stage_runtime/p/mapping.sqlite3';db.parent.mkdir(parents=True)
    db.write_bytes(b'SQLite format 3\0'+b'broken')
    with pytest.raises(sqlite3.DatabaseError):ce.stage_snapshot(work,tmp_path/'sealed')
    ordinary=tmp_path/'ordinary';ordinary.mkdir();(ordinary/'mapping.sqlite3').write_bytes(b'ordinary artifact')
    ce.stage_snapshot(ordinary,tmp_path/'ordinary_sealed')
    assert (tmp_path/'ordinary_sealed/mapping.sqlite3').read_bytes()==b'ordinary artifact'


def test_commit_directory_fsync_precedes_current_and_repeats_on_retry(tmp_path,monkeypatch):
    workspace=inert_capture(tmp_path);events=[];real_sync=publication.sync_directory;real_write=pr.write_json_atomic
    def sync(path):events.append(('sync',str(path)));return real_sync(path)
    def write(path,value):
        if Path(path).name=='CURRENT.json':events.append(('current',str(path)))
        return real_write(path,value)
    monkeypatch.setattr(publication,'sync_directory',sync);monkeypatch.setattr(pr,'write_json_atomic',write)
    first=pr.make_checkpoint(workspace,'STEP2_BARRIER')
    commits=str(pr._root(workspace)/'commits');current=next(i for i,x in enumerate(events) if x[0]=='current')
    assert ('sync',commits) in events[:current]
    count=len(events);again=pr.make_checkpoint(workspace,'STEP2_IDENTICAL_RETRY')
    assert again['latest_checkpoint']==first['latest_checkpoint']
    assert ('sync',commits) in events[count:]


def test_failed_commit_barrier_never_advances_pointer_and_missing_only_retry_works(tmp_path,monkeypatch):
    workspace=inert_capture(tmp_path);root=pr._root(workspace);real=publication.sync_directory
    def fail(path):
        if Path(path)==root/'commits':raise OSError('SIMULATED_DIRECTORY_FSYNC_FAILURE')
        return real(path)
    with monkeypatch.context() as m:
        m.setattr(publication,'sync_directory',fail)
        with pytest.raises(OSError,match='FSYNC_FAILURE'):pr.make_checkpoint(workspace,'FAILURE')
    assert not (root/'CURRENT.json').exists()
    assert len(list((root/'commits').glob('*.json')))==1
    resumed=pr.make_checkpoint(workspace,'RECOVER')
    assert resumed['latest_checkpoint']==json.loads((root/'CURRENT.json').read_bytes())['sha256']
    assert sub.save_status(workspace)['status']=='SAVE_REQUIRED'


@pytest.mark.parametrize('point',['plan','descriptors','plan_write','rows_write','spec_write','registry','capture','pointer'])
def test_preparation_interruption_resumes_without_duplicate_capture(tmp_path,monkeypatch,point):
    # Small explicit registered engineering input; no census or handler dispatch.
    engine=inert_engine(tmp_path);adapter=tmp_path/'adapter';adapter.mkdir()
    (adapter/'bulk_adapter.py').write_text('def handler(stage,runtime):\n    return None\ndef evaluate(payload):\n    return None\n')
    raw=tmp_path/'raw';raw.write_bytes(b'raw fixture');prefix=tmp_path/'prefix';prefix.write_bytes(b'prefix fixture')
    binding=tmp_path/'sqlite';binding.write_bytes(b'capture-only fixture')
    planned={'plan_sha256':'p','preserved_rows':1,'domain_rows':3,'prefix':{'records':1},'windows':[{'start':1,'stop':3,'rows':2}]}
    rows=[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64} for i in (1,2)]
    monkeypatch.setattr(bulk,'build_plan',lambda *a,**k:planned)
    monkeypatch.setattr(bulk,'window_descriptors',lambda *a,**k:rows)
    root=tmp_path/'window';args=(root,engine,adapter,raw,prefix,binding,0)
    with monkeypatch.context() as m:
        if point in {'plan','descriptors'}:
            name='build_plan' if point=='plan' else 'window_descriptors';real=getattr(bulk,name)
            def interrupted(*a,**k):real(*a,**k);raise RuntimeError('INTERRUPTED_PREPARATION')
            m.setattr(bulk,name,interrupted)
        elif point in {'plan_write','rows_write','spec_write','pointer'}:
            filename={'plan_write':'full-domain-plan.json','rows_write':'descriptor-window.json','spec_write':'SPECIFICATION.json','pointer':'WORKSPACE.json'}[point]
            real=bulk.write_json_atomic
            def interrupted(path,value):
                result=real(path,value)
                if Path(path).name==filename:raise RuntimeError('INTERRUPTED_PREPARATION')
                return result
            m.setattr(bulk,'write_json_atomic',interrupted)
        else:
            target=reg if point=='registry' else sub;name='initialize' if point=='registry' else 'capture';real=getattr(target,name)
            def interrupted(*a,**k):real(*a,**k);raise RuntimeError('INTERRUPTED_PREPARATION')
            m.setattr(target,name,interrupted)
        with pytest.raises(RuntimeError,match='INTERRUPTED_PREPARATION'):bulk.prepare_window(*args)
    prepared=bulk.prepare_window(*args);workspace=bulk.window_workspace(root)
    assert str(workspace)==prepared['capture']['workspace']
    assert len(list((root/'store/captures').glob('*/CAPTURE.json')))==1
    before=(workspace/'CAPTURE.json').read_bytes()
    assert bulk.prepare_window(*args)['capture']['capture_id']==prepared['capture']['capture_id']
    assert (workspace/'CAPTURE.json').read_bytes()==before
    with pytest.raises(ValueError,match='PREPARATION_CHANGED'):bulk.prepare_window(*args,workers=2)


def test_partial_registry_keeps_uuid_and_conflicting_source_refuses(tmp_path,monkeypatch):
    engine=inert_engine(tmp_path);store=tmp_path/'store'
    with monkeypatch.context() as m:
        m.setattr(reg,'put',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('INIT_INTERRUPTED')))
        with pytest.raises(RuntimeError):reg.initialize(store,'same',engine)
    identity=(store/'coordination/PROJECT.json').read_bytes()
    reg.initialize(store,'same',engine,resume=True)
    assert (store/'coordination/PROJECT.json').read_bytes()==identity
    reg.initialize(store,'same',engine,resume=True)
    (engine/'infinity_grid/__init__.py').write_text('# changed\n')
    with pytest.raises(reg.InvocationRefused,match='SOURCE_CHANGED'):reg.initialize(store,'same',engine,resume=True)


def test_window_pointer_relocation_conflict_and_ambiguity(tmp_path):
    workspace=inert_capture(tmp_path)
    directory=tmp_path/'window';directory.mkdir()
    shutil.move(str(tmp_path/'store'),str(directory/'store'))
    local=directory/'store/captures'/workspace.name
    write_json_atomic(directory/'WORKSPACE.json',{'path':str(workspace)})
    assert bulk.window_workspace(directory)==local
    pointer=directory/'WORKSPACE.json'
    pointer.write_text(json.dumps({'path':str(tmp_path/'wrong-id')}))
    with pytest.raises(ValueError,match='POINTER_IDENTITY'):bulk.window_workspace(directory)
    other=tmp_path/'other'/workspace.name;other.mkdir(parents=True)
    pointer.write_text(json.dumps({'path':str(other)}))
    with pytest.raises(ValueError,match='POINTER_CONFLICT'):bulk.window_workspace(directory)
    pointer.unlink();duplicate=local.parent/'duplicate';duplicate.mkdir()
    shutil.copy(local/'CAPTURE.json',duplicate/'CAPTURE.json')
    with pytest.raises(ValueError,match='RECOVERY_REQUIRED'):bulk.window_workspace(directory)


def test_killed_worker_retains_committed_mapping_and_only_missing_task_resumes(tmp_path):
    from infinity_grid.bulk_mapping_store import MappingStore
    from infinity_grid.execution import TaskSpec
    from infinity_grid.canon import canonical_sha256
    from infinity_grid.structural_encoding import structural_canonical_bytes
    tasks=[TaskSpec('t'+str(i),'ENG','a'*64,{'descriptors':[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64}]}) for i in range(2)]
    def result(i):
        w={'nodes':[0]};signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':[{'ordinal':i,'raw_edge':[i],
            'literal_descriptor_sha256':'d'*64,'relabeling_witness':w,'relabeling_witness_sha256':canonical_sha256(w)}]}
        return {'signature_canonical_bytes':structural_canonical_bytes(signature),'signature_sha256':canonical_sha256(signature),'outcome_count':1}
    source=Path(__file__).resolve().parents[1];db=tmp_path/'mapping.sqlite3';ready=tmp_path/'READY'
    code="""import sys,time,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from infinity_grid.bulk_mapping_store import MappingStore
from infinity_grid.execution import TaskSpec
from infinity_grid.canon import canonical_sha256 as sha
from infinity_grid.structural_encoding import structural_canonical_bytes as enc
tasks=[TaskSpec('t'+str(i),'ENG','a'*64,{'descriptors':[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64}]}) for i in range(2)]
w={'nodes':[0]};signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':[{'ordinal':0,'raw_edge':[0],'literal_descriptor_sha256':'d'*64,'relabeling_witness':w,'relabeling_witness_sha256':sha(w)}]}
s=MappingStore(sys.argv[2],{},tasks);s.commit('t0',{'signature_canonical_bytes':enc(signature),'signature_sha256':sha(signature),'outcome_count':1})
Path(sys.argv[3]).write_text('COMMITTED');time.sleep(30)
"""
    import time
    from infinity_grid.isolated_runtime import subprocess_argv,subprocess_environment
    child=subprocess.Popen(subprocess_argv(['-B','-c',code,str(source),str(db),str(ready)]),env=subprocess_environment())
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists();child.kill();child.wait(timeout=5)
        store=MappingStore(db,{},tasks);before=list(store.iter_rows())
        assert [t.task_id for t in store.pending()]==['t1']
        store.commit('t1',result(1));assert store.summary()['rows']==2
        assert list(store.iter_rows())[0]==before[0];store.close()
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)
