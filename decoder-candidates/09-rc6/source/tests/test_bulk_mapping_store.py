import json,sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from infinity_grid.bulk_mapping_store import MappingStore,BulkMappingError,read_mapping_rows
from infinity_grid.structural_encoding import structural_canonical_bytes as enc
from infinity_grid.canon import canonical_sha256 as sha


def task(i):
    return SimpleNamespace(task_id=f't{i:04d}',binding_sha256='a'*64,payload={'descriptors':[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64}]})

def result(i,witness=None):
    witness=witness or {'schema':'full','nodes':list(range(100))}
    row={'ordinal':i,'raw_edge':[i],'literal_descriptor_sha256':'d'*64,'relabeling_witness':witness,'relabeling_witness_sha256':sha(witness),'V3_identity':str(i)}
    raw={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':[row]}
    return {'signature_canonical_bytes':enc(raw),'signature_sha256':sha(raw),'outcome_count':1,'metrics':{}}

def test_more_than_64_tasks_deduplicates_exact_witness_and_reads_in_order(tmp_path):
    tasks=[task(i) for i in range(80)];p=tmp_path/'mapping.sqlite3';store=MappingStore(p,{'source':'a'},tasks)
    for i in reversed(range(80)):store.commit(tasks[i].task_id,result(i))
    summary=store.summary();assert summary['rows']==80 and summary['witness_objects']==1
    assert summary['witness_compressed_bytes']<summary['witness_raw_bytes']
    store.commit(tasks[0].task_id,result(0));store.close()
    assert [r['ordinal'] for r in read_mapping_rows(p)]==list(range(80))
    resumed=MappingStore(p,{'source':'a'},tasks);assert not resumed.pending();resumed.close()


def test_transaction_failure_rolls_back_rows_witnesses_and_task(tmp_path):
    tasks=[task(i) for i in range(2)];store=MappingStore(tmp_path/'mapping.sqlite3',{'source':'a'},tasks)
    r=result(0);r['signature_canonical_bytes']=enc({'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':[dict(json.loads(r['signature_canonical_bytes'])['rows'][0],relabeling_witness_sha256='b'*64)]});r['signature_sha256']=sha(json.loads(r['signature_canonical_bytes']))
    with pytest.raises(BulkMappingError,match='WITNESS_IDENTIFIER'):store.commit(tasks[0].task_id,r)
    assert store.conn.execute('select count(*) from witnesses').fetchone()[0]==0
    assert store.conn.execute('select count(*) from rows').fetchone()[0]==0
    assert store.conn.execute('select count(*) from tasks').fetchone()[0]==0
    store.close()


def test_changed_scope_and_changed_retry_are_refused(tmp_path):
    p=tmp_path/'mapping.sqlite3';tasks=[task(0)];s=MappingStore(p,{'source':'a'},tasks);s.commit(tasks[0].task_id,result(0))
    with pytest.raises(BulkMappingError,match='RETRY_CHANGED'):s.commit(tasks[0].task_id,result(0,{'other':'full'}))
    s.close()
    with pytest.raises(BulkMappingError,match='SCOPE_MISMATCH'):MappingStore(p,{'source':'b'},tasks)


def test_corrupt_compressed_witness_and_missing_result_are_refused(tmp_path):
    p=tmp_path/'mapping.sqlite3';tasks=[task(0)];s=MappingStore(p,{'source':'a'},tasks);s.commit(tasks[0].task_id,result(0));s.conn.execute('update witnesses set raw_size=raw_size+1');s.conn.commit();s.close()
    with pytest.raises(BulkMappingError,match='WITNESS_BYTES_INVALID'):list(read_mapping_rows(p))


def test_duplicate_ordinals_rejected_before_science(tmp_path):
    t=task(0);other=SimpleNamespace(task_id='other',binding_sha256='b'*64,payload=t.payload)
    with pytest.raises(BulkMappingError,match='DUPLICATE_OR_INVALID'):MappingStore(tmp_path/'mapping.sqlite3',{},[t,other])


def test_receipt_chain_preserves_prefix_and_refuses_gaps_or_changed_cutover():
    from infinity_grid.bulk_submission import advance_progress
    prefix={'records':56832,'files':[{'sha256':'keep'}]}
    plan={'plan_sha256':'plan','domain_rows':56836,'prefix':prefix,'windows':[{'start':56832,'stop':56834,'rows':2},{'start':56834,'stop':56836,'rows':2}]}
    a={'window_index':0,'start':56832,'stop':56834,'rows':2,'status':'VERIFIED_SAVED_WINDOW','source_sha256':'a','package_sha256':'b'}
    one=advance_progress(plan,{},a);assert one['inherited_prefix'] is prefix and one['next_ordinal']==56834
    assert advance_progress(plan,one,a)==one
    b=dict(a,window_index=1,start=56834,stop=56836)
    with pytest.raises(ValueError,match='ORDER'):advance_progress(plan,{},b)
    with pytest.raises(ValueError,match='SOURCE_CHANGED'):advance_progress(plan,one,dict(b,source_sha256='changed'))
    final=advance_progress(plan,one,b);assert final['missing_rows']==0 and final['inherited_prefix']==prefix


def test_orphan_rows_refused_on_resume(tmp_path):
    p=tmp_path/'mapping.sqlite3';tasks=[task(0)]
    s=MappingStore(p,{},tasks);s.commit(tasks[0].task_id,result(0))
    s.conn.execute('delete from tasks');s.conn.commit();s.close()
    with pytest.raises(BulkMappingError,match='ORPHAN_ROW'):
        MappingStore(p,{},tasks)


def test_malformed_zlib_refused(tmp_path):
    p=tmp_path/'mapping.sqlite3';tasks=[task(0)]
    s=MappingStore(p,{},tasks);s.commit(tasks[0].task_id,result(0))
    s.conn.execute("update witnesses set compressed=x'000102'");s.conn.commit();s.close()
    with pytest.raises(BulkMappingError,match='WITNESS_BYTES_INVALID'):
        MappingStore(p,{},tasks)


def test_constructor_failure_closes_connection(tmp_path,monkeypatch):
    import infinity_grid.bulk_mapping_store as module
    real=sqlite3.connect;connections=[]
    def connect(*a,**kw):
        c=real(*a,**kw);connections.append(c);return c
    monkeypatch.setattr(module.sqlite3,'connect',connect)
    with pytest.raises(BulkMappingError,match='DUPLICATE_TASK'):
        MappingStore(tmp_path/'mapping.sqlite3',{},[task(0),task(0)])
    with pytest.raises(sqlite3.ProgrammingError,match='closed'):
        connections[0].execute('select 1')


def test_forced_witness_digest_collision_refuses_changed_bytes(tmp_path,monkeypatch):
    import infinity_grid.bulk_mapping_store as module
    tasks=[task(0),task(1)];s=MappingStore(tmp_path/'mapping.sqlite3',{},tasks)
    first=result(0);s.commit(tasks[0].task_id,first)
    different={'other':'witness'};different_bytes=enc(different)
    fixed=sha(json.loads(first['signature_canonical_bytes'])['rows'][0]['relabeling_witness'])
    real=module.hashlib.sha256
    class Forced:
        def hexdigest(self):return fixed
    def digest(raw=b''):
        return Forced() if raw==different_bytes else real(raw)
    changed=result(1,different);signature=json.loads(changed['signature_canonical_bytes'])
    signature['rows'][0]['relabeling_witness_sha256']=fixed
    changed['signature_canonical_bytes']=enc(signature);changed['signature_sha256']=sha(signature)
    monkeypatch.setattr(module.hashlib,'sha256',digest)
    with pytest.raises(BulkMappingError,match='WITNESS_HASH_COLLISION'):
        s.commit(tasks[1].task_id,changed)
    assert s.conn.execute('select count(*) from tasks').fetchone()[0]==1
    assert s.conn.execute('select count(*) from rows').fetchone()[0]==1
    s.close()


def test_task_verification_uses_task_ordinal_index(tmp_path):
    s=MappingStore(tmp_path/'mapping.sqlite3',{},[task(0)])
    plan=s.conn.execute('explain query plan select ordinal from rows where task_id=? order by ordinal',('t0000',)).fetchall()
    assert any('rows_task_ordinal' in r[-1] and 'SEARCH' in r[-1] for r in plan)
    s.close()
