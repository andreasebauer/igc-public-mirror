import json,os,sqlite3
from pathlib import Path
from types import SimpleNamespace
import pytest
from infinity_grid import o7_edge_canonicalization as kernel
from infinity_grid.structural_encoding import structural_canonical_bytes as encode
from infinity_grid.bulk_mapping_store import MappingStore,iter_encoded_rows,batch_bytes
from infinity_grid.runtime_capacity import process_rss_bytes,process_tree_snapshot,shared_capacity_snapshot


def ctx(free=2):
    h=([2]*7,[free]*7)
    for _ in range(5):h=[h]
    return SimpleNamespace(n=2,parents=[SimpleNamespace(exact_color='x',h6=h),SimpleNamespace(exact_color='x',h6=h)])
EDGE=(0,0,0,0,0,0,0,1,0,0,0,0,0,0)


def test_cached_uncached_exact_witness_identity_and_restart():
    c=ctx();kernel.reset_hierarchy_cache(enabled=False);uncached=kernel.state_identity(c,(EDGE,))
    assert kernel.verify_relabeling_witness(c,(EDGE,),uncached['edges'],uncached['witness'])
    kernel.reset_hierarchy_cache();cached=kernel.state_identity(c,(EDGE,))
    assert encode(cached)==encode(uncached)
    assert kernel.verify_relabeling_witness(c,(EDGE,),cached['edges'],cached['witness'])
    stats=kernel.hierarchy_cache_statistics();assert stats['builds']==1 and stats['hits']>=3
    kernel.reset_hierarchy_cache();assert encode(kernel.state_identity(c,(EDGE,)))==encode(uncached)


def test_context_mutation_invalidation_and_eviction():
    c=ctx();kernel.reset_hierarchy_cache();first=kernel.state_identity(c,(EDGE,))
    leaf=c.parents[0].h6
    for _ in range(5):leaf=leaf[0]
    leaf[1][1]=1
    second=kernel.state_identity(c,(EDGE,));assert first['object_identity']!=second['object_identity']
    assert kernel.hierarchy_cache_statistics()['invalidations']==1
    other=ctx(1);kernel.state_identity(other,(EDGE,));assert kernel.hierarchy_cache_statistics()['entries']==1
    kernel.reset_hierarchy_cache(enabled=False);assert encode(kernel.state_identity(c,(EDGE,)))==encode(second)
    kernel.reset_hierarchy_cache()


def test_resource_accounting_never_reports_unknown_as_zero():
    own=process_rss_bytes(os.getpid());snapshot=process_tree_snapshot()
    assert own>0 and snapshot['aggregate_rss_bytes']>=own
    cap=shared_capacity_snapshot();assert cap['memory_ceiling_bytes']>own and cap['cpu_core_equivalents']>=1
    with pytest.raises((OSError,RuntimeError)):process_rss_bytes(2**30)


def test_component_composition_matches_original_canonical_bytes(tmp_path):
    from infinity_grid.execution import TaskSpec
    from infinity_grid.canon import canonical_sha256
    witness={'odd':'ä"\\','nodes':[[0,[],0,[]]]}
    rows=[{'ordinal':0,'raw_edge':[0],'literal_descriptor_sha256':'d'*64,'relabeling_witness':witness,'relabeling_witness_sha256':canonical_sha256(witness),'a':'π','z':'end'}]
    task=TaskSpec('t','TEST','a'*64,{'descriptors':[{'ordinal':0,'edge':[0],'literal_descriptor_sha256':'d'*64}]})
    signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':rows};raw=encode(signature)
    s=MappingStore(tmp_path/'m.sqlite3',{},[task]);s.commit('t',{'signature_canonical_bytes':raw,'signature_sha256':canonical_sha256(signature),'outcome_count':1})
    assert batch_bytes([v for _,v in iter_encoded_rows(s.conn)])==raw
    s.verify_task('t');s.close()


def test_sqlite_full_preserves_committed_task(tmp_path):
    from infinity_grid.execution import TaskSpec
    from infinity_grid.canon import canonical_sha256
    tasks=[TaskSpec(f't{i}','TEST','a'*64,{'descriptors':[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64}]}) for i in range(2)]
    s=MappingStore(tmp_path/'m.sqlite3',{},tasks)
    def result(i,size):
        witness={'full':True};rows=[{'ordinal':i,'raw_edge':[i],'literal_descriptor_sha256':'d'*64,'relabeling_witness':witness,'relabeling_witness_sha256':canonical_sha256(witness),'payload':'x'*size}]
        signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':rows}
        return {'signature_canonical_bytes':encode(signature),'signature_sha256':canonical_sha256(signature),'outcome_count':1}
    s.commit('t0',result(0,1));pages=s.conn.execute('pragma page_count').fetchone()[0];s.conn.execute(f'pragma max_page_count={pages}')
    with pytest.raises(sqlite3.OperationalError,match='full'):s.commit('t1',result(1,100000))
    assert s.conn.execute('select count(*) from tasks').fetchone()[0]==1
    s.verify_task('t0');s.close()
    restored=MappingStore(tmp_path/'m.sqlite3',{},tasks);assert [t.task_id for t in restored.pending()]==['t1'];restored.close()


def test_new_rolling_policy_removes_numerical_blockers_and_legacy_is_frozen():
    from infinity_grid.result_contracts import policy
    from infinity_grid.preservation import _check_backlog
    legacy=policy({});assert legacy['max_state_file_bytes']==33554432
    updated=policy({'preservation':{'rolling_unbounded_execution':1}})
    assert all(v is None for k,v in updated.items() if k.startswith('max_'))
    state={'pending_bytes':2**40,'pending_checkpoints':[1]*10000,'oldest_pending_age_seconds':10**9,'retained_checkpoint_count':100000}
    assert _check_backlog(state,updated,reserve=True) is state


def test_witness_cache_detects_corruption_after_prior_validation(tmp_path):
    from infinity_grid.execution import TaskSpec
    from infinity_grid.canon import canonical_sha256
    from infinity_grid.bulk_mapping_store import BulkMappingError
    witness={'full':[1,2,3]};rows=[{'ordinal':0,'raw_edge':[0],'literal_descriptor_sha256':'d'*64,'relabeling_witness':witness,'relabeling_witness_sha256':canonical_sha256(witness)}]
    t=TaskSpec('t','TEST','a'*64,{'descriptors':[{'ordinal':0,'edge':[0],'literal_descriptor_sha256':'d'*64}]});s=MappingStore(tmp_path/'m.sqlite3',{},[t]);signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':rows}
    s.commit('t',{'signature_canonical_bytes':encode(signature),'signature_sha256':canonical_sha256(signature),'outcome_count':1});s.verify_task('t')
    s.conn.execute("update witnesses set compressed=x'0001'");s.conn.commit()
    with pytest.raises(BulkMappingError,match='WITNESS_BYTES_INVALID'):s.verify_task('t')
    s.close()


def test_rolling_runner_refuses_unadopted_science(tmp_path,monkeypatch):
    from infinity_grid import bulk_submission as bulk,isolated_runtime as runtime,v05_controller_event_loop as loop
    monkeypatch.setattr(runtime,'verified_runtime',lambda:tmp_path)
    monkeypatch.setattr(loop,'_source_ids',lambda p:('source','package'))
    monkeypatch.setattr(bulk,'build_plan',lambda *a,**k:{'plan_sha256':'plan','preserved_rows':56832})
    with pytest.raises(ValueError,match='DATA_OWNER_ADOPTION_REQUIRED'):
        bulk.execute_adopted_windows(tmp_path/'run',tmp_path,tmp_path,'raw','prefix','sqlite',{},object())


def test_cpu_override_cannot_exceed_real_shared_quota(monkeypatch):
    from infinity_grid.execution import detect_effective_cpu_count
    monkeypatch.delenv('IG_DECODER_CPU_BUDGET',raising=False);actual=detect_effective_cpu_count()
    monkeypatch.setenv('IG_DECODER_CPU_BUDGET',str(actual*100));assert detect_effective_cpu_count()==actual


def test_namespace_safe_cpu_lease_liveness_and_release(tmp_path):
    from infinity_grid.execution import HostCpuLeaseManager,_lease_process_identity
    manager=HostCpuLeaseManager(tmp_path,budget=1)
    lease=manager.acquire(1,owner='IMPLEMENTATION_UNIT',wait=False)
    assert lease.record['pid']==_lease_process_identity()[0]
    assert lease.record['process_start']==_lease_process_identity()[1]
    assert manager.status()['used']==1
    lease.release();assert manager.status()['used']==0
