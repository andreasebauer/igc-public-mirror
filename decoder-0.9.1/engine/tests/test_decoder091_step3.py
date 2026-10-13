"""Engineering resource and immutable-context checks; no official science run."""
from pathlib import Path
import dataclasses,json,os,subprocess,time
import pytest
from infinity_grid import execution as ex,preservation as pr,runtime_capacity as cap
from infinity_grid import o7_edge_canonicalization as kernel
from infinity_grid import bulk_descriptor_plan as planning
from infinity_grid import v05_controller_event_loop as loop
from infinity_grid.canon import canonical_bytes,canonical_sha256
from infinity_grid.structural_encoding import structural_canonical_bytes as encode
from infinity_grid.isolated_runtime import subprocess_argv,subprocess_environment
from tests.test_decoder09_implementation import ctx,EDGE


def test_shared_native_environments_respect_one_resource_domain(tmp_path,monkeypatch):
    if ex.detect_effective_cpu_count(ignore_override=True)<2:pytest.skip('Two CPU equivalents required')
    monkeypatch.setenv('IG_DECODER_LEASE_ROOT',str(tmp_path/'shared'))
    resources={'workers':2,'start_method':'spawn','execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'}
    with loop._workspace_environment(tmp_path/'captureA/runtime',resources):
        first=ex.HostCpuLeaseManager(budget=2);lease=first.acquire(2,owner='captureA',wait=False)
    try:
        with loop._workspace_environment(tmp_path/'captureB/runtime',resources):
            second=ex.HostCpuLeaseManager(budget=2)
            assert first.root==second.root==tmp_path/'shared'
            with pytest.raises(ex.CpuLeaseError):second.acquire(2,owner='captureB',wait=False)
            assert second.status()['used']==2
    finally:lease.release()


def test_default_domain_and_physical_capacity_ignore_capture_request(monkeypatch):
    monkeypatch.delenv('IG_DECODER_LEASE_ROOT',raising=False)
    root=ex.shared_lease_root();physical=ex.detect_effective_cpu_count(ignore_override=True)
    monkeypatch.setenv('IG_DECODER_CPU_BUDGET','1')
    assert ex.detect_effective_cpu_count()==1
    assert cap.shared_capacity_snapshot()['cpu_core_equivalents']==physical
    assert ex.HostCpuLeaseManager().root==root
    assert ex.HostCpuLeaseManager().budget==physical


def test_actual_killed_lease_owner_is_reaped_and_child_memory_counted(tmp_path):
    ready=tmp_path/'ready';root=tmp_path/'leases'
    script="""
import sys,time,json
from pathlib import Path
from infinity_grid.execution import HostCpuLeaseManager,_lease_process_identity
allocated=bytearray(8*1024**2)
lease=HostCpuLeaseManager(sys.argv[1]).acquire(1,owner='ENG:KILLED',wait=False)
Path(sys.argv[2]).write_text(json.dumps({'proc_pid':_lease_process_identity()[0]}))
time.sleep(30)
"""
    child=subprocess.Popen(subprocess_argv(['-B','-c',script,str(root),str(ready)]),env=subprocess_environment())
    try:
        deadline=time.monotonic()+10
        while not ready.exists():
            if child.poll() is not None:raise AssertionError('child exited early')
            if time.monotonic()>deadline:raise AssertionError('child did not signal ready')
            time.sleep(0.01)
        pid=json.loads(ready.read_bytes())['proc_pid'];snapshot=cap.process_tree_snapshot()
        assert pid in [p['pid'] for p in snapshot['processes']]
        sample=next(p for p in snapshot['processes'] if p['pid']==pid)
        assert sample['accounted_memory_bytes']>=8*1024**2
        assert ex.HostCpuLeaseManager(root).status()['used']==1
    finally:child.kill();child.wait()
    assert ex.HostCpuLeaseManager(root).status()['used']==0
    with ex.HostCpuLeaseManager(root).acquire(1,owner='ENG:RESUME',wait=False):pass


def test_same_process_working_set_reservations_are_not_double_counted(tmp_path):
    if ex.detect_effective_cpu_count(ignore_override=True)<2:pytest.skip('Two CPU equivalents required')
    manager=ex.HostCpuLeaseManager(tmp_path,budget=2)
    with manager.acquire(1,owner='A',wait=False) as a,manager.acquire(1,owner='B',wait=False) as b:
        snapshot={'accounted_memory_bytes':123456,'memory_accounting':'TEST_PSS'}
        manager.account_memory(a.record['lease_id'],snapshot)
        assert manager.account_memory(b.record['lease_id'],snapshot)['observed_reserved_bytes']==123456


def test_proc_namespace_root_and_preservation_small_budget_refuse(tmp_path):
    snapshot=cap.process_tree_snapshot()
    assert snapshot['root_pid']==int(Path('/proc/self/stat').read_text().split(' ',1)[0])
    assert 0<snapshot['accounted_memory_bytes']<=snapshot['aggregate_rss_bytes']
    active={'workspace':str(tmp_path),'pid':os.getpid(),'last_poll':0,
            'admission':{'workspace':str(tmp_path),'job':{'resources':{'workspace_budget_bytes':None,'memory_budget_bytes':1}}}}
    token=pr._ACTIVE.set(active)
    try:
        with pytest.raises(Exception,match='REGISTERED_MEMORY_BUDGET'):pr.poll()
    finally:pr._ACTIVE.reset(token)


def test_live_memory_missing_refuses(tmp_path,monkeypatch):
    monkeypatch.setattr(cap,'_process_memory',lambda p:(_ for _ in ()).throw(OSError('NO_TELEMETRY')))
    with pytest.raises(RuntimeError,match='LIVE_PROCESS_MEMORY_UNKNOWN'):cap.process_tree_snapshot()


def test_rss_fallback_reports_conservative_accounting(tmp_path):
    (tmp_path/'statm').write_text('2 1\n')
    rss,value,method=cap._process_memory(tmp_path)
    assert rss==value==os.sysconf('SC_PAGE_SIZE') and method=='RSS_FALLBACK'


def test_immutable_context_copies_and_generic_mutation_guards_survive():
    mutable=ctx();kernel.reset_hierarchy_cache();frozen=kernel.bind_immutable_context(mutable)
    first=kernel.state_identity(frozen,(EDGE,))
    assert encode(first)==canonical_bytes(first)
    with pytest.raises(dataclasses.FrozenInstanceError):frozen.n=3
    with pytest.raises(TypeError):frozen.hierarchy[3][0]=()
    leaf=mutable.parents[0].h6
    for _ in range(5):leaf=leaf[0]
    leaf[1][1]=1
    assert encode(kernel.state_identity(frozen,(EDGE,)))==encode(first)
    assert kernel.state_identity(mutable,(EDGE,))['object_identity']!=first['object_identity']
    assert kernel.verify_relabeling_witness(frozen,(EDGE,),first['edges'],first['witness'])
    kernel.reset_hierarchy_cache(enabled=False)
    assert encode(kernel.state_identity(frozen,(EDGE,)))==encode(first)
    kernel.reset_hierarchy_cache()


def test_frozen_hierarchy_never_reserializes_per_row(monkeypatch):
    frozen=kernel.bind_immutable_context(ctx())
    from infinity_grid import structural_encoding
    real=structural_encoding.structural_canonical_bytes;calls=[]
    def tracked(value):
        if type(value) is dict and set(value)=={'n','parents'}:calls.append(value)
        return real(value)
    monkeypatch.setattr(structural_encoding,'structural_canonical_bytes',tracked)
    for _ in range(3):kernel.state_identity(frozen,(EDGE,))
    assert calls==[]


def test_frozen_plan_is_bound_to_independent_expected_digest():
    plan=json.loads(Path(planning.__file__).with_name('bulk_mapping_plan.json').read_bytes())
    bare=dict(plan);del bare['plan_sha256']
    assert canonical_sha256(bare)==planning.FROZEN_PLAN_SHA256==plan['plan_sha256']


def test_auto_lease_honors_job_budget_without_shrinking_shared_host(tmp_path,monkeypatch):
    monkeypatch.setenv('IG_DECODER_CPU_BUDGET','1')
    manager=ex.HostCpuLeaseManager(tmp_path)
    physical=ex.detect_effective_cpu_count(ignore_override=True)
    with manager.acquire('AUTO',owner='CAPTURE:ONE',wait=False) as lease:
        assert lease.workers==1 and manager.status()['budget']==physical


def test_bulk_live_counts_fault_and_missing_only_resume(tmp_path,monkeypatch):
    """Inert scheduler fixture; actual SQLite/status code, no native authority claim."""
    from types import SimpleNamespace
    from infinity_grid import v05_stage_runtime as stage
    tasks=[ex.TaskSpec(f't{i}','ENG.INERT','a'*64,{'descriptors':[{'ordinal':i,'edge':[i],'literal_descriptor_sha256':'d'*64}]}) for i in range(2)]
    class Fixture:
        chain_id='ENG';stage_id='INERT';question_sha256='e'*64;default_workers=1
        _root=tmp_path;_project_callable_bindings={}
        _execution_permit=SimpleNamespace(require_evaluator=lambda ref:None)
        run_exact_bulk_mapping=stage.StageScienceRuntime.run_exact_bulk_mapping
        _status=stage.StageScienceRuntime._status
        _finalize_idle_partition_database=staticmethod(stage.StageScienceRuntime._finalize_idle_partition_database)
        def _require_execution(self):pass
        def _phase_root(self,phase):
            p=self._root/phase;p.mkdir(exist_ok=True);return p
        def _enforce_execution_budgets(self,root):self._last_resource_snapshot={'engineering_fixture':True}
    monkeypatch.setattr(stage,'_resolve_ref',lambda *a,**k:lambda p:None)
    monkeypatch.setattr(stage,'require_controller_only_callable',lambda *a,**k:None)
    dispatched=[]
    def stream(pending,**kwargs):
        kwargs['telemetry'].set_workers(leased=1,active=1)
        for task in pending:
            dispatched.append(task.task_id);row=task.payload['descriptors'][0];witness={'nodes':[0]}
            signature={'schema':'IG_HET4_FULL_WITNESS_BATCH_V1','rows':[{'ordinal':row['ordinal'],'raw_edge':row['edge'],
                'literal_descriptor_sha256':row['literal_descriptor_sha256'],'relabeling_witness':witness,'relabeling_witness_sha256':canonical_sha256(witness)}]}
            kwargs['on_poll']();kwargs['on_result'](task.task_id,{'signature_canonical_bytes':encode(signature),'signature_sha256':canonical_sha256(signature),
                'outcome_count':1,'metrics':{'engine_evaluator_cpu_seconds':0.01,'engine_signature_encode_wall_seconds':0.001}})
            kwargs['telemetry'].mark_completed(task_units=1)
        return {'workers':1,'task_count':len(pending)}
    monkeypatch.setattr(stage,'execute_tasks_stream',stream)
    runtime=Fixture()
    with monkeypatch.context() as m:
        m.setenv('IG_V05_ENGINEERING_ABORT_AFTER_COMMITS','1');m.setenv('IG_V05_ENGINEERING_FAULT_INJECTION_ACK','STAGE_RUNTIME_ABORT_TEST')
        with pytest.raises(stage.StageRuntimeError,match='ABORT_AFTER_1'):
            runtime.run_exact_bulk_mapping(phase_id='P',tasks=tasks,evaluator_ref='ENG:inert')
    status=json.loads((tmp_path/'P/RUNTIME_STATUS.json').read_bytes())
    assert status['status']=='FAILED' and status['rows_committed']==1 and status['rows_pending']==1
    dispatched.clear();completed=runtime.run_exact_bulk_mapping(phase_id='P',tasks=tasks,evaluator_ref='ENG:inert')
    assert dispatched==['t1']
    metrics=completed['summary']['operating_metrics']
    assert metrics['rows_reused']==1 and metrics['rows_committed']==2 and metrics['rows_pending']==0
    assert metrics['commit_seconds']>=0 and metrics['control_and_preservation_seconds']>=0
    live=json.loads((tmp_path/'P/telemetry/RUNTIME_STATUS.json').read_bytes())
    assert live['status']=='COMPLETE' and live['tasks_total']==1 and live['detail']['science_hash_excluded']
    dispatched.clear();reused=runtime.run_exact_bulk_mapping(phase_id='P',tasks=tasks,evaluator_ref='ENG:inert')
    assert dispatched==[] and reused['summary']['full_rows_stream_sha256']==completed['summary']['full_rows_stream_sha256']


def test_zero_startup_rss_is_observed_explicitly_not_missing_telemetry(tmp_path):
    (tmp_path/'statm').write_text('0 0 0 0 0 0 0\n')
    assert cap._process_memory(tmp_path)==(0,0,'ZERO_RESIDENT_PAGES_OBSERVED')


def test_zero_controller_measurement_is_refused(monkeypatch):
    monkeypatch.setattr(cap,'_process_memory',lambda proc:(0,0,'ZERO_RESIDENT_PAGES_OBSERVED'))
    with pytest.raises(RuntimeError,match='LIVE_PROCESS_MEMORY_UNKNOWN'):cap.process_tree_snapshot()
