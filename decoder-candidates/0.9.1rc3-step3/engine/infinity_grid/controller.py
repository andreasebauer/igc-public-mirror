from __future__ import annotations
import json, os, resource, shutil, time
from pathlib import Path
from typing import Callable
from .canon import canonical_sha256, write_json_atomic
from .paths import IGPaths
from .store import ArtifactStore
from .datasets import DatasetStore
from .checkpoints import RunLock, CheckpointManager, CorruptCheckpoint
from .records import runtime_sha256, source_sha256, utc_now, new_run_record
from .protocols import ProtocolRegistry
from .lifecycle import ExecutionModeManager
from .verification import immutable_run_core, seal_run_core
from .safety import contained_path, validate_identifier
from .v05 import V05AdmissionService, V05StageContext, register_runner_policy, runner_policy
from .v05_worker import run_isolated_stage
from .v05_evidence import seal_execution_envelope
from .v05_packaging import initialize_packaging_state
from .v05_runtime import V05TelemetryLedger

RUNNERS: dict[str,Callable] = {}
RUNNER_META: dict[str,dict] = {}

def _append_event(run_dir:Path,event:str,**fields):
    rec={'event':event,'utc':utc_now(),**fields}; p=Path(run_dir)/'events.jsonl'; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf-8') as h:
        h.write(json.dumps(rec,sort_keys=True,separators=(',',':'))+'\n'); h.flush()
        try: os.fsync(h.fileno())
        except OSError: pass

def register_runner(name, *, v05_operation=None, call_style="legacy", dataset_param="fixture_dataset_sha256"):
    def deco(fn):
        RUNNERS[name]=fn
        RUNNER_META[name]={"v05_operation":v05_operation,"call_style":call_style}
        if v05_operation is not None:
            register_runner_policy(runner=name,operation=v05_operation,call_style=call_style,dataset_param=dataset_param)
        return fn
    return deco

class Controller:
    def __init__(self,paths:IGPaths):
        self.paths=paths.ensure(); self.store=ArtifactStore(paths.store); self.datasets=DatasetStore(self.store); self.registry=ProtocolRegistry(); self.registry.install_into_store(paths); self.modes=ExecutionModeManager(paths); self.v05=V05AdmissionService(paths.store)
    def validate_plan(self,plan:dict,*,phase1=None,execution_mode:dict|None=None):
        if plan.get('schema_id')!='IG_EXECUTION_PLAN_V0_17': raise ValueError('bad plan schema')
        validate_identifier(plan.get('run_id',''),field='run_id')
        for st in plan.get('stages',[]): validate_identifier(st.get('stage_id',''),field='stage_id')
        self.registry.verify_plan(plan,paths=self.paths)
        descriptor=self.registry.get(plan['protocol_id'],version=plan.get('protocol_version'),descriptor_sha=plan.get('descriptor_sha256'))
        mode=self.modes.check_plan(descriptor,plan,execution_mode or self.modes.active())
        base={k:v for k,v in plan.items() if k!='plan_sha256'}; observed=canonical_sha256(base)
        if plan.get('plan_sha256')!=observed: raise ValueError('plan hash mismatch')
        ids=[s['stage_id'] for s in plan['stages']]
        if len(ids)!=len(set(ids)): raise ValueError('duplicate stage id')
        seen=set()
        for st in plan['stages']:
            if any(d not in seen for d in st.get('depends_on',[])): raise ValueError(f"stage order is not topological: {st['stage_id']}")
            seen.add(st['stage_id'])
        return observed,mode
    def _input_refs(self,plan):
        out=[]
        for d in plan.get('input_datasets',[]):
            v=self.datasets.verify(d['dataset_sha256'])
            if v['status']!='PASS': raise RuntimeError(f'input dataset failed verification {v}')
            out.append({'dataset_sha256':d['dataset_sha256'],'logical_role':d.get('logical_role')})
        return out
    def _load_or_create_envelope(self,plan,plan_sha,mode,input_refs,code_sha,env_sha):
        run_id=plan['run_id']; run_dir=contained_path(self.paths.runs,run_id,field='run_id'); run_dir.mkdir(parents=True,exist_ok=True)
        plan_path=run_dir/'plan.json'; question_path=run_dir/'question.json'; run_path=run_dir/'run.json'; core_path=run_dir/'run_core.json'
        if plan_path.exists():
            old=json.loads(plan_path.read_text()); oldsha=canonical_sha256({k:v for k,v in old.items() if k!='plan_sha256'})
            if oldsha!=plan_sha or old!=plan: raise RuntimeError('run root already bound to different immutable plan')
        else: write_json_atomic(plan_path,plan)
        if question_path.exists():
            if json.loads(question_path.read_text())!=plan['question']: raise RuntimeError('run root question mismatch')
        else: write_json_atomic(question_path,plan['question'])
        desc=self.registry.get(plan['protocol_id'],version=plan.get('protocol_version'),descriptor_sha=plan.get('descriptor_sha256')); core=seal_run_core(immutable_run_core(plan=plan,descriptor=desc,input_refs=input_refs,execution_mode=mode,code_sha=code_sha,env_sha=env_sha))
        if core_path.exists():
            old=json.loads(core_path.read_text())
            if old!=core: raise RuntimeError('immutable run core mismatch')
        else: write_json_atomic(core_path,core)
        if run_path.exists():
            rr=json.loads(run_path.read_text())
            mirror={'run_id':rr.get('run_id'),'protocol':rr.get('protocol'),'subject':rr.get('subject'),'question_sha256':rr.get('question_sha256'),'plan_sha256':rr.get('plan_sha256'),'evidence':rr.get('evidence'),'input_artifacts':rr.get('input_artifacts'),'execution_mode':rr.get('execution_mode')}
            for k,v in mirror.items():
                if core.get(k)!=v: raise RuntimeError(f'run record immutable field tampered: {k}')
            if rr.get('run_core_sha256')!=core['run_core_sha256']: raise RuntimeError('run record core pointer mismatch')
            if rr.get('code_identity',{}).get('source_sha256')!=code_sha: raise RuntimeError('run root code identity changed; allocate a new run_id')
            if rr.get('environment_identity',{}).get('runtime_sha256')!=env_sha: raise RuntimeError('run root environment identity changed; allocate a new run_id')
        else:
            rr=new_run_record(run_id=run_id,protocol=core['protocol'],subject=plan['subject'],question_sha256=plan['question_sha256'],plan_sha256=plan_sha,input_artifacts=input_refs,evidence_record=plan['evidence'],run_core_sha256=core['run_core_sha256'],execution_mode=mode,code_sha256=code_sha,environment_sha256=env_sha); write_json_atomic(run_path,rr)
        return run_dir,run_path,rr,core
    def run(self,plan:dict,*,phase1=None,execution_mode:dict|None=None):
        from .invocation import InvocationRefused
        raise InvocationRefused("REGISTERED_COMMAND_REQUIRED", "historical-controller")
        # source_sha256() live-verifies executing package bytes against build identity.
        # All run-root mutation, including first-launch envelope creation, occurs under the
        # single-writer lock.  Read-only plan/input validation may happen before the lock.
        code_sha=source_sha256(); env_sha=runtime_sha256(); plan_sha,mode=self.validate_plan(plan,phase1=phase1,execution_mode=execution_mode); v05_service=getattr(self,'v05',None); admissions=v05_service.admit_plan(plan=plan,code_sha=code_sha,env_sha=env_sha) if v05_service is not None else {}; input_refs=self._input_refs(plan); run_id=plan['run_id']
        run_dir=contained_path(self.paths.runs,run_id,field='run_id')
        with RunLock(run_dir):
            run_dir,run_path,rr,core=self._load_or_create_envelope(plan,plan_sha,mode,input_refs,code_sha,env_sha); cpman=CheckpointManager(run_dir,self.store)
            p3_ledger = None
            if admissions:
                initialize_packaging_state(run_dir)
                p3_ledger = V05TelemetryLedger(run_dir/'v05_telemetry'/'p3-controller-spans.jsonl')
            rr['lifecycle']='RUNNING'; rr['started_utc']=rr.get('started_utc') or utc_now(); rr.pop('error',None); write_json_atomic(run_path,rr); _append_event(run_dir,'RUN_STARTED',run_id=run_id,plan_sha256=plan_sha,run_core_sha256=core['run_core_sha256'])
            all_stage_resources=[]; result_artifacts=[]
            try:
                for st in plan['stages']:
                    sid=st['stage_id']; deps=cpman.dependency_bindings(st.get('depends_on',[]))
                    identity={'plan_sha256':plan_sha,'code_sha256':code_sha,'environment_sha256':env_sha,'run_core_sha256':core['run_core_sha256'],'stage_spec_sha256':canonical_sha256(st),'input_artifacts':input_refs,'dependency_bindings':deps}
                    reuse=cpman.reuse_status(sid,identity)
                    if reuse['status']=='REUSABLE':
                        cp=reuse['checkpoint']; rr['stages']=[x for x in rr.get('stages',[]) if x.get('stage_id')!=sid]+[{'stage_id':sid,'status':'COMPLETE_VALID','attempt':cp['attempt'],'reused':True}]; result_artifacts.extend(cp.get('output_artifacts',[])); all_stage_resources.append(cp.get('resource_usage',{})); write_json_atomic(run_path,rr); _append_event(run_dir,'STAGE_REUSED',stage_id=sid,attempt=cp['attempt']); continue
                    runner=RUNNERS.get(st['runner'])
                    if runner is None: raise KeyError(f"runner not registered: {st['runner']}")
                    admission=admissions.get(sid)
                    work=contained_path(self.paths.workspace,run_id,field='run_id')/sid; work=work.resolve(strict=False)
                    try: work.relative_to(self.paths.workspace.resolve())
                    except ValueError: raise RuntimeError('workspace path escape')
                    if admission is not None and admission.execution_policy is not None:
                        # P2.4 durable logical-task state lives below this stage workspace;
                        # the worker runner refreshes ephemeral control/input/output trees while
                        # preserving the append-only _task_state across controller resume.
                        work.mkdir(parents=True,exist_ok=True)
                    else:
                        shutil.rmtree(work,ignore_errors=True); work.mkdir(parents=True,exist_ok=True)
                    _append_event(run_dir,'STAGE_STARTED',stage_id=sid,runner=st['runner'])
                    t0=time.perf_counter(); c0=time.process_time();
                    if admission is not None:
                        _append_event(run_dir,'V05_STAGE_ADMITTED',stage_id=sid,runner=st['runner'],registration_sha256=admission.registration_sha256,operation=admission.operation,contract_version=admission.contract_version)
                        if admission.worker_policy is not None:
                            _append_event(run_dir,'V05_WORKER_STARTED',stage_id=sid,runner=st['runner'],worker_uid=admission.worker_policy['uid'],worker_gid=admission.worker_policy['gid'])
                            res,worker_meta=run_isolated_stage(
                                admission=admission,datasets=self.datasets,plan=plan,stage=st,work_dir=work,dependency_bindings=deps,
                                heartbeat_callback=lambda hb: _append_event(run_dir,'V05_HEARTBEAT',stage_id=sid,invocation_id=hb['invocation_id'],heartbeat_index=hb['heartbeat_index'],elapsed_seconds=hb['elapsed_seconds'],liveness=hb.get('liveness'),science_progress=hb.get('science_progress'),no_progress=hb.get('no_progress'),worker_rss_bytes_sampled=hb.get('worker_rss_bytes_sampled'),worker_cpu_seconds_procfs=hb.get('worker_cpu_seconds_procfs')),
                                heartbeat_path=run_dir/'v05_heartbeat.json', telemetry_dir=run_dir/'v05_telemetry'
                            )
                            sr=dict(res.get('stage_result',{}))
                            sr['worker_boundary']={
                                'kind':'POSIX_DROP_PRIVILEGE','uid':worker_meta['worker_uid'],'gid':worker_meta['worker_gid'],
                                'pid':worker_meta['worker_pid'],'job_sha256':worker_meta['job_sha256'],'isolated_process':True,
                                'store_write_access':'DENIED_BY_POSIX' if worker_meta['store_write_denied'] else 'UNSAFE',
                                'publication_write_access':'DENIED_BY_POSIX' if worker_meta['publication_write_denied'] else 'UNSAFE',
                                'terminal_telemetry_sha256':worker_meta.get('telemetry',{}).get('telemetry_sha256'),
                                'heartbeat_count':worker_meta.get('telemetry',{}).get('heartbeat_count'),
                                'task_scope_sha256':worker_meta.get('task_scope_sha256'),
                            }
                            res=dict(res,stage_result=sr)
                            _append_event(run_dir,'V05_WORKER_COMPLETE',stage_id=sid,runner=st['runner'],worker_uid=worker_meta['worker_uid'],worker_gid=worker_meta['worker_gid'],worker_pid=worker_meta['worker_pid'],job_sha256=worker_meta['job_sha256'],terminal_status=worker_meta.get('telemetry',{}).get('terminal_status'),heartbeat_count=worker_meta.get('telemetry',{}).get('heartbeat_count'),telemetry_sha256=worker_meta.get('telemetry',{}).get('telemetry_sha256'))
                            if p3_ledger is not None:
                                p3_ledger.record('SCHEDULING',component='controller.dispatch_and_wait',wall_seconds=time.perf_counter()-t0,process_cpu_seconds=time.process_time()-c0,details={'stage_id':sid,'runner':st['runner'],'includes_worker_wait':True,'process_transport_policy':worker_meta.get('process_transport_policy')})
                        else:
                            ctx=V05StageContext(service=self.v05,admission=admission,datasets=self.datasets,work_dir=work)
                            res=runner(context=ctx,plan=plan,stage=st)
                    else:
                        res=runner(controller=self,plan=plan,stage=st,work_dir=work,run_dir=run_dir)
                    outputs=[]; bytes_written=0
                    for logical,p in sorted(res.get('outputs',{}).items()):
                        rec=self.store.put_file(p,logical_role='RUN_RESULT',source_name=logical,created_by_run_id=run_id); rec=dict(rec); rec['logical_name']=logical; outputs.append(rec); bytes_written+=rec['size_bytes']
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; workspace_peak=sum(p.stat().st_size for p in work.rglob('*') if p.is_file()); bytes_read=sum(s['size_bytes'] for x in plan.get('input_datasets',[]) for s in self.datasets.load(x['dataset_sha256'])['shards'])
                    controller_cpu=time.process_time()-c0
                    worker_cpu=0.0; worker_peak=0; cpu_semantics='CONTROLLER_ONLY_NO_ISOLATED_WORKER'
                    if admission is not None and admission.worker_policy is not None:
                        wr=worker_meta.get('worker_resource_self',{})
                        tel=worker_meta.get('telemetry',{})
                        self_cpu=None
                        if wr.get('user_cpu_seconds') is not None and wr.get('system_cpu_seconds') is not None:
                            self_cpu=float(wr['user_cpu_seconds'])+float(wr['system_cpu_seconds'])
                        job_cpu=tel.get('job_cpu_seconds_procfs_last_observed')
                        if job_cpu is not None:
                            worker_cpu=max(float(job_cpu), float(self_cpu or 0.0)); cpu_semantics='CONTROLLER_PLUS_SAMPLED_WORKER_PROCESS_TREE'
                        elif self_cpu is not None:
                            worker_cpu=float(self_cpu); cpu_semantics='CONTROLLER_PLUS_CLEAN_EXIT_WORKER_SELF_FALLBACK'
                        worker_peak=int(tel.get('peak_job_rss_bytes_observed') or wr.get('maxrss_bytes') or tel.get('peak_worker_rss_bytes_observed') or 0)
                    controller_lifetime_peak=int(rss*1024)
                    ru={'wall_seconds':time.perf_counter()-t0,'cpu_seconds':controller_cpu+worker_cpu,'controller_cpu_seconds':controller_cpu,'worker_cpu_seconds':worker_cpu,'cpu_accounting_semantics':cpu_semantics,'peak_rss_bytes':max(controller_lifetime_peak,worker_peak),'controller_lifetime_maxrss_bytes':controller_lifetime_peak,'worker_peak_rss_bytes':worker_peak,'peak_rss_semantics':'MAX_OF_PROCESS_PEAKS_NOT_CONCURRENT_SUM','bytes_read':bytes_read,'bytes_written':bytes_written,'workspace_peak_bytes':workspace_peak,'checkpoint_bytes':0}
                    cp_t0=time.perf_counter(); cp_c0=time.process_time()
                    cp=cpman.publish(sid,identity=identity,input_artifacts=input_refs,output_artifacts=outputs,dependencies=st.get('depends_on',[]),resource_usage=ru,stage_result=res.get('stage_result',{}))
                    if p3_ledger is not None:
                        p3_ledger.record('CHECKPOINT_IO',component='controller.stage_checkpoint_publish',wall_seconds=time.perf_counter()-cp_t0,process_cpu_seconds=time.process_time()-cp_c0,details={'stage_id':sid,'attempt':cp['attempt']})
                    ru=cp['resource_usage']; rr['stages']=[x for x in rr.get('stages',[]) if x.get('stage_id')!=sid]+[{'stage_id':sid,'status':'COMPLETE_VALID','attempt':cp['attempt'],'reused':False}]; result_artifacts.extend(outputs); all_stage_resources.append(ru); write_json_atomic(run_path,rr); _append_event(run_dir,'STAGE_COMPLETE',stage_id=sid,attempt=cp['attempt'],outputs=len(outputs))
                seen=set(); ded=[]
                for a in result_artifacts:
                    key=(a['sha256'],a.get('logical_name'))
                    if key not in seen: seen.add(key); ded.append(a)
                rr['result_artifacts']=ded; rr['lifecycle']='COMPLETE_VALID'; rr['finished_utc']=utc_now(); rr['resource_summary']={k:sum(float(r.get(k,0) or 0) for r in all_stage_resources) for k in ['wall_seconds','cpu_seconds','bytes_read','bytes_written','checkpoint_bytes']}; rr['resource_summary']['workspace_peak_bytes']=max([int(r.get('workspace_peak_bytes',0) or 0) for r in all_stage_resources] or [0]); rr['resource_summary']['peak_rss_bytes']=max([int(r.get('peak_rss_bytes',0) or 0) for r in all_stage_resources] or [0]); write_json_atomic(run_path,rr); _append_event(run_dir,'RUN_COMPLETE',run_id=run_id,lifecycle='COMPLETE_VALID')
                if admissions:
                    reg_sha=next(iter(admissions.values())).registration_sha256
                    env=seal_execution_envelope(self.paths,run_id,reg_sha)
                    _append_event(run_dir,'V05_EXECUTION_ENVELOPE_SEALED',run_id=run_id,envelope_sha256=env['envelope_sha256'],publication_state=env['publication_state'])
                return rr
            except CorruptCheckpoint as e:
                rr['lifecycle']='CORRUPT'; rr['finished_utc']=utc_now(); rr['error']=str(e); write_json_atomic(run_path,rr); _append_event(run_dir,'RUN_CORRUPT',run_id=run_id,error=str(e)); raise
            except Exception as e:
                rr['lifecycle']='FAILED'; rr['finished_utc']=utc_now(); rr['error']=f'{type(e).__name__}: {e}'; write_json_atomic(run_path,rr); _append_event(run_dir,'RUN_FAILED',run_id=run_id,error=rr['error']); raise
    def resume(self,run_id,*,phase1=None,execution_mode:dict|None=None):
        from .invocation import InvocationRefused
        raise InvocationRefused("REGISTERED_COMMAND_REQUIRED", "historical-controller")
        validate_identifier(run_id,field='run_id'); p=contained_path(self.paths.runs,run_id,field='run_id')/'plan.json'
        if not p.is_file(): raise FileNotFoundError(p)
        return self.run(json.loads(p.read_text()),phase1=phase1,execution_mode=execution_mode)


"""Stable 0.6 command name for Decoder's native controller implementation."""
import sys


def main(argv=None):
    from .v05_controller_event_loop import main as native_main
    if argv is None: return native_main()
    previous = sys.argv
    try:
        sys.argv = ['ig-decoder', *argv]
        return native_main()
    finally: sys.argv = previous


if __name__ == '__main__':
    raise SystemExit(main())
