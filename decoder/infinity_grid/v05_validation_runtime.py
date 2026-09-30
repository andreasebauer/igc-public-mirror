from __future__ import annotations

"""Decoder-owned registered validation runtime (C5).

Validation declarations are data.  The controller chooses topology, forks the
validation children, and deterministically reduces their node outcomes.  pytest
is only a library inside controller-created children; it is not the scheduler.
"""

import contextlib, ctypes, hashlib, io, json, os, shutil, signal, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any, Iterable
from .canon import canonical_sha256
from .v05_execution_authority import ExecutionAuthorityError
from .v05_origin_guard import require_controller_execution_origin

REGISTERED_VALIDATION_GROUPS={
    "engineering_layer":[
        "historical_tests/test_v05_engineering_jobs.py::test_engineering_stage_handlers_pass_architecture_gate",
        "historical_tests/test_v05_engineering_jobs.py::test_engineering_source_digest_is_content_bound",
        "historical_tests/test_v05_engineering_jobs.py::test_overlay_manifest_rejects_unlisted_member",
        "historical_tests/test_v05_engineering_jobs.py::test_compact_validation_tail_keeps_all_failed_nodes",
        "tests/test_v05_engineering_same_identity_authorization.py",
    ],
    "l2_service":["tests/test_v05_registered_service_l2.py"],
    "l1_authority":["tests/test_v05_execution_authority_v03088.py"],
    "encoding_kernel":["tests/test_structural_encoding_e2.py","tests/test_engine_kernel_e1.py"],
    "l2_closeout":["tests/test_v05_l2_software_closeout.py"],
    "c5_parallel_probe":[
        "tests/test_v05_c5_origin_exclusivity.py::test_c5_parallel_probe_1",
        "tests/test_v05_c5_origin_exclusivity.py::test_c5_parallel_probe_2",
        "tests/test_v05_c5_origin_exclusivity.py::test_c5_parallel_probe_3",
        "tests/test_v05_c5_origin_exclusivity.py::test_c5_parallel_probe_4",
    ],
    "c5_final":["tests/test_v05_c5_origin_exclusivity.py"],
    "g6_s5r_support":["tests/test_g6_s5r_observer_continuity.py"],
    "c6_recovery":["tests/test_v05_c6_persistent_recovery.py"],
    "full_regression":["__ALL_TEST_FILES__"],
    "o3c_bench_s4":["tests/test_v05_o3c_benchmarks.py::test_o3c_benchmark_s4_relation_reuse"],
    "o3c_bench_s5":["tests/test_v05_o3c_benchmarks.py::test_o3c_benchmark_s5_q_decode_reuse"],
    "o3c_bench_s6":["tests/test_v05_o3c_benchmarks.py::test_o3c_benchmark_s6_write_child_reuse"],
}

class ValidationRuntimeError(RuntimeError): pass

def _compact(output:str,limit:int=12000)->str:
    lines=output.splitlines(); failed=[x for x in lines if x.startswith('FAILED ')]
    summaries=[x for x in lines if (' failed' in x or ' passed' in x or ' error' in x or ' skipped' in x) and (' in ' in x or x.startswith('='))]
    chosen=failed+summaries[-4:]
    if not chosen: chosen=lines[-50:]
    return '\n'.join(chosen)[-limit:]

def _benchmark_payloads(output:str)->list[dict[str,Any]]:
    """Retain benchmark records even when pytest prefixes progress markers."""
    marker='IG_BENCHMARK_JSON:'; out=[]
    for line in output.splitlines():
        at=line.find(marker)
        if at<0: continue
        try: out.append(json.loads(line[at+len(marker):]))
        except Exception: pass
    return out

def _nodes(candidate:Path,group:str)->list[str]:
    if group not in REGISTERED_VALIDATION_GROUPS: raise ValidationRuntimeError('VALIDATION_GROUP_NOT_REGISTERED:'+group)
    raw=REGISTERED_VALIDATION_GROUPS[group]
    if raw==['__ALL_TEST_FILES__']:
        return [p.relative_to(candidate).as_posix() for p in sorted((candidate/'tests').glob('test_*.py'))]
    return list(raw)

def _partition(nodes:list[str],workers:int)->list[list[str]]:
    if not nodes:return []
    n=max(1,min(int(workers),len(nodes)))
    out=[[] for _ in range(n)]
    for i,node in enumerate(nodes): out[i%n].append(node)
    return out

def _publish_worker_log(active_path:Path,final_path:Path)->dict[str,Any]:
    """Publish closed output without overwriting any previous worker evidence."""
    raw=active_path.read_bytes()
    # The stream is already flushed, fsynced and closed. Exclusive hard-link
    # publication keeps partial output separate and refuses an existing name.
    os.link(active_path,final_path)
    if final_path.read_bytes()!=raw:
        raise ValidationRuntimeError('VALIDATION_LOG_PUBLICATION_READBACK')
    active_path.unlink()
    return {'log_sha256':hashlib.sha256(raw).hexdigest(),'log_size_bytes':len(raw)}


def _verify_worker_log(log_path:Path,result:dict[str,Any])->None:
    """Parent readback must match the bytes acknowledged by the finished child."""
    raw=log_path.read_bytes()
    if (result.get('log_sha256')!=hashlib.sha256(raw).hexdigest()
            or result.get('log_size_bytes')!=len(raw)):
        raise ValidationRuntimeError('VALIDATION_WORKER_LOG_MISMATCH')


def _publish_node_output(final_path:Path,raw:bytes)->dict[str,Any]:
    """Create one immutable selector log from bytes captured without a pathname.

    The subprocess writes to a pipe owned by ``Popen``.  ``communicate`` does
    not return until every inherited writer has closed that pipe, so a forked
    descendant cannot recreate or append to a named staging file.  Publication
    is a fresh inode created only after the stream reaches EOF.
    """
    with final_path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    if final_path.read_bytes()!=raw:
        raise ValidationRuntimeError('VALIDATION_NODE_LOG_PUBLICATION_READBACK')
    return {'log_sha256':hashlib.sha256(raw).hexdigest(),'log_size_bytes':len(raw)}


def _run_node_process(command:list[str],*,cwd:Path,env:dict[str,str],final_path:Path
                     )->tuple[subprocess.CompletedProcess[bytes],bytes]:
    """Capture through an anonymous pipe, then publish exactly once."""
    proc=subprocess.Popen(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    raw,_=proc.communicate()
    _publish_node_output(final_path,raw)
    return subprocess.CompletedProcess(command,proc.returncode,raw,None),raw


def _enable_validation_subreaper()->None:
    """Make the worker own orphaned descendants created by validation nodes.

    Decoder's production validation runtime is Linux.  A direct waitpid loop is
    insufficient when a test child forks again and its immediate parent exits:
    without a subreaper the surviving grandchild is adopted outside the worker
    and may retain or recreate worker evidence after publication.
    """
    if not sys.platform.startswith('linux'):
        raise ValidationRuntimeError('VALIDATION_SUBREAPER_UNAVAILABLE')
    libc=ctypes.CDLL(None,use_errno=True)
    prctl=libc.prctl
    prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    prctl.restype=ctypes.c_int
    # Linux prctl(2): PR_SET_CHILD_SUBREAPER=36, PR_GET_CHILD_SUBREAPER=37.
    if prctl(36,1,0,0,0)!=0:
        raise ValidationRuntimeError('VALIDATION_SUBREAPER_SET_FAILED:'+str(ctypes.get_errno()))
    state=ctypes.c_int(0)
    if prctl(37,ctypes.addressof(state),0,0,0)!=0 or state.value!=1:
        raise ValidationRuntimeError('VALIDATION_SUBREAPER_VERIFY_FAILED:'+str(ctypes.get_errno()))


def _wait_for_validation_descendants(*, timeout_seconds:float=5.0)->None:
    """Reap forked test children before publishing immutable worker evidence.

    A validation node may exercise fork-based code.  If such a child survives
    the node, it retains the worker's redirected output descriptor and Python
    frame.  Publishing before that child exits permits late output—or a second
    publication attempt—to change the evidence set after completion is bound.
    """
    deadline=time.monotonic()+float(timeout_seconds)
    while True:
        try: pid,_status=os.waitpid(-1,os.WNOHANG)
        except ChildProcessError:return
        if pid:
            continue
        if time.monotonic()>=deadline:
            raise ValidationRuntimeError('VALIDATION_DESCENDANT_NOT_QUIESCENT')
        time.sleep(0.01)


def _evidence_baseline(root:Path,pattern:str)->dict[str,dict[str,Any]]:
    """Freeze immutable evidence already present at a resumed-run boundary."""
    return {p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                    'size_bytes':p.stat().st_size}
            for p in sorted(root.glob(pattern))}


def _verify_evidence_baseline(root:Path,baseline:dict[str,dict[str,Any]])->None:
    for name,record in baseline.items():
        path=root/name
        if (not path.is_file() or path.stat().st_size!=record['size_bytes']
                or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']):
            raise ValidationRuntimeError('VALIDATION_PRIOR_EVIDENCE_CHANGED:'+name)


def _verify_worker_evidence_set(log_root:Path,results:list[dict[str,Any]],*,
                                prior:dict[str,dict[str,Any]]|None=None)->None:
    """Bind current worker evidence while retaining an exact resume baseline."""
    prior={} if prior is None else dict(prior)
    _verify_evidence_baseline(log_root,prior)
    active=sorted(p.name for p in log_root.glob('worker-*.active'))
    actual=sorted(p.name for p in log_root.glob('worker-*.log'))
    bound=sorted(r.get('log_path') for r in results if r.get('log_path'))
    expected=sorted(set(prior)|set(bound))
    if set(prior)&set(bound) or active or actual!=expected:
        detail=json.dumps({'active':active,'actual':actual,'bound':bound,
                           'prior':sorted(prior),'expected':expected},sort_keys=True)
        raise ValidationRuntimeError('VALIDATION_WORKER_EVIDENCE_SET_MISMATCH:'+detail)


def _verify_node_evidence_set(log_root:Path,results:list[dict[str,Any]],*,
                              prior:dict[str,dict[str,Any]]|None=None)->None:
    """Bind current selector streams while retaining an exact resume baseline."""
    node_root=log_root/'node_process_logs'
    prior={} if prior is None else dict(prior)
    _verify_evidence_baseline(node_root,prior)
    active=sorted(p.name for p in node_root.glob('*.active')) if node_root.is_dir() else []
    actual=sorted(p.name for p in node_root.glob('*.log')) if node_root.is_dir() else []
    bound=[]
    for result in results:
        for position,node in enumerate(result.get('nodes',[])):
            bound.append(f'{position:06d}-{canonical_sha256(node)}.log')
    bound.sort()
    expected=sorted(set(prior)|set(bound))
    if set(prior)&set(bound) or active or actual!=expected:
        detail=json.dumps({'active':active,'actual':actual,'bound':bound,
                           'prior':sorted(prior),'expected':expected},sort_keys=True)
        raise ValidationRuntimeError('VALIDATION_NODE_EVIDENCE_SET_MISMATCH:'+detail)


def _child(candidate:Path,nodes:list[str],uid:int|None,gid:int|None,wfd:int,log_path:Path|None=None)->None:
    rc=2; payload={}; owner_pid=os.getpid()
    try:
        _enable_validation_subreaper()
        tmp=Path(tempfile.mkdtemp(prefix='ig-decoder-validation-'))
        if uid is not None and gid is not None and hasattr(os,'setuid'):
            os.chown(tmp,int(uid),int(gid)); os.setgroups([]); os.setgid(int(gid)); os.setuid(int(uid))
        os.chdir(candidate); os.umask(0o077)
        os.environ['PYTHONDONTWRITEBYTECODE']='1'; os.environ['PYTHONPATH']=str(candidate)
        os.environ['TMPDIR']=str(tmp); os.environ['TEMP']=str(tmp); os.environ['TMP']=str(tmp)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        plan=json.loads((log_path.parent/'REPORT_PLAN.json').read_text())
        node_root=log_path.parent/'node_process_logs';node_root.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ);env['PYTHONPATH']=str(candidate)
        code=0;node_logs=[];phase_receipts=[]
        # Never execute pytest inside the long-lived worker.  Each selector gets
        # a fresh exec boundary, so test forks cannot inherit the worker's
        # evidence stream or Python finalization frame.
        for position,node in enumerate(nodes):
            stem=f'{position:06d}-{canonical_sha256(node)}'
            final=node_root/(stem+'.log')
            proc,raw=_run_node_process(
                [sys.executable,'-m','infinity_grid.validation_node_runner',
                 '--candidate',str(candidate),'--log-root',str(log_path.parent),
                 '--binding',plan['binding'],'--selector',node],
                cwd=candidate,env=env,final_path=final)
            node_logs.append(raw)
            if proc.returncode!=0:code=proc.returncode
            else:
                from .validation_reports import verify_finalization
                receipt=json.loads((log_path.parent/'finalization_refs'/(canonical_sha256([node])+'.json')).read_text())
                if receipt.get('selectors') != [node]:
                    raise ValidationRuntimeError('VALIDATION_FINAL_SELECTOR')
                verify_finalization(log_path.parent,plan['binding'],receipt)
                phase_receipts.append(receipt)
        _wait_for_validation_descendants()
        active_path=log_path.with_suffix('.active')
        with active_path.open('xb') as stream:
            for raw in node_logs:stream.write(raw)
            stream.flush();os.fsync(stream.fileno())
        if os.getpid()!=owner_pid:os._exit(0)
        log_record=_publish_worker_log(active_path,log_path)
        phase_path=log_path.with_suffix('.phases.json')
        from .canon import write_json_atomic
        write_json_atomic(phase_path,{'binding':plan['binding'],'receipts':phase_receipts})
        phase_raw=phase_path.read_bytes()
        text=log_path.read_bytes().decode('utf-8',errors='replace')
        benchmarks=_benchmark_payloads(text)
        payload={'return_code':code,'nodes':nodes,'status':'PASS' if code==0 else 'FAIL',**log_record,'tail':_compact(text),'benchmarks':benchmarks}
        payload.update(phase_manifest_sha256=hashlib.sha256(phase_raw).hexdigest(),phase_manifest_size_bytes=len(phase_raw))
        rc=0
    except BaseException as exc:
        payload={'return_code':2,'nodes':nodes,'status':'FAIL','error':type(exc).__name__+':'+str(exc),'tail':''}
        rc=0
    if log_path is not None:
        # Full text is on disk; keep the result pipe below PIPE_BUF so the legacy
        # wait-then-read reducer cannot wait for a child blocked on a large log.
        payload.pop("nodes", None)
        payload["tail"] = str(payload.get("tail", ""))[-1000:]
        payload.pop("benchmarks", None)
        if "error" in payload: payload["error"] = str(payload["error"])[-500:]
    try: os.write(wfd,(json.dumps(payload,sort_keys=True)+'\n').encode('utf-8'))
    except Exception: rc=3
    try: os.close(wfd)
    except OSError: pass
    os._exit(rc)

def _run_group(candidate:Path,group:str,*,workers:int,uid:int|None,gid:int|None,wall_seconds_max:float|None,registered_nodes:list[str]|None=None,log_root:Path|None=None)->dict[str,Any]:
    nodes=_nodes(candidate,group) if registered_nodes is None else list(registered_nodes)
    from .validation_reports import pending_selectors,reduce_reports
    from .v05_engineering_worker import engineering_source_tree_digest
    from .canon import write_json_atomic
    log_root=Path(log_root) if log_root is not None else Path(tempfile.mkdtemp(prefix='decoder-validation-records-'))
    log_root.mkdir(parents=True,exist_ok=True)
    binding=canonical_sha256({'source':engineering_source_tree_digest(candidate),'nodes':nodes,'python':sys.version})
    plan=log_root/'REPORT_PLAN.json'
    if plan.exists() and json.loads(plan.read_text())['binding']!=binding:raise ValidationRuntimeError('VALIDATION_REPORT_PLAN_CHANGED')
    write_json_atomic(plan,{'binding':binding,'nodes':nodes})
    pending_nodes=pending_selectors(log_root,binding,nodes)
    prior_worker_evidence=_evidence_baseline(log_root,'worker-*.log')
    node_root=log_root/'node_process_logs'
    prior_node_evidence=_evidence_baseline(node_root,'*.log') if node_root.is_dir() else {}
    parts=_partition(pending_nodes,workers)
    children=[]; started=time.monotonic()
    for idx,part in enumerate(parts):
        log_path=log_root / f"worker-{idx}-{time.time_ns()}.log"
        rfd,wfd=os.pipe(); pid=os.fork()
        if pid==0:
            os.close(rfd); _child(candidate,part,uid,gid,wfd,log_path)
        os.close(wfd); children.append((idx,pid,rfd,part,log_path))
    results=[]; deadline=None if wall_seconds_max is None else started+float(wall_seconds_max)
    pending={pid:(idx,rfd,part,log_path) for idx,pid,rfd,part,log_path in children}
    try:
        while pending:
            progressed=False
            for pid in list(pending):
                got,st=os.waitpid(pid,os.WNOHANG)
                if not got: continue
                progressed=True; idx,rfd,part,log_path=pending.pop(pid)
                raw=b''
                while True:
                    b=os.read(rfd,65536)
                    if not b: break
                    raw+=b
                os.close(rfd)
                try: obj=json.loads(raw.decode('utf-8').strip())
                except Exception: obj={'return_code':2,'nodes':part,'status':'FAIL','error':'MALFORMED_VALIDATION_CHILD_RESULT','tail':raw.decode('utf-8','replace')[-2000:]}
                try: _verify_worker_log(log_path,obj)
                except (OSError,ValidationRuntimeError) as exc:
                    reason=type(exc).__name__+':'+str(exc)
                    obj.update(return_code=2,status='FAIL',error=reason,tail=reason)
                obj['log_path']=log_path.name
                obj['nodes']=part; obj['worker_index']=idx; obj['pid']=pid; results.append(obj)
            if pending and deadline is not None and time.monotonic()>=deadline:
                for pid,(idx,rfd,part,log_path) in pending.items():
                    try: os.kill(pid,signal.SIGKILL)
                    except OSError: pass
                for pid,(idx,rfd,part,log_path) in list(pending.items()):
                    try: os.waitpid(pid,0)
                    except OSError: pass
                    try: os.close(rfd)
                    except OSError: pass
                    results.append({'worker_index':idx,'pid':pid,'nodes':part,'return_code':124,'status':'FAIL','error':'VALIDATION_TIMEOUT','tail':''})
                pending.clear(); break
            if pending:
                from .preservation import poll, safe_point
                poll();safe_point('VALIDATION_PROGRESS')
            if pending and not progressed: time.sleep(0.02)
    finally:
        for pid,(idx,rfd,part,log_path) in list(pending.items()):
            try:os.kill(pid,signal.SIGKILL)
            except OSError:pass
            try:os.waitpid(pid,0)
            except OSError:pass
            try:os.close(rfd)
            except OSError:pass
    _verify_worker_evidence_set(log_root,results,prior=prior_worker_evidence)
    _verify_node_evidence_set(log_root,results,prior=prior_node_evidence)
    from .validation_reports import verify_finalization
    for result in results:
        if result.get('return_code') != 0: continue
        phase_path=(log_root/result['log_path']).with_suffix('.phases.json')
        raw=phase_path.read_bytes()
        if (hashlib.sha256(raw).hexdigest()!=result.get('phase_manifest_sha256') or
                len(raw)!=result.get('phase_manifest_size_bytes')):
            raise ValidationRuntimeError('VALIDATION_PARENT_PHASE_MANIFEST')
        packet=json.loads(raw)
        if packet.get('binding')!=binding or [r['selectors'] for r in packet['receipts']]!=[[n] for n in result['nodes']]:
            raise ValidationRuntimeError('VALIDATION_PARENT_PHASE_SELECTORS')
        for receipt in packet['receipts']: verify_finalization(log_root,binding,receipt)
    node_rows,group_status,collection_errors=reduce_reports(log_root,binding,nodes,results)
    result_core={'schema_id':'IG_DECODER_VALIDATION_GROUP_RESULT_V1','group':group,'nodes':node_rows,'status':group_status}
    if collection_errors:result_core['collection_errors']=collection_errors
    result_sha=canonical_sha256(result_core)
    ordered=sorted(results,key=lambda x:x['worker_index'])
    return {'group':group,'status':result_core['status'],'result_sha256':result_sha,'result':result_core,
            'execution_metadata':{'workers_requested':int(workers),'workers_used':len(parts),'partitions':[r.get('nodes',[]) for r in ordered],
                                  'wall_seconds':time.monotonic()-started,'child_pids':[r.get('pid') for r in ordered],
                                  'worker_logs':[{'worker_index':r.get('worker_index'),'nodes':r.get('nodes',[]),'status':r.get('status'),'return_code':r.get('return_code'),'log_path':r.get('log_path'),'log_sha256':r.get('log_sha256'),'log_size_bytes':r.get('log_size_bytes'),'phase_manifest_sha256':r.get('phase_manifest_sha256'),'phase_manifest_size_bytes':r.get('phase_manifest_size_bytes'),'error':r.get('error'),'benchmarks':r.get('benchmarks',[])} for r in ordered]},
            'failure_tails':[r.get('tail','') for r in results if r.get('status')!='PASS']}

def run_registered_validation_groups(candidate_root:str|Path,groups:Iterable[str],*,workers:int=4,worker_uid:int|None=None,worker_gid:int|None=None,wall_seconds_max:float|None=900)->dict[str,Any]:
    require_controller_execution_origin('run_registered_validation_groups')
    candidate=Path(candidate_root).resolve(strict=True)
    gs=list(groups)
    if not gs or len(gs)!=len(set(gs)): raise ValidationRuntimeError('VALIDATION_GROUP_LIST')
    results=[]; execution=[]
    for g in gs:
        r=_run_group(candidate,g,workers=workers,uid=worker_uid,gid=worker_gid,wall_seconds_max=wall_seconds_max)
        results.append({'group':g,'status':r['status'],'result_sha256':r['result_sha256'],'result':r['result']})
        execution.append({'group':g,**r['execution_metadata']})
        if r['status']!='PASS': raise ValidationRuntimeError('VALIDATION_FAILED:'+g+':'+('\n'.join(r['failure_tails'])[-5000:]))
    core={'schema_id':'IG_DECODER_REGISTERED_VALIDATION_SET_V1','groups':results,'status':'PASS'}
    return {'status':'PASS','result_sha256':canonical_sha256(core),'result':core,'execution_metadata':execution}


def run_registered_validation_nodes(candidate_root: str | Path, nodes: list[str], *,
                                    workers: int, wall_seconds_max: float,
                                    output_dir: str | Path) -> dict[str, Any]:
    """Frozen explicit nodes, same Decoder scheduler, ordinary current-user children."""
    require_controller_execution_origin("run_registered_validation_nodes")
    from .v05_origin_guard import require_registered_output, require_native_caller
    import sys
    caller = sys._getframe(1).f_globals.get('__name__')
    if caller == 'infinity_grid.change_validation':
        require_native_caller(caller, {'validate_revision'}, 'registered-validation')
    else:
        require_native_caller('infinity_grid.v05_controller_event_loop', {'_dispatch_workspace_job'}, 'registered-validation')
    require_registered_output(output_dir, 'registered-validation')
    candidate = Path(candidate_root).resolve(strict=True)
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValidationRuntimeError("VALIDATION_NODE_LIST")
    for node in nodes:
        rel = node.split("::", 1)[0]
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not rel.startswith("tests/") or not rel.endswith(".py"):
            raise ValidationRuntimeError("VALIDATION_NODE_PATH")
        if not (candidate / p).resolve(strict=True).is_relative_to(candidate):
            raise ValidationRuntimeError("VALIDATION_NODE_PATH")
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    result = _run_group(candidate, "REGISTERED_NODES", workers=workers, uid=None, gid=None,
                        wall_seconds_max=wall_seconds_max, registered_nodes=nodes, log_root=out)
    from .canon import write_json_atomic
    write_json_atomic(out / "VALIDATION_RESULT.json", result)
    return result
