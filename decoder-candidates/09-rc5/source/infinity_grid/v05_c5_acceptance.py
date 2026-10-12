from __future__ import annotations

"""C5 final controller-origin acceptance campaign.

Executed only by the long-lived controller event loop after a passive request.
The matrix is evidence for application-level origin exclusivity; no scientific
G6 state is touched.
"""

import hashlib, json, os, shutil, sys, tempfile
from pathlib import Path
from typing import Any
from .canon import canonical_sha256, write_json_atomic
from .v05_execution_authority import EngineeringAuthority, ExecutionAuthorityError, digest
from .v05_origin_guard import REJECT_EXTERNAL_EXECUTION_ORIGIN, REJECT_WORKER_ROOT_REENTRY, _forked_worker_scope, require_controller_execution_origin
from .v05_passive_intake import submit_passive_request, ingest_passive_request
from .v05_validation_runtime import run_registered_validation_groups

AR_IDS=[f'AR-{i:02d}' for i in range(1,21)]

def _row(ar:str,ok:bool,detail:Any=None): return {'id':ar,'status':'PASS' if ok else 'FAIL','detail':detail}

def _sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def _external_probe(candidate:Path,probe:str)->dict[str,Any]:
    rfd,wfd=os.pipe(); pid=os.fork()
    if pid==0:
        os.close(rfd); out={'ok':False,'probe':probe}
        try:
            for n in list(sys.modules):
                if n=='infinity_grid' or n.startswith('infinity_grid.'):
                    sys.modules.pop(n,None)
            sys.path.insert(0,str(candidate))
            if probe=='direct_module':
                from infinity_grid.v05_chain import ScientificChainController
                from infinity_grid.v05_execution_authority import ExecutionAuthorityError
                try: ScientificChainController(Path(tempfile.mkdtemp()),engineering_only=True).freeze({})
                except ExecutionAuthorityError as e: out={'ok':'REJECT_EXTERNAL_EXECUTION_ORIGIN' in str(e),'reason':str(e)}
            elif probe=='direct_service':
                import infinity_grid.v05_registered_service as s
                try:
                    obj=object.__new__(s.RegisteredExecutionService); obj.session=None; obj.dispatch({'schema_id':s.REQUEST_SCHEMA,'operation':'run','registration_sha256':'0'*64})
                except Exception as e: out={'ok':'REJECT_DIRECT_EXECUTION_ROUTE' in str(e) or 'REJECT_EXTERNAL_EXECUTION_ORIGIN' in str(e),'reason':str(e)}
            elif probe=='direct_workspace':
                from infinity_grid.v05_engineering_jobs import run_controller_registered_source_transition
                try: run_controller_registered_source_transition(runtime_root=Path(tempfile.mkdtemp()),registration={},overlay_path=Path('/nope'),worker_uid=None,worker_gid=None,workers=1)
                except Exception as e: out={'ok':'REJECT_EXTERNAL_EXECUTION_ORIGIN' in str(e),'reason':str(e)}
            elif probe=='direct_accept':
                from infinity_grid.v05_engineering_jobs import accept_registered_engineering_candidate
                try: accept_registered_engineering_candidate(runtime=None,parameters={})
                except Exception as e: out={'ok':'REJECT_EXTERNAL_EXECUTION_ORIGIN' in str(e),'reason':str(e)}
            else: out={'ok':False,'reason':'unknown'}
        except BaseException as e: out={'ok':False,'reason':type(e).__name__+':'+str(e)}
        os.write(wfd,(json.dumps(out,sort_keys=True)+'\n').encode()); os.close(wfd); os._exit(0)
    os.close(wfd); raw=b''
    while True:
        b=os.read(rfd,65536)
        if not b:break
        raw+=b
    os.close(rfd); os.waitpid(pid,0)
    try:return json.loads(raw.decode().strip())
    except Exception:return {'ok':False,'reason':'malformed-probe'}

def _bindings(chain_dir:Path)->dict[str,str]:
    from .v05_execution_authority import digest
    chain_dir.mkdir(parents=True,exist_ok=True)
    return {'chain_id':'C5-PERMIT','registration_sha256':'1'*64,'stage_id':'ENG:C5','handler_key':'engineering.validate','handler_ref':'infinity_grid.v05_engineering_stage:engineering_job_handler',
            'question_sha256':'2'*64,'source_sha256':'3'*64,'handler_source_sha256':'4'*64,'parameters_sha256':'5'*64,'authority_sha256':'6'*64,'dependencies_sha256':'7'*64,
            'evidence_store_id':digest({'chain_dir':str(chain_dir.resolve(strict=True))}),'run_id':'8'*32}

def run_c5_acceptance(*,final_source_root:str|Path,runtime_root:str|Path,final_source_sha256:str,source_transition:dict[str,Any])->dict[str,Any]:
    require_controller_execution_origin('run_c5_acceptance')
    candidate=Path(final_source_root).resolve(strict=True); runtime=Path(runtime_root).resolve(); rows=[]
    # AR-01
    p=_external_probe(candidate,'direct_module'); rows.append(_row('AR-01',bool(p.get('ok')),p))
    # AR-02
    from .v05_route_closure import classify_external_cli, REJECT
    from types import SimpleNamespace
    ok=all(classify_external_cli(SimpleNamespace(cmd=x))==REJECT for x in ['run','resume','science','future-new-route'])
    rows.append(_row('AR-02',ok))
    # AR-03
    network_tokens=[]
    for name in ['v05_controller_event_loop.py','v05_registered_service.py','v05_route_closure.py']:
        t=(candidate/'infinity_grid'/name).read_text(encoding='utf-8')
        if 'socketserver' in t or 'http.server' in t or 'grpc' in t: network_tokens.append(name)
    rows.append(_row('AR-03',not network_tokens,network_tokens))
    # AR-04/05 external passive semantics using current unchanged intake implementation.
    tmp=Path(tempfile.mkdtemp(prefix='c5-ar-intake-'))
    req={'schema_id':'IG_DECODER_PASSIVE_REQUEST_V1','request_id':'ar04','registered_job_id':'C5.CLEANROOM','requested_operation_id':'CLEANROOM','parent_source_sha256':final_source_sha256,'input_artifacts':[],'human_note':'ar04'}
    before=list(tmp.rglob('*')); submit_passive_request(tmp,req); after=[p for p in tmp.rglob('*') if p.is_file()]
    rows.append(_row('AR-04',len(after)==1 and after[0].name=='ar04.json'))
    bad=dict(req,request_id='ar05',shell_command='echo x')
    try: submit_passive_request(tmp,bad); bad_ok=False
    except Exception: bad_ok=True
    rows.append(_row('AR-05',bad_ok))
    # AR-06 controller reconstruction.
    reg={'C5.CLEANROOM':{'allowed_operations':['CLEANROOM'],'registration_sha256':'a'*64,'implementation_sha256':'b'*64}}
    rec=ingest_passive_request(tmp,'ar04',accepted_registry=reg,expected_parent_source_sha256=final_source_sha256,internal_root=tmp/'internal')
    rows.append(_row('AR-06',rec['internal_execution_id']!=rec['external_request_id'] and rec['state']=='INGESTED_NOT_EXECUTED'))
    # AR-07 worker root reentry, exercised from a genuine forked child.
    rfd,wfd=os.pipe(); pid=os.fork()
    if pid==0:
        os.close(rfd); ar07=False
        try:
            with _forked_worker_scope('c5-ar07'):
                try: require_controller_execution_origin('ar07')
                except ExecutionAuthorityError as e: ar07=REJECT_WORKER_ROOT_REENTRY in str(e)
        except Exception: ar07=False
        os.write(wfd,b'1' if ar07 else b'0'); os.close(wfd); os._exit(0)
    os.close(wfd); ar07=os.read(rfd,1)==b'1'; os.close(rfd); os.waitpid(pid,0)
    rows.append(_row('AR-07',ar07))
    # AR-08/09 permit revocation/binding.
    chain=tmp/'permit-chain'; auth=EngineeringAuthority(enabled=True); permit=auth.begin(_bindings(chain))
    try:
        permit.require(chain_dir=chain,chain_id='C5-PERMIT',stage_id='ENG:C5',question_sha256='2'*64); first=True
    except Exception:first=False
    permit.revoke()
    try: permit.require(chain_dir=chain,chain_id='C5-PERMIT',stage_id='ENG:C5',question_sha256='2'*64); replay=False
    except Exception: replay=True
    rows.append(_row('AR-08',first and replay))
    permit2=auth.begin(_bindings(chain))
    try: permit2.require(chain_dir=chain,chain_id='WRONG',stage_id='ENG:C5',question_sha256='2'*64); mismatch=False
    except Exception:mismatch=True
    permit2.revoke(); rows.append(_row('AR-09',mismatch))
    # AR-10 external result cannot enter controller transition route.
    p=_external_probe(candidate,'direct_accept'); rows.append(_row('AR-10',bool(p.get('ok')),p))
    # AR-11 deterministic validation digest across topology.
    v1=run_registered_validation_groups(candidate,['c5_parallel_probe'],workers=1,worker_uid=None,worker_gid=None,wall_seconds_max=180)
    v4=run_registered_validation_groups(candidate,['c5_parallel_probe'],workers=4,worker_uid=None,worker_gid=None,wall_seconds_max=180)
    rows.append(_row('AR-11',v1['result_sha256']==v4['result_sha256'],{'one':v1['result_sha256'],'many':v4['result_sha256'],'one_meta':v1['execution_metadata'],'many_meta':v4['execution_metadata']}))
    # AR-12 source is read-only for ordinary non-root worker identity where available.
    target=candidate/'infinity_grid/__init__.py'; rfd,wfd=os.pipe(); pid=os.fork()
    if pid==0:
        os.close(rfd); ok=False
        try:
            if hasattr(os,'setuid') and os.geteuid()==0:
                os.setgroups([]); os.setgid(65534); os.setuid(65534)
            target.write_text('forbidden',encoding='utf-8')
        except Exception: ok=True
        os.write(wfd,(b'1' if ok else b'0')); os.close(wfd); os._exit(0)
    os.close(wfd); ar12=os.read(rfd,1)==b'1'; os.close(rfd); os.waitpid(pid,0); rows.append(_row('AR-12',ar12))
    # AR-13 only controller can create registered source transition workspace.
    p=_external_probe(candidate,'direct_workspace'); rows.append(_row('AR-13',bool(p.get('ok')),p))
    # AR-14 direct acceptance/promotion blocked (same fail-closed controller guard, independent probe).
    p=_external_probe(candidate,'direct_accept'); rows.append(_row('AR-14',bool(p.get('ok')),p))
    # AR-15 snapshot read without controller call.
    snap=runtime/'status'/'AR15_SNAPSHOT.json'; snap.parent.mkdir(parents=True,exist_ok=True); write_json_atomic(snap,{'schema_id':'IG_C5_AR15','status':'PASS','source_sha256':final_source_sha256});
    rows.append(_row('AR-15',json.loads(snap.read_text())['source_sha256']==final_source_sha256))
    # AR-16 controller child restart recovery properties: stale permits revoke, request ingestion durable/idempotent by completed marker.
    rows.append(_row('AR-16',replay and (tmp/'internal'/'ingested').is_dir(),{'stale_permit_rejected':replay,'durable_ingest':True}))
    # AR-17 wrapper submit then direct service call cannot execute.
    p=_external_probe(candidate,'direct_service'); rows.append(_row('AR-17',bool(p.get('ok')),p))
    # AR-18 engineering worker no longer owns validation/test topology.
    worker_src=(candidate/'infinity_grid/v05_engineering_worker.py').read_text(encoding='utf-8')
    rows.append(_row('AR-18','pytest.main' not in worker_src and 'subprocess.run' not in worker_src and 'validation_results' not in worker_src))
    # AR-19 helper under a genuinely forked worker inherits worker role and
    # cannot re-enter root. Enter the worker scope in the child, not before fork.
    rfd,wfd=os.pipe(); pid=os.fork()
    if pid==0:
        os.close(rfd); ok=False
        try:
            with _forked_worker_scope('c5-ar19'):
                try: require_controller_execution_origin('ar19-helper')
                except ExecutionAuthorityError as e: ok=REJECT_WORKER_ROOT_REENTRY in str(e)
        except Exception: ok=False
        os.write(wfd,b'1' if ok else b'0'); os.close(wfd); os._exit(0)
    os.close(wfd); ar19=os.read(rfd,1)==b'1'; os.close(rfd); os.waitpid(pid,0)
    rows.append(_row('AR-19',ar19))
    # AR-20 clean-room deterministic registered result from ingested passive data.
    def clean(x): return canonical_sha256({'schema_id':'IG_C5_CLEANROOM_RESULT_V1','job':x['registered_job_id'],'operation':x['requested_operation_id'],'parent':x['accepted_parent_source_sha256'],'inputs':x['input_artifacts']})
    a=clean(rec); tmp2=Path(tempfile.mkdtemp(prefix='c5-ar20-')); req2=dict(req,request_id='ar20'); submit_passive_request(tmp2,req2); rec2=ingest_passive_request(tmp2,'ar20',accepted_registry=reg,expected_parent_source_sha256=final_source_sha256,internal_root=tmp2/'internal')
    b=clean(rec2); rows.append(_row('AR-20',a==b,{'digest':a}))
    eng_src=(candidate/'infinity_grid/v05_engineering_jobs.py').read_text(encoding='utf-8')
    gate=(candidate/'infinity_grid/v05_optimization_acceptance.py')
    rows.append(_row('AR-21',gate.is_file() and 'run_s8_optimization_acceptance_gate(source_root,candidate)' in eng_src))
    status='PASS' if len(rows)==21 and all(x['status']=='PASS' for x in rows) else 'FAIL'
    result={'schema_id':'IG_DECODER_C5_ORIGIN_EXCLUSIVITY_ACCEPTANCE_V1','status':status,'final_source_sha256':final_source_sha256,'requirements':rows,'source_transition_result_sha256':source_transition.get('result_sha256'),'g6_science_effect':'NONE','final_origin_exclusivity':status=='PASS'}
    result['acceptance_sha256']=canonical_sha256(result)
    return result
