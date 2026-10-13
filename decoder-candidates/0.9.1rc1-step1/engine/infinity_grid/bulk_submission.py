"""Preparation and receipt-only rolling window coordination. Never dispatches science."""
from pathlib import Path
import hashlib
import json
import sys
from .bulk_descriptor_plan import build_plan,window_descriptors
from .canon import canonical_bytes,canonical_sha256,write_json_atomic
from . import submission as sub,portable_registry as reg
from .runtime_capacity import host_memory_ceiling_bytes


def validate_progress(plan, progress, *, source_sha256=None, package_sha256=None,
                      verify_receipt=None):
    """Validate every inherited receipt before it can suppress missing work.

    The reducer uses structural validation; execution additionally supplies the
    native saved-receipt verifier and the admitted cutover identities.
    """
    if type(progress) is not dict:
        raise ValueError('PROGRESS_FIELDS')
    if not progress:
        return []
    fields = {'schema_id', 'plan_sha256', 'inherited_prefix', 'completed_windows',
              'next_ordinal', 'missing_rows', 'source_cutover_frozen', 'scientific_adoption'}
    if set(progress) != fields or progress['schema_id'] != 'IG_HET4_ROLLING_MAPPING_PROGRESS_V1':
        raise ValueError('PROGRESS_FIELDS')
    if progress['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('PROGRESS_PLAN_CHANGED')
    if canonical_bytes(progress['inherited_prefix']) != canonical_bytes(plan['prefix']):
        raise ValueError('PROGRESS_PREFIX_CHANGED')
    completed = progress['completed_windows']
    if type(completed) is not list or len(completed) > len(plan['windows']):
        raise ValueError('PROGRESS_WINDOW_COUNT')
    if progress['source_cutover_frozen'] is not True or progress['scientific_adoption'] is not False:
        raise ValueError('PROGRESS_AUTHORITY')
    source = source_sha256
    package = package_sha256
    for index, receipt in enumerate(completed):
        if type(receipt) is not dict or type(receipt.get('window_index')) is not int:
            raise ValueError('WINDOW_RECEIPT_ORDER')
        if receipt['window_index'] != index or receipt.get('status') != 'VERIFIED_SAVED_WINDOW':
            raise ValueError('WINDOW_RECEIPT_ORDER')
        window = plan['windows'][index]
        if any(type(receipt.get(k)) is not int or receipt[k] != window[k]
               for k in ('start', 'stop', 'rows')):
            raise ValueError('WINDOW_RECEIPT_COVERAGE')
        if source is None:
            source, package = receipt.get('source_sha256'), receipt.get('package_sha256')
        if (not source or not package or receipt.get('source_sha256') != source
                or receipt.get('package_sha256') != package):
            raise ValueError('WINDOW_CUTOVER_SOURCE_CHANGED')
        if verify_receipt is not None:
            observed = verify_receipt(index, receipt)
            if canonical_bytes(observed) != canonical_bytes(receipt):
                raise ValueError('WINDOW_SAVED_RECEIPT_CHANGED')
    next_ordinal = completed[-1]['stop'] if completed else plan['prefix']['records']
    if (type(progress['next_ordinal']) is not int or progress['next_ordinal'] != next_ordinal
            or type(progress['missing_rows']) is not int
            or progress['missing_rows'] != plan['domain_rows'] - next_ordinal):
        raise ValueError('PROGRESS_DERIVED_COVERAGE')
    return completed


def _saved_progress_receipt(root, plan, index, receipt):
    """Reopen the actual original producer; allow a later saved checkpoint."""
    from . import preservation as pr
    directory = Path(root) / f'window{index:04d}'
    pointer = directory / 'WORKSPACE.json'
    if pointer.exists():
        workspace = Path(json.loads(pointer.read_bytes())['path'])
    else:
        candidates = list((directory / 'store/captures').glob('*/CAPTURE.json'))
        if len(candidates) != 1:
            raise ValueError('WINDOW_CAPTURE_RECOVERY_REQUIRED')
        workspace = candidates[0].parent
    observed = verify_window_receipt(plan, index, workspace)
    # Checkpoints can legitimately advance after the original completion. Keep
    # its recorded digest, but require the original commit to remain preserved.
    old_checkpoint = receipt.get('checkpoint_sha256')
    if old_checkpoint not in pr._commits(workspace):
        raise ValueError('WINDOW_CHECKPOINT_MISSING')
    observed['checkpoint_sha256'] = old_checkpoint
    return observed


def prepare_window(root, engine_source, project_source, raw_archive, prefix_archive, sqlite_binding, index, *, window_rows=16384, workers=4):
    root=Path(root).resolve();root.mkdir(exist_ok=False)
    plan=build_plan(raw_archive,prefix_archive,window_rows=window_rows)
    if type(index) is not int or not 0<=index<len(plan['windows']):raise ValueError('WINDOW_INDEX')
    window=plan['windows'][index]
    descriptors=window_descriptors(raw_archive,window)
    plan_path=root/'full-domain-plan.json';rows_path=root/'descriptor-window.json'
    write_json_atomic(plan_path,plan);write_json_atomic(rows_path,descriptors)
    job_id=f"HET4.MAPPING.WINDOW.{window['start']:06d}.{window['stop']:06d}"
    artifacts=[('sqlite_runtime_binding',sqlite_binding),('descriptor_sample',rows_path),('full_domain_plan',plan_path),('raw_descriptor_archive',raw_archive),('preserved_prefix_archive',prefix_archive)]
    spec={'schema_id':sub.SPEC_SCHEMA,'job_id':job_id,'engine_source':str(Path(engine_source).resolve()),'project_source':str(Path(project_source).resolve()),
        'question':{'stage_id':'HET4:MAPPING-WINDOW','description':f"Source-bound mapping window [{window['start']},{window['stop']}); preserve existing prefix",'outcomes':['PASS'],'stopping_rule':'Complete exactly this registered window; advancement requires verified native saved completion'},
        'execution':{'kind':'STAGE','handler_ref':'project.bulk_adapter:handler','evaluator_refs':['project.bulk_adapter:evaluate'],'parameters':{'registered_rows':len(descriptors),'batch_size':16,'workers':workers,'window_index':index,'window_rows':window_rows}},
        'resources':{'workers':workers,'start_method':'spawn','memory_budget_bytes':host_memory_ceiling_bytes(),'workspace_budget_bytes':None,'execution_policy':'NO_AUTOMATIC_RUNTIME_DEADLINE_V1'},
        'inputs':[],'environment':{'python':f'{sys.version_info.major}.{sys.version_info.minor}','requirements':[],'artifacts':[{'logical_name':name,'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for name,p in artifacts]},
        'output_contract':{'schema_id':'IG_DECODER_RESULT_CONTRACT_V1','claim':'EXECUTION_ONLY','required_artifacts':[],'result_checks':[{'pointer':'/outcome','equals':'PASS'}],'prerequisites':[],'preservation':{'project_evidence_externalization':1,'max_project_raw_bytes':None,'completion_project_closure':1,'rolling_unbounded_execution':1}}}
    write_json_atomic(root/'SPECIFICATION.json',spec)
    store=root/'store';reg.initialize(store,'Exact HET4 independent mapping window',engine_source)
    return {'plan':plan,'window':window,'capture':sub.capture(store,spec),'dispatched':False,'scientific_adoption':False}


def verify_window_receipt(plan, index, workspace):
    """Read-only gate; callers publish a new compact progress artifact after success."""
    from . import preservation as pr,v05_controller_event_loop as loop
    from .sqlite_attestation import admit_capture_runtime
    if type(index) is not int or not 0 <= index < len(plan['windows']):
        raise ValueError('WINDOW_INDEX')
    root=Path(workspace).resolve();rec=sub.capture_record(root)
    if sub.save_status(root)['status']!='SAVED' or pr.status(root)['status']!='PRESERVED':raise ValueError('WINDOW_SAVE_NOT_CLOSED')
    admission=loop.validate_workspace_job(root,rec['job']['job_id'],check_loaded=False)
    admit_capture_runtime(root, rec)
    done=loop.verified_completion(admission)
    if done is None or done['status']!='COMPLETED':raise ValueError('WINDOW_COMPLETION_REQUIRED')
    params=rec['job']['execution']['parameters'];window=plan['windows'][index]
    if params.get('window_index')!=index or params['registered_rows']!=window['rows']:raise ValueError('WINDOW_COMPLETION_SCOPE')
    mapping=done['result']['mapping']
    if mapping['plan_sha256']!=plan['plan_sha256'] or mapping['descriptor_stream_sha256']!=window['descriptor_stream_sha256'] or mapping['rows']!=window['rows']:
        raise ValueError('WINDOW_PLAN_BINDING')
    checkpoint=pr.status(root)['latest_checkpoint']
    return {'window_index':index,'start':window['start'],'stop':window['stop'],'rows':window['rows'],'capture_id':rec['capture_id'],'source_sha256':rec['workspace']['source_sha256'],'package_sha256':rec['workspace']['package_sha256'],'completion_sha256':done['completion_sha256'],'checkpoint_sha256':checkpoint,'full_rows_stream_sha256':mapping['full_rows_stream_sha256'],'database_sha256':mapping['database_sha256'],'status':'VERIFIED_SAVED_WINDOW'}


def advance_progress(plan, progress, receipt):
    """Pure ordered receipt reducer; initial old prefix is never rewritten."""
    completed=list(validate_progress(plan, progress));index=len(completed)
    if type(receipt) is not dict or type(receipt.get('window_index')) is not int or receipt['window_index'] < 0:
        raise ValueError('WINDOW_RECEIPT_ORDER')
    if receipt['window_index']<index:
        if canonical_bytes(completed[receipt['window_index']])!=canonical_bytes(receipt):raise ValueError('WINDOW_RECEIPT_RETRY_CHANGED')
        return progress
    if index >= len(plan['windows']) or receipt['window_index']!=index or receipt['status']!='VERIFIED_SAVED_WINDOW':raise ValueError('WINDOW_RECEIPT_ORDER')
    expected=plan['windows'][index]
    if any(type(receipt.get(k)) is not int or receipt[k]!=expected[k] for k in ['start','stop','rows']):raise ValueError('WINDOW_RECEIPT_COVERAGE')
    if completed and any(receipt[k]!=completed[0][k] for k in ['source_sha256','package_sha256']):raise ValueError('WINDOW_CUTOVER_SOURCE_CHANGED')
    completed.append(receipt)
    updated={'schema_id':'IG_HET4_ROLLING_MAPPING_PROGRESS_V1','plan_sha256':plan['plan_sha256'],'inherited_prefix':plan['prefix'],'completed_windows':completed,'next_ordinal':receipt['stop'],'missing_rows':plan['domain_rows']-receipt['stop'],'source_cutover_frozen':True,'scientific_adoption':False}
    validate_progress(plan,updated)
    return updated


def execute_adopted_windows(root, engine_source, project_source, raw_archive, prefix_archive,
                            sqlite_binding, adoption, transport, *, window_rows=16384, workers=4):
    """One sequential writer; transport saves actual engine obligations.

    Data adoption is a required explicit input. This function is not called by
    Decoder qualification. It cannot advance on upload success without native
    saved-completion receipts. Transport must provide save_capture(workspace)
    and preserve_workspace(workspace); both are independently checked below.
    """
    from . import preservation as pr, v05_controller_event_loop as loop
    from .isolated_runtime import verified_runtime, BUNDLED_MANIFEST_SHA256
    from .immutable_publication import sync_directory
    import fcntl, os
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ig-cutover-binding-') as temp:
        composite=Path(temp)
        sub._write_files(composite,sub._tree(engine_source))
        sub._write_files(composite/'project',sub._tree(project_source))
        source_sha,package_sha=loop._source_ids(composite)
    verified_runtime()  # full file-set, bytes and mode verification before work
    plan=build_plan(raw_archive,prefix_archive,window_rows=window_rows)
    expected={'schema_id':'IG_DATA_OWNER_DECODER_ADOPTION_V1','status':'ADOPTED',
        'source_sha256':source_sha,'package_sha256':package_sha,
        'runtime_manifest_sha256':BUNDLED_MANIFEST_SHA256,'plan_sha256':plan['plan_sha256'],
        'preserved_prefix_rows':plan['preserved_rows']}
    if any(adoption.get(k)!=v for k,v in expected.items()):raise ValueError('DATA_OWNER_ADOPTION_REQUIRED')
    # This JSON is an operator/data-owner attestation, not remote identity auth.
    fd=os.open(root/'WRITER.lock',os.O_CREAT|os.O_RDWR,0o600)
    try:
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('COMPETING_WINDOW_WRITER') from None
        progress_path=root/'PROGRESS.json'
        progress=json.loads(progress_path.read_bytes()) if progress_path.exists() else {}
        verify_saved=lambda index,receipt:_saved_progress_receipt(root,plan,index,receipt)
        validate_progress(plan, progress, source_sha256=source_sha, package_sha256=package_sha,
                          verify_receipt=verify_saved)
        while len(progress.get('completed_windows',[]))<len(plan['windows']):
            index=len(progress.get('completed_windows',[]));directory=root/f'window{index:04d}'
            if not directory.exists():
                prepared=prepare_window(directory,engine_source,project_source,raw_archive,prefix_archive,
                    sqlite_binding,index,window_rows=window_rows,workers=workers)
                workspace=Path(prepared['capture']['workspace'])
                write_json_atomic(directory/'WORKSPACE.json',{'path':str(workspace)})
            else:
                pointer=directory/'WORKSPACE.json'
                if not pointer.exists():
                    candidates=list((directory/'store/captures').glob('*/CAPTURE.json'))
                    if len(candidates)!=1:raise ValueError('WINDOW_CAPTURE_RECOVERY_REQUIRED')
                    workspace=candidates[0].parent
                    write_json_atomic(pointer,{'path':str(workspace)})
                else:
                    workspace=Path(json.loads(pointer.read_bytes())['path'])
            record=sub.capture_record(workspace)
            if record['workspace']['source_sha256']!=source_sha or record['workspace']['package_sha256']!=package_sha:
                raise ValueError('WINDOW_SOURCE_CHANGED')
            transport.save_capture(workspace)
            if sub.save_status(workspace)['status']!='SAVED':raise ValueError('CAPTURE_TRANSPORT_NOT_ACKNOWLEDGED')
            while True:
                try:
                    loop.run_workspace_job(workspace,record['job']['job_id'])
                    break
                except sub.SubmissionError as exc:
                    if 'SAVE_BACKLOG_PAUSE' not in str(exc):raise
                    transport.preserve_workspace(workspace)
                    if pr.status(workspace)['status']!='PRESERVED':raise ValueError('BACKLOG_TRANSPORT_NOT_ACKNOWLEDGED')
            transport.preserve_workspace(workspace)
            receipt=verify_window_receipt(plan,index,workspace)
            progress=advance_progress(plan,progress,receipt)
            write_json_atomic(progress_path,progress);sync_directory(root)
        validate_progress(plan, progress, source_sha256=source_sha, package_sha256=package_sha,
                          verify_receipt=verify_saved)
        return progress
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
