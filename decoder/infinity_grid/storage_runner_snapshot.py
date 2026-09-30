"""Bounded transport check using the original runner's read-only verifier.

This does not close result/evidence dependencies or authorize/resume execution.
"""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from .storage_schema import strict_loads
from .replay_reference_data import HASH_RE
from .replay_dag_runner import ReplayDagRunner,CHECKPOINT_SCHEMA,CAPSULE_SCHEMA,DECISION_SCHEMA


class RunnerSnapshotError(ValueError):pass


def verify_runner_snapshot(manifest_raw,state_raw,*,checkpoints=(),capsules=(),decisions=(),max_total_bytes=67108864,max_records=4096):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864 or type(max_records) is not int or not 2<=max_records<=4096:
        raise RunnerSnapshotError('INVALID_SNAPSHOT_BUDGET')
    groups=[('checkpoints',checkpoints,'checkpoint_sha256',CHECKPOINT_SCHEMA),
            ('audit_capsules',capsules,'audit_capsule_sha256',CAPSULE_SCHEMA),
            ('external_decisions',decisions,'decision_sha256',DECISION_SCHEMA)]
    if any(not isinstance(rows,(list,tuple)) for _,rows,_,_ in groups):raise RunnerSnapshotError('SNAPSHOT_LIST_REQUIRED')
    if 2+sum(len(rows) for _,rows,_,_ in groups)>max_records:raise RunnerSnapshotError('SNAPSHOT_RECORD_BUDGET')
    total=0;inventory=[]
    def parse(raw,role):
        nonlocal total
        if type(raw) is not bytes:raise RunnerSnapshotError('SNAPSHOT_BYTES_REQUIRED')
        total+=len(raw)
        if total>max_total_bytes:raise RunnerSnapshotError('SNAPSHOT_BYTE_BUDGET')
        obj=strict_loads(raw,max_bytes=4194304)
        if not isinstance(obj,dict):raise RunnerSnapshotError('SNAPSHOT_OBJECT_REQUIRED')
        inventory.append({'role':role,'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}})
        return obj
    manifest=parse(manifest_raw,'MANIFEST');state=parse(state_raw,'RUNNER_STATE')
    with TemporaryDirectory(prefix='ig-runner-snapshot-') as temp:
        root=Path(temp);(root/'runner_state.json').write_bytes(state_raw)
        for directory,rows,field,schema in groups:
            target=root/directory;target.mkdir();seen=set()
            for raw in rows:
                obj=parse(raw,directory);digest=obj.get(field)
                if obj.get('schema_id')!=schema or not isinstance(digest,str) or HASH_RE.fullmatch(digest) is None:
                    raise RunnerSnapshotError('INVALID_SNAPSHOT_RECORD_IDENTITY')
                if digest in seen:raise RunnerSnapshotError('DUPLICATE_SNAPSHOT_IDENTITY')
                seen.add(digest);(target/(digest+'.json')).write_bytes(raw)
        # Constructor validates seals, manifest/run bindings, inventory and audit
        # history. Do not call next_action: it can checkpoint gates and mutate state.
        runner=ReplayDagRunner(manifest,root,state)
        if (root/'runner_state.json').read_bytes()!=state_raw:raise RunnerSnapshotError('SNAPSHOT_VERIFIER_MUTATED_STATE')
    return {'status':'RUNNER_SNAPSHOT_VERIFIED','scope':'RUNNER_INDEX_AND_AUDIT_HISTORY_ONLY',
        'scientific_acceptance':'NOT_GRANTED','execution_authorized':False,
        'root_run_id':runner.state['root_run_id'],'manifest_sha256':manifest['dag_sha256'],
        'state_sha256':state['state_sha256'],'completed_node_ids':list(state['completed_node_ids']),
        'records_checked':len(inventory),'bytes_checked':total,'inventory':inventory}
