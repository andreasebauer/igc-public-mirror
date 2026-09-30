"""Bound original snapshot results to sealed comparison/evidence records.

This verifies record identities, not observation bytes or scientific outcomes.
"""
import hashlib
from .canon import canonical_sha256
from .storage_schema import strict_loads
from .replay_reference_data import verify_reference_record, empty_manifest
from .replay_dag_runner import RESULT_SCHEMA
from .storage_runner_snapshot import verify_runner_snapshot, RunnerSnapshotError


PROFILES={'SINGLE_REPLAY_RECORD_V1','ORDERED_REPLAY_SEALS_V1','ORDERED_PAIRED_SEALS_V1'}


def verify_snapshot_results(manifest_raw,state_raw,dataset_raw,bindings,*,
                            checkpoints=(),capsules=(),decisions=(),
                            max_total_bytes=67108864,max_records=4096):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864 or type(max_records) is not int or not 3<=max_records<=4096:
        raise RunnerSnapshotError('INVALID_RESULT_BINDING_BUDGET')
    if type(bindings) is not dict or len(bindings)>max_records:
        raise RunnerSnapshotError('INVALID_RESULT_BINDINGS')
    if any(not isinstance(rows,(list,tuple)) for rows in (checkpoints,capsules,decisions)):
        raise RunnerSnapshotError('SNAPSHOT_LIST_REQUIRED')
    raws=[manifest_raw,state_raw,dataset_raw,*checkpoints,*capsules,*decisions]
    for binding in bindings.values():
        if not isinstance(binding,dict) or set(binding)!={'profile','records'} or binding['profile'] not in PROFILES or not isinstance(binding['records'],(list,tuple)):
            raise RunnerSnapshotError('UNSUPPORTED_RESULT_BINDING_PROFILE')
        if len(binding['records'])>max_records:raise RunnerSnapshotError('RESULT_RECORD_BUDGET')
        raws.extend(binding['records'])
    if len(raws)>max_records:raise RunnerSnapshotError('RESULT_RECORD_BUDGET')
    if any(type(raw) is not bytes for raw in raws):raise RunnerSnapshotError('RESULT_BYTES_REQUIRED')
    total=sum(len(raw) for raw in raws)
    if total>max_total_bytes or any(len(raw)>4194304 for raw in raws):raise RunnerSnapshotError('RESULT_BYTE_BUDGET')
    snapshot=verify_runner_snapshot(manifest_raw,state_raw,checkpoints=checkpoints,capsules=capsules,decisions=decisions)
    state=strict_loads(state_raw,max_bytes=4194304);dataset=strict_loads(dataset_raw,max_bytes=4194304)
    if dataset!=empty_manifest() or state.get('dataset_root')!={'state':'EMPTY','sha256':dataset['manifest_sha256']}:
        raise RunnerSnapshotError('EMPTY_INITIAL_DATASET_BINDING_MISMATCH')
    manifest=strict_loads(manifest_raw,max_bytes=4194304)
    nodes={n['canonical_id']:n for n in manifest['nodes']};expected={}
    for raw in checkpoints:
        cp=strict_loads(raw,max_bytes=4194304);a=cp['acceptance'];mode=a.get('mode')
        if mode=='AUTOMATIC_WORKFLOW_GATE':
            if nodes[cp['node_id']]['node_kind']!='WORKFLOW_GATE' or set(a)!={'mode','outcome','science_authority_effect'} or a['outcome']!='CERTIFIED_REPLAY_PASS' or a['science_authority_effect']!='NONE':
                raise RunnerSnapshotError('INVALID_GATE_RESULT_PROFILE')
            continue
        if mode not in {'AUTOMATIC_HISTORICAL_REPLAY','AUTOMATIC_FRESH_RECOMPUTATION','EXTERNAL_AUDIT_DECISION'} or not isinstance(a.get('node_result'),dict):
            raise RunnerSnapshotError('UNSUPPORTED_CHECKPOINT_RESULT_PROFILE')
        expected[cp['checkpoint_sha256']]=(cp['node_id'],a['node_result'])
    for raw in capsules:
        cp=strict_loads(raw,max_bytes=4194304)
        if cp.get('node_result') is not None:expected[cp['audit_capsule_sha256']]=(cp['stopped_node_id'],cp['node_result'])
    if set(expected)!=set(bindings):raise RunnerSnapshotError('RESULT_BINDING_INVENTORY_MISMATCH')
    verified=[]
    for owner,(node_id,result) in expected.items():
        if result.get('schema_id')!=RESULT_SCHEMA or result.get('node_id')!=node_id or type(result.get('attempt')) is not int or result['attempt']<1:
            raise RunnerSnapshotError('INVALID_BOUND_NODE_RESULT')
        binding=bindings[owner];records={};seals={};inventory=[]
        for raw in binding['records']:
            obj=strict_loads(raw,max_bytes=4194304)
            if not isinstance(obj,dict):raise RunnerSnapshotError('RESULT_RECORD_OBJECT_REQUIRED')
            rec=verify_reference_record(obj);rid=rec['record_id'];digest=rec['record_sha256']
            if rid in records or digest in seals:raise RunnerSnapshotError('DUPLICATE_RESULT_RECORD')
            records[rid]=rec;seals[digest]=rec
            inventory.append({'record_id':rid,'record_sha256':digest,'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}})
        comparison=seals.get(result.get('result_sha256'))
        if comparison is None or comparison['record_type']!='COMPARISON' or comparison['payload']['obligation_id']!=node_id:
            raise RunnerSnapshotError('RESULT_COMPARISON_BINDING_MISMATCH')
        p=comparison['payload'];historical=p['historical_record_ids'];replay=p['replay_record_ids']
        if not historical or len(historical)!=len(replay):raise RunnerSnapshotError('UNSUPPORTED_COMPARISON_CARDINALITY')
        needed={comparison['record_id'],*historical,*replay}
        for rid in historical+replay:
            rec=records.get(rid)
            if rec is None or rec['record_type']!='SRCF_EVIDENCE' or rec['payload']['obligation_id']!=node_id:
                raise RunnerSnapshotError('RESULT_EVIDENCE_BINDING_MISMATCH')
        audit_hash=result.get('audit_authorization_record_sha256')
        if audit_hash is not None:
            audit=seals.get(audit_hash)
            if audit is None or audit['record_type']!='AUDIT_AUTHORIZATION' or node_id not in audit['payload']['obligation_ids']:
                raise RunnerSnapshotError('RESULT_AUDIT_RECORD_MISMATCH')
            needed.add(audit['record_id'])
        if set(records)!=needed:raise RunnerSnapshotError('RESULT_RECORD_INVENTORY_MISMATCH')
        profile=binding['profile']
        if profile=='SINGLE_REPLAY_RECORD_V1':
            if len(replay)!=1:raise RunnerSnapshotError('SINGLE_REPLAY_CARDINALITY')
            digest=records[replay[0]]['record_sha256']
        else:
            ids=replay if profile=='ORDERED_REPLAY_SEALS_V1' else [rid for pair in zip(historical,replay) for rid in pair]
            digest=canonical_sha256([records[rid]['record_sha256'] for rid in ids])
        if result.get('evidence_sha256')!=digest:raise RunnerSnapshotError('RESULT_EVIDENCE_SEAL_MISMATCH')
        verified.append({'owner_sha256':owner,'node_id':node_id,'profile':profile,'records':inventory})
    return {'status':'SNAPSHOT_RESULT_IDENTITIES_VERIFIED','scope':'RESULT_RECORD_IDENTITIES_AND_EMPTY_INITIAL_ROOT_ONLY',
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED','dependency_closure_verified':False,
        'snapshot':snapshot,'bindings_verified':verified,'records_checked':len(raws),'bytes_checked':total,
        'initial_dataset_content_ref':{'sha256':hashlib.sha256(dataset_raw).hexdigest(),'size_bytes':str(len(dataset_raw))}}
