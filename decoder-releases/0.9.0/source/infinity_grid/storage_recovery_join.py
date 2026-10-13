"""Cross-bind supported recovery components without granting resume authority."""
import hashlib
from .storage_manifest_sources import verify_manifest_sources
from .storage_frontier_snapshot import verify_frontier_snapshot
from .storage_result_closure import verify_snapshot_result_closure
from .storage_dataset_snapshot import verify_dataset_snapshot
from .storage_schema import strict_loads
from .storage_closure import ClosureError


def verify_supported_recovery(store,request_ref,*,snapshot_inputs,dataset_inputs,
                              frontier_raw,catalogue_raw,sources,max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise ClosureError('INVALID_RECOVERY_JOIN_BUDGET')
    if not isinstance(snapshot_inputs,dict) or not set(snapshot_inputs)<= {'manifest_raw','state_raw','dataset_raw','bindings','checkpoints','capsules','decisions'}:
        raise ClosureError('INVALID_RECOVERY_SNAPSHOT_INPUTS')
    if not isinstance(dataset_inputs,dict) or not set(dataset_inputs)<= {'manifest_raw','records','transactions','commits'}:
        raise ClosureError('INVALID_RECOVERY_DATASET_INPUTS')
    used=0
    def remaining():
        if used>=max_total_bytes:raise ClosureError('RECOVERY_JOIN_BYTE_BUDGET')
        return max_total_bytes-used
    pins=verify_manifest_sources(snapshot_inputs['manifest_raw'],catalogue_raw,sources,max_total_bytes=remaining())
    used+=pins['bytes_checked']
    frontier=verify_frontier_snapshot(frontier_raw,snapshot_inputs['manifest_raw'],snapshot_inputs['state_raw'],
        **{k:snapshot_inputs[k] for k in ('checkpoints','capsules','decisions') if k in snapshot_inputs},max_total_bytes=remaining())
    used+=frontier['bytes_checked']
    results=verify_snapshot_result_closure(store,request_ref,snapshot_inputs=snapshot_inputs,max_total_bytes=remaining())
    used+=results['bytes_checked']
    dataset=verify_dataset_snapshot(**dataset_inputs,max_total_bytes=remaining())
    used+=dataset['bytes_checked']
    expected={r['record_id']:r for r in results['declared_closure']['legacy_records_verified']}
    frow={'record_id':frontier['record_id'],'record_sha256':frontier['record_sha256'],
        'content_ref':frontier['frontier_content_ref']}
    if frow['record_id'] in expected:raise ClosureError('RECOVERY_FRONTIER_RECORD_COLLISION')
    expected[frow['record_id']]=frow
    actual={}
    for raw in dataset_inputs.get('records',()):
        record=strict_loads(raw,max_bytes=4194304)
        actual[record['record_id']]={'record_id':record['record_id'],'record_sha256':record['record_sha256'],
            'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}}
    if actual!=expected:raise ClosureError('RECOVERY_DATASET_RECORD_BINDING_MISMATCH')
    if dataset['record_ids'][-1]!=frontier['record_id']:raise ClosureError('RECOVERY_FRONTIER_NOT_FINAL_RECORD')
    return {'status':'SUPPORTED_RECOVERY_BUNDLE_VERIFIED',
        'scope':'SUPPORTED_PROFILES_WITH_EXACT_DATASET_AND_DECLARED_DEPENDENCIES',
        'scientific_acceptance':'NOT_GRANTED','execution_authorized':False,
        'production_release_verified':False,'accepted_head_freshness_verified':False,
        'manifest_sources':pins,'frontier_binding':frontier,'result_closure':results,
        'dataset_snapshot':dataset,'bytes_checked':used,'byte_budget':max_total_bytes}
