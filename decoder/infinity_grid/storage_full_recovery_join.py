"""Join exact dataset, results, auxiliary closure and historical frontiers.

Read-only verification of supplied content. No current-head or release authority.
"""
import hashlib
from .storage_schema import strict_loads
from .storage_manifest_sources import verify_manifest_sources
from .storage_result_closure import verify_snapshot_result_closure
from .storage_frontier_history import verify_frontier_history
from .storage_dataset_snapshot import verify_dataset_snapshot
from .storage_closure import verify_declared_closure,ClosureLimits,ClosureError


def verify_full_recovery_inventory(store,result_request_ref,auxiliary_request_ref,*,
        snapshot_inputs,dataset_inputs,frontiers,frontier_snapshots,current_frontier_id,
        catalogue_raw,sources,max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise ClosureError('INVALID_FULL_RECOVERY_BUDGET')
    if type(snapshot_inputs) is not dict or type(dataset_inputs) is not dict:
        raise ClosureError('INVALID_FULL_RECOVERY_INPUTS')
    used=0
    def remaining():
        if used>=max_total_bytes:raise ClosureError('FULL_RECOVERY_BYTE_BUDGET')
        return max_total_bytes-used
    pins=verify_manifest_sources(snapshot_inputs['manifest_raw'],catalogue_raw,sources,max_total_bytes=remaining());used+=pins['bytes_checked']
    results=verify_snapshot_result_closure(store,result_request_ref,snapshot_inputs=snapshot_inputs,max_total_bytes=remaining());used+=results['bytes_checked']
    auxiliary=verify_declared_closure(store,auxiliary_request_ref,limits=ClosureLimits(max_total_bytes=remaining()));used+=auxiliary['bytes_read']
    history=verify_frontier_history(snapshot_inputs['manifest_raw'],frontiers,frontier_snapshots,current_frontier_id=current_frontier_id,max_total_bytes=remaining());used+=history['bytes_checked']
    current=frontier_snapshots[current_frontier_id]
    if current['state_raw']!=snapshot_inputs['state_raw']:
        raise ClosureError('FULL_RECOVERY_CURRENT_STATE_MISMATCH')
    for key in ('checkpoints','capsules','decisions'):
        if sorted(current[key])!=sorted(snapshot_inputs.get(key,())):
            raise ClosureError('FULL_RECOVERY_CURRENT_INPUT_MISMATCH')
    dataset=verify_dataset_snapshot(**dataset_inputs,max_total_bytes=remaining());used+=dataset['bytes_checked']
    expected={};classes={}
    def add(row,kind):
        rid=row['record_id']
        if rid in expected:raise ClosureError('FULL_RECOVERY_COMPONENT_OVERLAP')
        expected[rid]=row;classes[rid]=kind
    for row in results['declared_closure']['legacy_records_verified']:add(row,'RESULT')
    for row in auxiliary['legacy_records_verified']:add(row,'AUXILIARY')
    for rid,out in history['frontiers'].items():
        add({'record_id':rid,'record_sha256':out['record_sha256'],'content_ref':out['frontier_content_ref']},'FRONTIER')
    actual={}
    for raw in dataset_inputs['records']:
        r=strict_loads(raw,max_bytes=4194304);rid=r['record_id']
        actual[rid]={'record_id':rid,'record_sha256':r['record_sha256'],'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}}
        if classes.get(rid)=='AUXILIARY' and r['record_type'] not in {'CANONICAL_OBJECT','GENERATION_RECIPE','NEGATIVE_RESULT'}:
            raise ClosureError('FULL_RECOVERY_UNSUPPORTED_AUXILIARY_TYPE')
    if actual!=expected:raise ClosureError('FULL_RECOVERY_EXACT_INVENTORY_MISMATCH')
    if not dataset['record_ids'] or dataset['record_ids'][-1]!=current_frontier_id:
        raise ClosureError('FULL_RECOVERY_CURRENT_FRONTIER_NOT_FINAL')
    return {'status':'FULL_RECOVERY_INVENTORY_VERIFIED',
        'scope':'EXACT_DATASET_HISTORY_AND_DECLARED_REQUIRED_CONTENT',
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED',
        'recovery_performed':False,'accepted_head_freshness_verified':False,'production_release_verified':False,
        'record_count':len(expected),'record_classes':classes,
        'manifest_sources':pins,'result_closure':results,'auxiliary_closure':auxiliary,
        'frontier_history':history,'dataset_snapshot':dataset,'bytes_checked':used,'byte_budget':max_total_bytes}
