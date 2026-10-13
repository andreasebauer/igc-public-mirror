"""Bind a declared frontier inventory to original snapshots without advancing it."""
from .storage_schema import strict_loads
from .storage_frontier_snapshot import verify_frontier_snapshot


class FrontierHistoryError(ValueError):
    pass


def verify_frontier_history(manifest_raw,frontiers,snapshots,*,current_frontier_id,max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise FrontierHistoryError('INVALID_HISTORY_BUDGET')
    if not isinstance(frontiers,(list,tuple)) or not 1<=len(frontiers)<=256 or type(snapshots) is not dict:
        raise FrontierHistoryError('INVALID_FRONTIER_INVENTORY')
    if type(manifest_raw) is not bytes or any(type(b) is not bytes for b in frontiers):
        raise FrontierHistoryError('FRONTIER_BYTES_REQUIRED')
    if len(manifest_raw)+sum(map(len,frontiers))>max_total_bytes:
        raise FrontierHistoryError('FRONTIER_HISTORY_BYTE_BUDGET')
    ids=[]
    for raw in frontiers:
        obj=strict_loads(raw,max_bytes=4194304)
        if not isinstance(obj,dict) or type(obj.get('record_id')) is not str:raise FrontierHistoryError('INVALID_FRONTIER_RECORD')
        ids.append(obj['record_id'])
    if len(set(ids))!=len(ids):raise FrontierHistoryError('DUPLICATE_FRONTIER_RECORD')
    if set(snapshots)!=set(ids) or current_frontier_id not in ids:
        raise FrontierHistoryError('FRONTIER_SNAPSHOT_INVENTORY_MISMATCH')
    results={};states={};spent=0
    for rid,raw in zip(ids,frontiers):
        inputs=snapshots[rid]
        if type(inputs) is not dict or set(inputs)!={'state_raw','checkpoints','capsules','decisions'}:
            raise FrontierHistoryError('INVALID_SNAPSHOT_BINDING')
        if max_total_bytes-spent<3:raise FrontierHistoryError('FRONTIER_HISTORY_BYTE_BUDGET')
        out=verify_frontier_snapshot(raw,manifest_raw,**inputs,max_total_bytes=max_total_bytes-spent)
        spent+=out['bytes_checked'];results[rid]=out
        states[rid]=strict_loads(inputs['state_raw'],max_bytes=4194304)
    current=states[current_frontier_id];completed=current['completed_node_ids'];seen=set()
    for rid,state in states.items():
        seal=state['state_sha256']
        if seal in seen:raise FrontierHistoryError('DUPLICATE_FRONTIER_STATE')
        seen.add(seal);prefix=state['completed_node_ids']
        if state['root_run_id']!=current['root_run_id'] or completed[:len(prefix)]!=prefix:
            raise FrontierHistoryError('FRONTIER_HISTORY_PREFIX_MISMATCH')
        if rid!=current_frontier_id and len(prefix)>=len(completed):
            raise FrontierHistoryError('HISTORICAL_FRONTIER_NOT_EARLIER')
        if any(current['accepted_checkpoint_sha256_by_node'].get(n)!=h for n,h in state['accepted_checkpoint_sha256_by_node'].items()):
            raise FrontierHistoryError('FRONTIER_HISTORY_CHECKPOINT_MISMATCH')
    return {'status':'FRONTIER_HISTORY_BOUND','scope':'DECLARED_FRONTIER_INVENTORY_AND_RUNNER_SNAPSHOTS_ONLY',
        'current_frontier_id':current_frontier_id,'historical_frontier_ids':sorted(set(ids)-{current_frontier_id}),
        'frontiers':results,'bytes_checked':spent,'execution_authorized':False,
        'scientific_acceptance':'NOT_GRANTED','accepted_head_freshness_verified':False,
        'production_recovery_verified':False,'dependency_closure_verified':False}
