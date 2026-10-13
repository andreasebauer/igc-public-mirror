"""Bind native collection membership to verified complete recovery inputs.

Frontiers retain their dedicated snapshot verifier; no opaque bypass is used.
"""
import hashlib
from .storage_schema import strict_loads
from .storage_catalog import inventory
from .storage_collections import collection_records,CollectionLimits
from .storage_full_recovery_join import verify_full_recovery_inventory
from .storage_closure import ClosureError


def verify_collection_recovery(store,collection,result_request_ref,auxiliary_request_ref,*,
        purpose,recovery_inputs,max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise ClosureError('INVALID_COLLECTION_RECOVERY_BUDGET')
    fields={'snapshot_inputs','dataset_inputs','frontiers','frontier_snapshots','current_frontier_id','catalogue_raw','sources'}
    if type(recovery_inputs) is not dict or set(recovery_inputs)!=fields:
        raise ClosureError('INVALID_COLLECTION_RECOVERY_INPUTS')
    if type(collection) is not dict or collection.get('record_schema_id')!='IG_REPLAY_REFERENCE_RECORD_V1':
        raise ClosureError('NATIVE_RECOVERY_COLLECTION_REQUIRED')
    full=verify_full_recovery_inventory(store,result_request_ref,auxiliary_request_ref,**recovery_inputs,max_total_bytes=max_total_bytes)
    expected={}
    def row(raw):
        r=strict_loads(raw,max_bytes=4194304)
        return r['record_id'],{'record_id':r['record_id'],'record_sha256':r['record_sha256'],
            'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}}
    for raw in recovery_inputs['dataset_inputs']['records']:
        rid,ref=row(raw);expected[rid]=ref
    remaining=max_total_bytes-full['bytes_checked']
    if remaining<1:raise ClosureError('COLLECTION_RECOVERY_BYTE_BUDGET')
    spent=0;actual={}
    def observed(ref,role,raw):
        nonlocal spent
        spent+=len(raw)
    limits=CollectionLimits(max_total_bytes=remaining,max_records=min(100000,len(expected)+1))
    def records():
        for item in collection_records(store,collection,collection_kind='reference_records',purpose=purpose,limits=limits,on_read=observed):
            if item.profile!='LEGACY_REFERENCE_RECORD_V1':raise ClosureError('NATIVE_RECOVERY_PROFILE_REQUIRED')
            rid,ref=row(item.raw)
            if rid in actual:raise ClosureError('DUPLICATE_RECOVERY_COLLECTION_RECORD')
            actual[rid]=ref
            yield item
    logical=inventory(records())
    if actual!=expected:raise ClosureError('COLLECTION_RECOVERY_EXACT_INVENTORY_MISMATCH')
    return {'status':'COLLECTION_RECOVERY_VERIFIED',
        'scope':'NATIVE_COLLECTION_EXACT_MEMBERSHIP_AND_VERIFIED_DECLARED_RECOVERY_CLOSURE',
        'collection':collection,'purpose':purpose,'collection_inventory':logical,
        'record_count':len(actual),'recovery_verification':full,
        'collection_bytes_checked':spent,'bytes_checked':full['bytes_checked']+spent,'byte_budget':max_total_bytes,
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED','recovery_performed':False,
        'production_release_verified':False,'accepted_head_freshness_verified':False}
