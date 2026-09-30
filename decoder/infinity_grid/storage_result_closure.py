"""Join verified snapshot result identities to their declared content closure."""
import hashlib
from .storage_snapshot_results import verify_snapshot_results
from .storage_closure import verify_declared_closure, ClosureLimits, ClosureError
from .storage_schema import strict_loads
from .storage_catalog import _ref


def verify_snapshot_result_closure(store,request_ref,*,snapshot_inputs,
                                   max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise ClosureError('INVALID_JOINED_BYTE_BUDGET')
    fields={'manifest_raw','state_raw','dataset_raw','bindings','checkpoints','capsules','decisions'}
    if not isinstance(snapshot_inputs,dict) or not set(snapshot_inputs)<=fields:
        raise ClosureError('INVALID_SNAPSHOT_INPUTS')
    snapshot=verify_snapshot_results(**snapshot_inputs,max_total_bytes=max_total_bytes)
    expected={}
    for binding in snapshot['bindings_verified']:
        for row in binding['records']:
            prior=expected.get(row['record_id'])
            if prior is not None and prior!=row:raise ClosureError('CONFLICTING_SNAPSHOT_RESULT_RECORD')
            expected[row['record_id']]=row
    if not expected:raise ClosureError('RESULT_CLOSURE_ROOTS_REQUIRED')
    remaining=max_total_bytes-snapshot['bytes_checked']
    if remaining<1:raise ClosureError('JOINED_BYTE_BUDGET')
    class BudgetedStore:
        used=0
        def read(self,ref,bound):
            _ref(ref);size=int(ref['size_bytes'])
            if size>bound or self.used+size>remaining:raise ClosureError('JOINED_BYTE_BUDGET')
            self.used+=size
            raw=store.read(ref,bound)
            if type(raw) is not bytes or len(raw)!=size or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
                raise ClosureError('PROVIDER_CONTENT_MISMATCH')
            return raw
    provider=BudgetedStore()
    request=strict_loads(provider.read(request_ref,1048576),max_bytes=1048576)
    roots=request.get('roots') if isinstance(request,dict) else None
    if not isinstance(roots,list):raise ClosureError('RESULT_CLOSURE_ROOT_MISMATCH')
    for ref in roots:_ref(ref)
    actual={(r['sha256'],r['size_bytes']) for r in roots}
    wanted={(r['content_ref']['sha256'],r['content_ref']['size_bytes']) for r in expected.values()}
    if len(actual)!=len(roots) or actual!=wanted:raise ClosureError('RESULT_CLOSURE_ROOT_MISMATCH')
    closure=verify_declared_closure(provider,request_ref,limits=ClosureLimits(max_total_bytes=remaining))
    verified={r['record_id']:r for r in closure['legacy_records_verified']}
    if any(verified.get(rid)!=row for rid,row in expected.items()):
        raise ClosureError('RESULT_CLOSURE_NATIVE_BINDING_MISMATCH')
    return {'status':'SNAPSHOT_RESULT_DECLARED_CLOSURE_VERIFIED',
        'scope':'SNAPSHOT_RESULT_RECORDS_AND_DECLARED_DEPENDENCIES_ONLY',
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED',
        'full_frontier_closure_verified':False,'populated_dataset_verified':False,
        'snapshot_result_verification':snapshot,'declared_closure':closure,
        'bytes_checked':snapshot['bytes_checked']+provider.used,
        'byte_budget':max_total_bytes,'result_roots_verified':len(expected)}
