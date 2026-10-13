"""Review exact fixture admission bindings without authenticating authority.

Hashes prove content identity, not who authorized it. This API never grants
science execution or publication, never writes a seal and never applies a
runner decision. Production policy/report profiles are intentionally unhandled.
"""
from .storage_schema import canonical_bytes,strict_loads,validate_record_bytes
from .storage_catalog import _ref
from .storage_index_descriptor import content_ref
from .storage_native_records import native_record
from .storage_legacy import reference


class AdmissionBindingError(ValueError):
    pass


SCOPE='REFERENCE_ONLY_FIXTURE_ROOT_RECOVERY_AND_INDEX'


def review_admission_bindings(store,root_ref,policy_ref,report_ref,decision_refs,*,
        max_total_bytes=8388608):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=8388608:
        raise AdmissionBindingError('INVALID_ADMISSION_BINDING_BUDGET')
    if type(decision_refs) is not list or not 1<=len(decision_refs)<=128:
        raise AdmissionBindingError('DECISION_INVENTORY_REQUIRED')
    used=0
    def read(ref,bound):
        nonlocal used
        _ref(ref);size=int(ref['size_bytes'])
        if size>bound or used+size>max_total_bytes:raise AdmissionBindingError('ADMISSION_BINDING_BYTE_BUDGET')
        used+=size;raw=store.read(ref,bound)
        if type(raw) is not bytes or content_ref(raw)!=ref:raise AdmissionBindingError('ADMISSION_PROVIDER_MISMATCH')
        return raw
    def canonical(ref,bound):
        raw=read(ref,bound);obj=strict_loads(raw,max_bytes=bound)
        if type(obj) is not dict or canonical_bytes(obj)!=raw:raise AdmissionBindingError('NONCANONICAL_ADMISSION_DOCUMENT')
        return obj
    root=validate_record_bytes(read(root_ref,65536),max_bytes=65536)
    if (root['schema_id']!='IG_STORAGE_RELEASEROOT_V1' or root['purpose']!='SCHEMA_FIXTURE'
        or root['extensions'] or root['acceptance_policy_ref']!=policy_ref):
        raise AdmissionBindingError('ROOT_POLICY_OR_PROFILE_MISMATCH')
    policy=canonical(policy_ref,65536)
    if (set(policy)!={'schema_id','purpose','verification_scope','required_obligation_ids','required_decision'}
        or policy['schema_id']!='IG_STORAGE_ADMISSION_BINDING_POLICY_V1'
        or policy['purpose']!='SCHEMA_FIXTURE' or policy['verification_scope']!=SCOPE
        or policy['required_decision']!='CERTIFY_AND_ADVANCE'):
        raise AdmissionBindingError('UNSUPPORTED_ADMISSION_POLICY')
    obligations=policy['required_obligation_ids']
    if (type(obligations) is not list or not 1<=len(obligations)<=128
        or any(type(s) is not str or not s or len(s)>512 for s in obligations)
        or obligations!=sorted(set(obligations))):
        raise AdmissionBindingError('INVALID_POLICY_OBLIGATIONS')
    report=canonical(report_ref,4194304)
    if (report.get('status')!='REFERENCE_CANDIDATE_VERIFIED' or report.get('scope')!=SCOPE
        or report.get('candidate_ref')!=root_ref or report.get('scientific_acceptance')!='NOT_GRANTED'
        or any(report.get(k) is not False for k in ('production_release_verified',
            'metadata_semantics_verified','execution_authorized','publication_seal_verified','recovery_performed'))):
        raise AdmissionBindingError('REPORT_ROOT_SCOPE_OR_AUTHORITY_MISMATCH')
    # Only document bindings are reviewed here. Truth of the candidate report
    # must be established by its independent verifier, not this status string.
    expected_scope={'release_root':root_ref,'acceptance_policy_ref':policy_ref,
        'verification_report_ref':report_ref,'purpose':'SCHEMA_FIXTURE','verification_scope':SCOPE}
    seen_ids=set();seen_auth=set();covered=set();reviewed=[]
    for entry in decision_refs:
        reference(entry);raw=read(entry['content_ref'],65536);rec,_=native_record(raw,entry=entry)
        payload=rec['payload']
        if (rec['record_type']!='AUDIT_AUTHORIZATION' or rec['epistemic_status']!='EXTERNAL_DECISION'
            or rec['provenance']['status']!='PINNED' or rec['science_execution']!='NONE'):
            raise AdmissionBindingError('AUDIT_DECISION_PROFILE_REQUIRED')
        if rec['record_id'] in seen_ids or payload['authorization_id'] in seen_auth:
            raise AdmissionBindingError('DUPLICATE_DECISION_IDENTITY')
        seen_ids.add(rec['record_id']);seen_auth.add(payload['authorization_id'])
        if (payload['authorized_scope']!=expected_scope or rec['scope']!=expected_scope
            or payload['decision']!=policy['required_decision']):
            raise AdmissionBindingError('DECISION_ROOT_POLICY_REPORT_SCOPE_MISMATCH')
        if report_ref['sha256'] not in {r['sha256'] for r in payload['evidence_hashes']}:
            raise AdmissionBindingError('DECISION_REPORT_EVIDENCE_REQUIRED')
        ids=payload['obligation_ids']
        if len(ids)!=len(set(ids)) or not set(ids)<=set(obligations) or covered.intersection(ids):
            raise AdmissionBindingError('DECISION_OBLIGATION_OVERLAP_OR_EXTRA')
        covered.update(ids)
        reviewed.append({'record_ref':entry,'authorization_id':payload['authorization_id'],
            'limitations':payload['limitations'],'nonclaims':rec['nonclaims']})
    if covered!=set(obligations):raise AdmissionBindingError('MISSING_REQUIRED_DECISION')
    return {'status':'BINDINGS_REVIEWED_AUTHORITY_NOT_VERIFIED','scope':'SCHEMA_FIXTURE_BINDINGS_ONLY',
        'root_ref':root_ref,'policy_ref':policy_ref,'report_ref':report_ref,'decisions':reviewed,
        'obligations':obligations,'bytes_checked':used,'byte_budget':max_total_bytes,
        'authority_authenticated':False,'verification_report_truth_verified':False,
        'production_release_verified':False,'execution_authorized':False,
        'publication_authorized':False,'scientific_acceptance':'NOT_GRANTED',
        'remaining':['INDEPENDENT_REPORT_VERIFICATION','AUTHORIZED_DECISION_CHANNEL','PRODUCTION_POLICY_PROFILE']}
