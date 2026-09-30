"""Reference-only fixture candidate assembly and joined read verification.

This profile does not certify native science collections, metadata semantics,
transitive environment dependencies, publication authority or a master block.
Reports and index descriptors are made after the immutable candidate root.
"""
from .storage_schema import canonical_bytes, validate_record_bytes
from .storage_catalog import _ref, inventory
from .storage_collections import collection_records, CollectionLimits
from .storage_collection_recovery import verify_collection_recovery
from .storage_index_descriptor import content_ref, verify_index_descriptor
from .storage_closure import ClosureError

PURPOSE='SCHEMA_FIXTURE'
LABEL='REFERENCE_RECOVERY_CANDIDATE_NOT_A_MASTER_BLOCK'
CAPTURE_LABEL='REFERENCE_RECOVERY_CAPTURE_CANDIDATE_NOT_A_MASTER_BLOCK'
REGISTRY_LABEL='REFERENCE_RECOVERY_REGISTRY_CANDIDATE_NOT_A_MASTER_BLOCK'
META={'contract_freeze_ref','source_ref','environment_ref','registry_ref','acceptance_policy_ref'}
EMPTY={'layers':('LAYER','SET'),'objects':('OBJECT','SET'),
       'occurrences':('OCCURRENCE','OCCURRENCE_LOG'),'relations':('RELATION','SET'),
       'incidences':('INCIDENCE','ORDERED'),'coverage':('COVERAGE','SET')}


class _Budget:
    def __init__(self,limit):
        if type(limit) is not int or not 1<=limit<=67108864:raise ClosureError('INVALID_CANDIDATE_BUDGET')
        self.limit=limit;self.used=0
    def charge(self,n):
        if self.used+n>self.limit:raise ClosureError('CANDIDATE_BYTE_BUDGET')
        self.used+=n
    def remaining(self):
        if self.used>=self.limit:raise ClosureError('CANDIDATE_BYTE_BUDGET')
        return self.limit-self.used


def _read(store,ref,budget,bound=4194304):
    _ref(ref);n=int(ref['size_bytes'])
    if n>bound:raise ClosureError('CANDIDATE_CONTENT_BUDGET')
    budget.charge(n);raw=store.read(ref,bound)
    if type(raw) is not bytes or content_ref(raw)!=ref:raise ClosureError('CANDIDATE_PROVIDER_MISMATCH')
    return raw


def _collection_blobs(kind,sid,semantics,rows):
    profile={'schema_id':'IG_STORAGE_COLLECTION_KEY_PROFILE_V1','format':'CANONICAL_STORAGE_JSONL_V1',
        'key_encoding':'CANONICAL_STRING_ARRAY_UTF8_HEX','order':'ASCII_BYTEWISE',
        'bounds':'INCLUSIVE_DISJOINT','key_fields':[['content_ref','sha256']]}
    blobs=[canonical_bytes(profile)];pref=content_ref(blobs[0]);entries=[]
    rows=sorted(rows,key=lambda r:canonical_bytes([r['content_ref']['sha256']]).hex())
    for start in range(0,len(rows),64):
        group=rows[start:start+64];raw=b''.join(canonical_bytes(r)+b'\n' for r in group);blobs.append(raw)
        entries.append({'entry_type':'SHARD','content_ref':content_ref(raw),
            'first_key':canonical_bytes([group[0]['content_ref']['sha256']]).hex(),
            'last_key':canonical_bytes([group[-1]['content_ref']['sha256']]).hex(),'record_count':str(len(group))})
    page={'schema_id':'IG_STORAGE_COLLECTIONPAGE_V1','contract_version':'1.0.0','purpose':PURPOSE,
        'collection_kind':kind,'record_schema_id':sid,'key_definition_ref':pref,'depth':0,
        'first_key':entries[0]['first_key'] if entries else None,'last_key':entries[-1]['last_key'] if entries else None,
        'record_count':str(len(rows)),'entries':entries,'extensions':[]}
    blobs.append(canonical_bytes(page))
    ref={'root':content_ref(blobs[-1]),'record_schema_id':sid,'row_count':str(len(rows)),
        'key_definition_ref':pref,'cardinality_semantics':semantics}
    return ref,blobs


def _prepare(store,collection,metadata,result_ref,auxiliary_ref,recovery_inputs,budget,writer=None,capture_ref=None,contract_registry=False):
    if type(metadata) is not dict or set(metadata)!=META:raise ClosureError('CANDIDATE_METADATA_FIELDS')
    required={};structural=set();nodes=0
    def remember(ref):
        _ref(ref);key=ref['sha256']
        if key in required and required[key]!=ref:raise ClosureError('CANDIDATE_LENGTH_CONFLICT')
        required[key]=dict(ref)
        if len(required)>8192:raise ClosureError('CANDIDATE_OBJECT_BUDGET')
    def write(raw):
        ref=content_ref(raw)
        if ref['sha256'] in required:
            if required[ref['sha256']]!=ref:raise ClosureError('CANDIDATE_LENGTH_CONFLICT')
            return ref
        if writer is not None:
            if writer(raw)!=ref:raise ClosureError('CANDIDATE_WRITER_MISMATCH')
        if _read(store,ref,budget)!=raw:raise ClosureError('CANDIDATE_INPUT_BYTES_MISMATCH')
        remember(ref);return ref
    # Tagged encoding preserves the exact input tree, including byte/string and
    # list/tuple distinctions. Nothing from this tree is evaluated as code.
    def encode(value,depth=0):
        nonlocal nodes
        nodes+=1
        if depth>32 or nodes>100000:raise ClosureError('CANDIDATE_INPUT_STRUCTURE_BUDGET')
        if type(value) is bytes:return ['bytes',write(value)]
        if type(value) is dict:
            if any(type(k) is not str for k in value):raise ClosureError('CANDIDATE_INPUT_KEY_TYPE')
            return ['dict',[[k,encode(value[k],depth+1)] for k in sorted(value)]]
        if type(value) in (list,tuple):return ['list' if type(value) is list else 'tuple',[encode(v,depth+1) for v in value]]
        if value is None or type(value) in (str,int,bool):return ['scalar',value]
        raise ClosureError('CANDIDATE_INPUT_TYPE')
    tree=encode(recovery_inputs)
    recipe=canonical_bytes({'schema_id':'IG_REFERENCE_CANDIDATE_RECIPE_V1','purpose':PURPOSE,
        'result_request_ref':result_ref,'auxiliary_request_ref':auxiliary_ref,'recovery_inputs':tree})
    recipe_ref=write(recipe)
    for ref in metadata.values():_read(store,ref,budget);remember(ref)
    class Observed:
        def read(self,ref,bound):
            raw=store.read(ref,bound)
            if type(raw) is not bytes or content_ref(raw)!=ref:raise ClosureError('CANDIDATE_PROVIDER_MISMATCH')
            remember(ref);return raw
    capture_report=None;registry_report=None
    if capture_ref is not None:
        from .storage_capture_metadata import verify_capture_metadata
        from .storage_contract_registry import SOURCE_PATHS,verify_contract_registry
        capture_report=verify_capture_metadata(Observed(),capture_ref,max_total_bytes=budget.remaining(),
            source_paths=SOURCE_PATHS if contract_registry else ())
        budget.charge(capture_report['bytes_checked'])
        # The root metadata pins explicit descriptors, not an unchecked report.
        # Recompute the report from its bytes and compare exact descriptor bytes.
        source,environment=capture_metadata_descriptors(capture_report)
        for key,expected in [('source_ref',source),('environment_ref',environment)]:
            if _read(store,metadata[key],budget)!=expected:
                raise ClosureError('CANDIDATE_CAPTURE_METADATA_BINDING')
        if contract_registry:
            registry_report=verify_contract_registry(Observed(),metadata['contract_freeze_ref'],
                metadata['registry_ref'],capture_report,max_total_bytes=budget.remaining())
            budget.charge(registry_report['bytes_checked'])
    # This verifier already accounts all reads; avoid charging them twice here.
    recovery=verify_collection_recovery(Observed(),collection,result_ref,auxiliary_ref,
        purpose=PURPOSE,recovery_inputs=recovery_inputs,max_total_bytes=budget.remaining())
    budget.charge(recovery['bytes_checked'])
    def observe(ref,role,raw):
        budget.charge(len(raw));remember(ref)
        if role=='COLLECTION_PAGE':structural.add(ref['sha256'])
    logical=inventory(collection_records(store,collection,collection_kind='reference_records',purpose=PURPOSE,
        limits=CollectionLimits(max_total_bytes=budget.remaining()),on_read=observe))
    if logical!=recovery['collection_inventory']:raise ClosureError('CANDIDATE_COLLECTION_CHANGED')
    # Native collection pages are discovered structurally. The inventory's own
    # profile/pages/shards and the root are also structural, never self-listed.
    for sha in structural:required.pop(sha,None)
    rows=[{'schema_id':'IG_STORAGE_REQUIREDCONTENT_V1','contract_version':'1.0.0','purpose':PURPOSE,
        'content_ref':required[k],'content_role':'RECONSTRUCTION',
        'interpretation_ref':{'state':'NOT_APPLICABLE','content_ref':None,
            'explanation':'Exact bytes under the bounded reference-candidate recipe; no generic interpretation claim'},
        'extensions':[]} for k in sorted(required)]
    inv,blobs=_collection_blobs('required_content_inventory','IG_STORAGE_REQUIREDCONTENT_V1','SET',rows)
    empty={}
    for kind,(sid,semantics) in EMPTY.items():
        empty[kind],extra=_collection_blobs(kind,'IG_STORAGE_'+sid+'_V1',semantics,[]);blobs.extend(extra)
    root={'schema_id':'IG_STORAGE_RELEASEROOT_V1','contract_version':'1.0.0','purpose':PURPOSE,
        'label':_profile_label(capture_ref,contract_registry),**metadata,'recipe_ref':recipe_ref,'layers':empty.pop('layers'),
        'collections':dict(empty,reference_records=collection),'required_content_inventory':inv,
        'previous_release_root':None,'extensions':[]}
    raw=canonical_bytes(root);validate_record_bytes(raw)
    return raw,blobs,recovery,logical,len(required),capture_report,registry_report


def _profile_label(capture_ref,contract_registry):
    if type(contract_registry) is not bool or (contract_registry and capture_ref is None):
        raise ClosureError('CANDIDATE_REGISTRY_PROFILE_REQUIRES_CAPTURE')
    return REGISTRY_LABEL if contract_registry else CAPTURE_LABEL if capture_ref is not None else LABEL


def capture_metadata_descriptors(verified_capture_report):
    """Encode descriptor bytes; this convenience function grants no verification.

    The candidate verifier independently recalculates every report field used
    here. Caller-supplied report claims are never used as verification evidence.
    """
    r=verified_capture_report
    return (canonical_bytes({'schema_id':'IG_REFERENCE_CAPTURE_SOURCE_V1','purpose':PURPOSE,
                'capture_ref':r['capture_ref'],'capture_id':r['capture_id'],
                'source_archive_ref':r['source_ref'],'source_sha256':r['source_sha256'],
                'package_sha256':r['package_sha256']}),
            canonical_bytes({'schema_id':'IG_REFERENCE_CAPTURE_ENVIRONMENT_V1','purpose':PURPOSE,
                'capture_ref':r['capture_ref'],'capture_id':r['capture_id'],
                'environment_sha256':r['environment_sha256'],'runtime_binding_ref':r['runtime_binding_ref']}))


def assemble_reference_candidate(store,writer,*,collection,metadata,result_request_ref,
        auxiliary_request_ref,recovery_inputs,max_total_bytes=67108864,capture_ref=None,contract_registry=False):
    """Publish immutable candidate objects only; no index, report or seal exists yet.

    Writer must return the raw content reference. Every written object is read
    back through store before success; interruption leaves unaccepted objects.
    """
    _profile_label(capture_ref,contract_registry)
    budget=_Budget(max_total_bytes)
    raw,blobs,recovery,logical,count,capture_report,registry_report=_prepare(store,collection,metadata,result_request_ref,
        auxiliary_request_ref,recovery_inputs,budget,writer,capture_ref,contract_registry)
    for blob in blobs+[raw]:
        ref=content_ref(blob)
        if writer(blob)!=ref:raise ClosureError('CANDIDATE_WRITER_MISMATCH')
        if _read(store,ref,budget)!=blob:raise ClosureError('CANDIDATE_WRITER_READBACK_MISMATCH')
    return {'status':'REFERENCE_CANDIDATE_ASSEMBLED','candidate_ref':content_ref(raw),
        'candidate_raw':raw,'inventory':logical,'required_content_count':count,'bytes_checked':budget.used,
        'production_release_verified':False,'scientific_acceptance':'NOT_GRANTED','execution_authorized':False}


def verify_reference_candidate(store,candidate_ref,descriptor_raw,index_directory,*,metadata,
        result_request_ref,auxiliary_request_ref,recovery_inputs,max_total_bytes=67108864,capture_ref=None,contract_registry=False):
    budget=_Budget(max_total_bytes);raw=_read(store,candidate_ref,budget,65536);root=validate_record_bytes(raw,max_bytes=65536)
    label=_profile_label(capture_ref,contract_registry)
    if root['schema_id']!='IG_STORAGE_RELEASEROOT_V1' or root['purpose']!=PURPOSE or root['label']!=label:
        raise ClosureError('CANDIDATE_PROFILE_REQUIRED')
    collection=root['collections']['reference_records']
    expected,blobs,recovery,logical,count,capture_report,registry_report=_prepare(store,collection,metadata,result_request_ref,
        auxiliary_request_ref,recovery_inputs,budget,capture_ref=capture_ref,contract_registry=contract_registry)
    if expected!=raw:raise ClosureError('CANDIDATE_ROOT_OR_INVENTORY_MISMATCH')
    # Exact deterministic bytes verify every inventory shard and each empty
    # collection page; extra/missing inventory rows cannot be hidden by rehashing.
    for blob in blobs:
        if _read(store,content_ref(blob),budget)!=blob:raise ClosureError('CANDIDATE_STRUCTURAL_BYTES_MISMATCH')
    index=verify_index_descriptor(store,descriptor_raw,index_directory,collection=collection,
        release_root=candidate_ref,purpose=PURPOSE,max_total_bytes=budget.remaining())
    budget.charge(index['bytes_checked'])
    if index['inventory']!=logical:raise ClosureError('CANDIDATE_INDEX_INVENTORY_MISMATCH')
    return {'status':'REFERENCE_CANDIDATE_VERIFIED','scope':
        'REFERENCE_ONLY_FIXTURE_ROOT_RECOVERY_INDEX_CAPTURE_AND_REGISTRY' if contract_registry else
        'REFERENCE_ONLY_FIXTURE_ROOT_RECOVERY_INDEX_AND_CAPTURE_METADATA' if capture_ref is not None
        else 'REFERENCE_ONLY_FIXTURE_ROOT_RECOVERY_AND_INDEX',
        'candidate_ref':candidate_ref,'recipe_ref':root['recipe_ref'],'collection':collection,'inventory':logical,
        'required_content_count':count,'recovery_verification':recovery,'index_verification':index,
        'bytes_checked':budget.used,'byte_budget':max_total_bytes,
        'capture_metadata_verification':capture_report,
        'contract_registry_verification':registry_report,
        'frozen_contract_and_reference_registry_verified':registry_report is not None,
        'source_and_runtime_declarations_verified':capture_report is not None,
        'production_release_verified':False,'metadata_semantics_verified':False,'execution_authorized':False,
        'scientific_acceptance':'NOT_GRANTED','publication_seal_verified':False,'recovery_performed':False}
