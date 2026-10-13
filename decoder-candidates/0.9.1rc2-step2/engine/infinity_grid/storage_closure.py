"""Bounded declared-content closure, separately from release/decoded acceptance.

A frozen request pins roots, interpretation formats, and an exact inventory.
Opaque leaves are an explicit declaration, not an inferred property of bytes.
No publication authority or full ReleaseRoot acceptance is conferred here.
"""
from collections import deque
from dataclasses import dataclass, asdict
import hashlib

from .storage_catalog import _ref
from .storage_schema import canonical_bytes, strict_loads, validate_record_bytes
from .storage_collections import collection_records, CollectionLimits, _profile
from .storage_dependencies import record_dependencies
from .storage_legacy import legacy_dependencies,reference
from .canon import canonical_sha256 as legacy_canonical_sha256


class ClosureError(ValueError):
    pass


@dataclass(frozen=True)
class ClosureLimits:
    max_content_bytes: int = 4194304
    max_total_bytes: int = 67108864
    max_objects: int = 8192
    max_edges: int = 32768
    max_records: int = 100000
    max_depth: int = 32

    def __post_init__(self):
        for value,ceiling in zip(asdict(self).values(),(4194304,67108864,8192,32768,100000,32)):
            if type(value) is not int or not 1<=value<=ceiling:raise ClosureError('INVALID_CLOSURE_BUDGET')


def verify_declared_closure(store, request_ref, *, limits=None):
    limits=limits or ClosureLimits()
    if not isinstance(limits,ClosureLimits):raise ClosureError('INVALID_CLOSURE_BUDGET')
    fetched=0;lengths={};required={};structural={};edges=[];decoded=[];lineage=[];skipped=[]
    records_seen=0;identities={};native=[];visited=set();queue=deque();record_schemas={};expected_schemas={};opaque=[]

    def identity(ref):
        _ref(ref);sha=ref['sha256']
        if sha in lengths and lengths[sha]!=ref['size_bytes']:raise ClosureError('CONTENT_LENGTH_CONFLICT')
        lengths[sha]=ref['size_bytes']
        if len(lengths)>limits.max_objects:raise ClosureError('OBJECT_BUDGET')
        return sha

    def read(ref,bound):
        nonlocal fetched
        identity(ref);size=int(ref['size_bytes'])
        if size>min(bound,limits.max_content_bytes) or fetched+size>limits.max_total_bytes:raise ClosureError('BYTE_BUDGET')
        fetched+=size
        raw=store.read(ref,min(bound,limits.max_content_bytes))
        # Do not trust a user-supplied provider to attest its own return value.
        if type(raw) is not bytes or len(raw)!=size or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise ClosureError('PROVIDER_CONTENT_MISMATCH')
        return raw

    raw=read(request_ref,1048576);request=strict_loads(raw,max_bytes=1048576)
    version=request.get('schema_id') if isinstance(request,dict) else None
    fields={'schema_id','purpose','roots','interpretations','required_content'}
    if version=='IG_STORAGE_DECLARED_CLOSURE_REQUEST_V2':fields.add('legacy_bindings')
    if (not isinstance(request,dict) or set(request)!=fields
        or version not in {'IG_STORAGE_DECLARED_CLOSURE_REQUEST_V1','IG_STORAGE_DECLARED_CLOSURE_REQUEST_V2'}
        or request['purpose'] not in {'SCIENCE','SCHEMA_FIXTURE'} or canonical_bytes(request)!=raw):
        raise ClosureError('INVALID_CLOSURE_REQUEST')
    purpose=request['purpose'];formats={};inventory={};legacy_map={};legacy_verified={};legacy_expected=[]
    bindings=request.get('legacy_bindings',[])
    if not isinstance(bindings,list) or len(bindings)>limits.max_objects:raise ClosureError('LEGACY_BINDING_BUDGET_OR_TYPE')
    for row in bindings:
        if not isinstance(row,dict) or set(row)!={'content_ref','binding_ref'}:raise ClosureError('INVALID_LEGACY_BINDING_MAP')
        sha=identity(row['content_ref']);identity(row['binding_ref'])
        if sha in legacy_map:raise ClosureError('DUPLICATE_LEGACY_BINDING_MAP')
        legacy_map[sha]=row['binding_ref']
    allowed={'OPAQUE_LEAF_V1','STORAGE_RECORD_V1','JSON_DEPENDENCIES_V1','COLLECTION_KEY_PROFILE_V1','NATIVE_REFERENCE_RECORD_V1'}
    for field in ['roots','interpretations','required_content']:
        if not isinstance(request[field],list):raise ClosureError('INVALID_REQUEST_LIST')
        if len(request[field])>limits.max_objects:raise ClosureError('OBJECT_BUDGET')
    if not request['roots']:raise ClosureError('ROOTS_REQUIRED')
    for row in request['interpretations']:
        if not isinstance(row,dict) or set(row)!={'content_ref','format'} or row['format'] not in allowed:raise ClosureError('UNSUPPORTED_INTERPRETATION')
        sha=identity(row['content_ref'])
        if sha in formats:raise ClosureError('DUPLICATE_INTERPRETATION')
        formats[sha]=row['format']
    for ref in request['required_content']:
        sha=identity(ref)
        if sha in inventory:raise ClosureError('DUPLICATE_INVENTORY_CONTENT')
        inventory[sha]=ref

    def enqueue(ref,role,parent,path,depth,forced=None):
        sha=identity(ref)
        if depth>limits.max_depth:raise ClosureError('DEPTH_BUDGET')
        if len(edges)>=limits.max_edges:raise ClosureError('EDGE_BUDGET')
        edges.append({'parent':parent,'path':path,'ref':ref,'role':role})
        declared=formats.get(sha)
        if forced is not None and declared is not None and declared!=forced:raise ClosureError('INTERPRETATION_ROLE_CONFLICT')
        fmt=forced or declared
        if fmt is None:raise ClosureError('INTERPRETATION_REQUIRED')
        required[sha]=ref
        queue.append((ref,fmt,depth))

    def scan(raw,location,depth):
        nonlocal records_seen
        records_seen+=1
        if records_seen>limits.max_records:raise ClosureError('RECORD_BUDGET')
        record=validate_record_bytes(raw)
        if record['purpose']!=purpose:raise ClosureError('PURPOSE_MISMATCH')
        deps=record_dependencies(raw)
        if deps['unresolved_fields']:raise ClosureError('UNRESOLVED_FIELD')
        decoded.extend({'record':location,**r} for r in deps['decoded_checks'])
        lineage.extend({'record':location,**r} for r in deps['lineage'])
        skipped.extend({'record':location,**r} for r in deps['skipped_optional_extensions'])
        sid=record['schema_id'];record_schemas[location]=sid;kind=sid.removeprefix('IG_STORAGE_').removesuffix('_V1')
        for row in deps['native_references']:
            token=canonical_bytes(row['ref']).decode()
            if row['role']=='IDENTITY':
                k=(kind,token)
                if k in identities and identities[k]!=location:raise ClosureError('AMBIGUOUS_NATIVE_IDENTITY')
                identities[k]=location
            else:
                path=row['path']
                if path=='/endpoint/ref':target_kind=record['endpoint']['kind']
                elif path=='/object_ref':target_kind='OBJECT'
                elif path=='/parent_occurrence_ref':target_kind='OCCURRENCE'
                elif path=='/relation_ref':target_kind='RELATION'
                else:raise ClosureError('UNSUPPORTED_NATIVE_REFERENCE_ROLE')
                native.append((target_kind,token,location,path))
        legacy_expected.extend(r['ref'] for r in deps['reference_records'])
        collection_paths={r['path'] for r in deps['collections']}
        for row in deps['collections']:
            cref=row['ref'];token=canonical_bytes(cref).decode();key=('COLLECTION',token)
            if key in visited:continue
            visited.add(key)
            if depth+1>limits.max_depth:raise ClosureError('DEPTH_BUDGET')
            def observer(ref,role,data):
                sha=identity(ref)
                (structural if role=='COLLECTION_PAGE' else required)[sha]=ref
                if len(edges)>=limits.max_edges:raise ClosureError('EDGE_BUDGET')
                edges.append({'parent':location,'path':row['path'],'ref':ref,'role':role})
            # The root page declares collection_kind; cross-page consistency is
            # checked by the existing collection verifier with the pinned ref.
            root_raw=read(cref['root'],262144);root_page=validate_record_bytes(root_raw,max_bytes=262144)
            if root_page['schema_id']!='IG_STORAGE_COLLECTIONPAGE_V1':raise ClosureError('NOT_COLLECTION_PAGE')
            class Provider:
                def read(self,ref,bound):return read(ref,bound)
            for item in collection_records(Provider(),cref,collection_kind=root_page['collection_kind'],purpose=purpose,
                                           limits=CollectionLimits(),on_read=observer):
                scan(item.raw,cref['root']['sha256']+':'+item.key,depth+1)
        for row in deps['stored_content']:
            path=row['path'];role=row['role'];ref=row['ref']
            if any(path==p+'/root' or path==p+'/key_definition_ref' for p in collection_paths):continue
            forced=None
            expected=None
            if role=='IDENTITY_PROFILE':expected='IG_STORAGE_IDENTITYPROFILE_V1'
            elif sid=='IG_STORAGE_OBJECT_V1' and path=='/canonical_payload_ref':expected='IG_STORAGE_PAYLOAD_V1'
            elif sid=='IG_STORAGE_RELATION_V1' and path=='/type_ref':expected='IG_STORAGE_RELATIONTYPE_V1'
            if expected is not None:
                expected_schemas.setdefault(ref['sha256'],set()).add(expected)
                forced='STORAGE_RECORD_V1'
            elif role=='NATIVE_REFERENCE_RECORD':forced='NATIVE_REFERENCE_RECORD_V1'
            elif role=='COLLECTION_KEY_PROFILE':forced='COLLECTION_KEY_PROFILE_V1'
            elif role in {'COLLECTION_PAGE','COLLECTION_SHARD'}:raise ClosureError('COLLECTION_CONTEXT_REQUIRED')
            enqueue(ref,role,location,path,depth+1,forced)
        return record

    for ref in request['roots']:enqueue(ref,'ROOT',request_ref['sha256'],'/roots',0)
    while queue:
        ref,fmt,depth=queue.popleft();key=(ref['sha256'],fmt)
        if key in visited:continue
        visited.add(key);raw=read(ref,limits.max_content_bytes)
        if fmt=='OPAQUE_LEAF_V1':opaque.append(ref);continue
        if fmt=='STORAGE_RECORD_V1':
            obj=scan(raw,ref['sha256'],depth)
        elif fmt=='COLLECTION_KEY_PROFILE_V1':_profile(raw)
        elif fmt=='NATIVE_REFERENCE_RECORD_V1':
            binding_ref=legacy_map.get(ref['sha256'])
            if binding_ref is None:raise ClosureError('NATIVE_REFERENCE_CLOSURE_ADAPTER_REQUIRED')
            identity(binding_ref);required[binding_ref['sha256']]=binding_ref
            if len(edges)>=limits.max_edges:raise ClosureError('EDGE_BUDGET')
            edges.append({'parent':ref['sha256'],'path':'/legacy_binding','ref':binding_ref,'role':'LEGACY_DEPENDENCY_BINDING'})
            result=legacy_dependencies(raw,read(binding_ref,262144));rr=result['record_ref']
            for check in result['semantic_checks']:
                value=strict_loads(read(check['ref'],limits.max_content_bytes),max_bytes=limits.max_content_bytes)
                if legacy_canonical_sha256(value)!=check['sha256']:
                    raise ClosureError('LEGACY_RESULT_IDENTITY_MISMATCH')
            witness=result.get('oracle_witness_check')
            if witness is not None:
                from .storage_oracle_witnesses import verify_oracle_negative_witnesses
                oracle_raw=read(witness['oracle_ref'],limits.max_content_bytes)
                witness_files={row['file']:read(row['content_ref'],limits.max_content_bytes) for row in witness['files']}
                verify_oracle_negative_witnesses(raw,oracle_raw,witness_files)
            prior=legacy_verified.get(rr['record_id'])
            if prior is not None and prior!=rr:raise ClosureError('AMBIGUOUS_LEGACY_IDENTITY')
            legacy_verified[rr['record_id']]=rr
            for link in result['record_links']:
                legacy_expected.append(link)
                enqueue(link['content_ref'],'LEGACY_RECORD_LINK',ref['sha256'],link['record_id'],depth+1,'NATIVE_REFERENCE_RECORD_V1')
            for row in result['stored_content']:
                enqueue(row['ref'],row['role'],ref['sha256'],row['path'],depth+1)
        elif fmt=='JSON_DEPENDENCIES_V1':
            obj=strict_loads(raw,max_bytes=limits.max_content_bytes)
            if (not isinstance(obj,dict) or set(obj)!={'schema_id','dependencies'}
                or obj['schema_id']!='IG_STORAGE_EXTERNAL_DEPENDENCIES_V1'
                or canonical_bytes(obj)!=raw or not isinstance(obj['dependencies'],list)):
                raise ClosureError('INVALID_EXTERNAL_DEPENDENCIES')
            for i,child in enumerate(obj['dependencies']):enqueue(child,'EXPLICIT_EXTERNAL_DEPENDENCY',ref['sha256'],'/dependencies/'+str(i),depth+1)
    for rr in legacy_expected:
        reference(rr)
        if legacy_verified.get(rr['record_id'])!=rr:raise ClosureError('LEGACY_REFERENCE_BINDING_MISMATCH')
    for sha,expected in expected_schemas.items():
        if expected!={record_schemas.get(sha)}:raise ClosureError('TARGET_SCHEMA_MISMATCH')
    for kind,token,location,path in native:
        if (kind,token) not in identities:raise ClosureError('DANGLING_NATIVE_REFERENCE')
    if inventory!=required:
        raise ClosureError('REQUIRED_INVENTORY_MISMATCH:missing='+str(len(set(required)-set(inventory)))+',extra='+str(len(set(inventory)-set(required))))
    return {'status':'STRUCTURAL_PASS','verification_scope':'DECLARED_REQUIRED_CONTENT_CLOSURE',
            'scientific_acceptance':'NOT_GRANTED','request_ref':request_ref,'resource_limits':asdict(limits),
            'required_content':[required[k] for k in sorted(required)],'structural_content':[structural[k] for k in sorted(structural)],
            'dependency_edges':edges,'native_bindings':len(identities),'resolved_native_references':len(native),
            'decoded_checks_pending':decoded,'lineage_not_traversed':lineage,'skipped_optional_extensions':skipped,
            'legacy_records_verified':list(legacy_verified.values()),'opaque_leaf_declarations':opaque,'records_checked':records_seen,'bytes_read':fetched}
