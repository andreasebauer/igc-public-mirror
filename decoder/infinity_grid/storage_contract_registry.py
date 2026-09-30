"""Exact reference-profile contract/registry bindings, not production admission.

The running qualified verifier defines the supported profile. Supplied registry
bytes cannot introduce code, formats, paths or authority. Captured source pins
must match that implementation; no supplied source is imported or executed.
"""
from pathlib import Path
from .storage_schema import canonical_bytes, SCHEMA_SHA256
from .storage_index_descriptor import content_ref, registered_profiles
from .storage_catalog import _ref

CONTRACT_SHA256='ee3cbf814e6deaa62294388ede5a8e0fa562be2c0da94d3883d4e354102d83f1'
CONTRACT_SIZE=28106
SOURCE_PATHS=tuple('infinity_grid/'+p for p in (
    'resources/storage/FROZEN_CONTRACT_V1.txt',
    'resources/storage/IG_STORAGE_CONTRACT_V1.schema.json',
    'resources/storage/COLLECTION_JSONL_PROFILE_V1.txt',
    'resources/storage/NATIVE_REFERENCE_COLLECTION_V1.txt',
    'storage_catalog.py','storage_collections.py','storage_native_records.py',
    'storage_legacy.py','storage_schema.py','replay_reference_data.py',
    'storage_index_descriptor.py','storage_capture_metadata.py','storage_candidate.py',
    'storage_contract_registry.py'))


class ContractRegistryError(ValueError):
    pass


def registered_contract_objects():
    """Return supported registry and dependencies as raw bytes for publication."""
    base=Path(__file__).parent.parent
    objects={p:(base/p).read_bytes() for p in SOURCE_PATHS}
    contract=objects[SOURCE_PATHS[0]]
    if content_ref(contract)!={'sha256':CONTRACT_SHA256,'size_bytes':str(CONTRACT_SIZE)}:
        raise ContractRegistryError('LOCAL_FROZEN_CONTRACT_MISMATCH')
    if content_ref(objects[SOURCE_PATHS[1]])['sha256']!=SCHEMA_SHA256:
        raise ContractRegistryError('LOCAL_STORAGE_SCHEMA_MISMATCH')
    builder,index_schema=registered_profiles()
    objects['native_index_builder_profile']=builder
    objects['native_index_schema_profile']=index_schema
    registry=canonical_bytes({'schema_id':'IG_REFERENCE_CANDIDATE_REGISTRY_V1',
        'purpose':'SCHEMA_FIXTURE','scope':'PINNED_REFERENCE_FORMATS_AND_READER_IMPLEMENTATIONS',
        'contract_version':'1.0.0','contract_ref':content_ref(contract),
        'formats':['CANONICAL_STORAGE_JSONL_V1','REFERENCE_RECORD_REFS_JSONL_V1'],
        'native_record_schema':'IG_REPLAY_REFERENCE_RECORD_V1',
        'objects':{k:content_ref(v) for k,v in objects.items()},
        'production_registry':False,'transitive_code_closure':False,'science_authority':'NOT_GRANTED'})
    return registry,objects


def verify_contract_registry(store,contract_ref,registry_ref,capture_report,*,max_total_bytes=8388608):
    """Read all registered objects and match independently computed capture pins.

    This helper expects the caller's freshly verified capture report. Candidate
    integration computes it itself; arbitrary report dictionaries authenticate
    neither the capture nor its origin when this helper is called separately.
    """
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise ContractRegistryError('CONTRACT_REGISTRY_BUDGET')
    used=0
    def read(ref):
        nonlocal used
        _ref(ref);n=int(ref['size_bytes'])
        if n>4194304 or used+n>max_total_bytes:raise ContractRegistryError('CONTRACT_REGISTRY_BYTE_BUDGET')
        used+=n;raw=store.read(ref,4194304)
        if type(raw) is not bytes or content_ref(raw)!=ref:raise ContractRegistryError('CONTRACT_REGISTRY_PROVIDER_MISMATCH')
        return raw
    if contract_ref!={'sha256':CONTRACT_SHA256,'size_bytes':str(CONTRACT_SIZE)}:
        raise ContractRegistryError('FROZEN_CONTRACT_REFERENCE_MISMATCH')
    expected,objects=registered_contract_objects()
    if read(registry_ref)!=expected:raise ContractRegistryError('UNSUPPORTED_CONTRACT_REGISTRY')
    pins=capture_report.get('selected_source_member_sha256')
    if type(pins) is not dict or set(pins)!=set(SOURCE_PATHS):
        raise ContractRegistryError('REGISTRY_CAPTURE_SOURCE_MEMBERS')
    for path,raw in objects.items():
        ref=content_ref(raw)
        if path in SOURCE_PATHS and pins[path]!=ref['sha256']:
            raise ContractRegistryError('REGISTRY_CAPTURE_SOURCE_MISMATCH')
        if read(ref)!=raw:raise ContractRegistryError('REGISTRY_OBJECT_BYTES_MISMATCH')
    return {'status':'FROZEN_CONTRACT_AND_REFERENCE_REGISTRY_VERIFIED',
        'contract_ref':contract_ref,'registry_ref':registry_ref,'capture_ref':capture_report['capture_ref'],
        'source_member_count':len(SOURCE_PATHS),'dependency_count':len(objects),'bytes_checked':used,
        'production_registry_verified':False,'transitive_code_closure_verified':False,
        'science_authority':'NOT_GRANTED'}
