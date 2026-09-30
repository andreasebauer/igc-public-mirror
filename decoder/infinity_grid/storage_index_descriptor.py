"""Read-only native IndexDescriptor binding; no publication or build-history claim."""
import hashlib
from itertools import zip_longest
from pathlib import Path
from .storage_catalog import ReadIndex, ReadIndexError, DDL, INDEX_VERSION, file_ref, inventory, _ref
from .storage_collections import collection_records, CollectionLimits
from .storage_schema import canonical_bytes, strict_loads, validate_record_bytes


def content_ref(raw):
    return {'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}


def registered_profiles():
    """Exact entrypoint implementation pins, not a transitive environment seal."""
    base=Path(__file__).parent
    names=('storage_catalog.py','storage_collections.py','storage_native_records.py',
           'storage_legacy.py','storage_schema.py','replay_reference_data.py')
    builder=canonical_bytes({'schema_id':'IG_NATIVE_INDEX_BUILDER_PROFILE_V1',
        'entrypoint':'infinity_grid.storage_collections.build_collection_index',
        'implementation_files':{n:file_ref(base/n) for n in names}})
    schema=canonical_bytes({'schema_id':'IG_NATIVE_INDEX_SCHEMA_PROFILE_V1',
        'index_version':INDEX_VERSION,'ddl':DDL,'record_profile':'LEGACY_REFERENCE_RECORD_V1'})
    return builder,schema


def verify_index_descriptor(store,descriptor_raw,directory,*,collection,release_root,purpose,max_total_bytes=67108864):
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise ReadIndexError('INVALID_DESCRIPTOR_BUDGET')
    used=0
    def charge(n):
        nonlocal used
        if used+n>max_total_bytes:raise ReadIndexError('DESCRIPTOR_BYTE_BUDGET')
        used+=n
    charge(len(descriptor_raw))
    d=validate_record_bytes(descriptor_raw,max_bytes=65536)
    if d['schema_id']!='IG_STORAGE_INDEXDESCRIPTOR_V1':raise ReadIndexError('INDEX_DESCRIPTOR_REQUIRED')
    _ref(release_root)
    if d['release_root']!=release_root or d['purpose']!=purpose or d['source_collections']!=[collection['root']]:
        raise ReadIndexError('DESCRIPTOR_SCOPE_MISMATCH')
    if collection.get('record_schema_id')!='IG_REPLAY_REFERENCE_RECORD_V1':raise ReadIndexError('NATIVE_DESCRIPTOR_ONLY')
    if d['extensions']:raise ReadIndexError('DESCRIPTOR_EXTENSIONS_UNSUPPORTED')
    def read(ref):
        _ref(ref);size=int(ref['size_bytes'])
        if size>65536:raise ReadIndexError('DESCRIPTOR_METADATA_BUDGET')
        charge(size);raw=store.read(ref,65536)
        if type(raw) is not bytes or content_ref(raw)!=ref:raise ReadIndexError('DESCRIPTOR_PROVIDER_MISMATCH')
        return raw
    builder,schema=registered_profiles()
    if read(d['builder_ref'])!=builder:raise ReadIndexError('BUILDER_PROFILE_MISMATCH')
    if read(d['index_schema_ref'])!=schema:raise ReadIndexError('INDEX_SCHEMA_PROFILE_MISMATCH')
    report_raw=read(d['validation_report_ref']);report=strict_loads(report_raw,max_bytes=65536)
    directory=Path(directory);manifest=directory/'INDEX.json';db=directory/'index.sqlite'
    if directory.is_symlink() or manifest.is_symlink() or db.is_symlink():raise ReadIndexError('UNSAFE_INDEX_LOCATION')
    if manifest.stat().st_size>65536:raise ReadIndexError('DESCRIPTOR_METADATA_BUDGET')
    charge(manifest.stat().st_size);charge(db.stat().st_size)
    manifest_ref=file_ref(manifest)
    with ReadIndex(directory,expected_manifest_ref=manifest_ref,release_root=release_root) as reader:
        m=reader.manifest
        if m['index_ref']!=d['index_content_ref'] or m['binding']['source_collections']!=d['source_collections']:
            raise ReadIndexError('DESCRIPTOR_INDEX_BINDING_MISMATCH')
        normalize=lambda s:' '.join(s.split())
        expected_sql={normalize(s) for s in DDL.split(';') if s.strip().startswith('CREATE ')}
        con=reader._con;ticks=0
        def progress():
            nonlocal ticks
            ticks+=1000;return int(ticks>=5000000)
        con.set_progress_handler(progress,1000)
        try:
            actual_sql={normalize(row[0]) for row in con.execute('SELECT sql FROM sqlite_master WHERE sql IS NOT NULL')}
            if actual_sql!=expected_sql:raise ReadIndexError('INDEX_DDL_MISMATCH')
            for table in ('native_bindings','incidences','graph_edges'):
                if con.execute('SELECT 1 FROM '+table+' LIMIT 1').fetchone() is not None:
                    raise ReadIndexError('UNEXPECTED_NATIVE_PROJECTION')
            if max_total_bytes-used<1:raise ReadIndexError('DESCRIPTOR_BYTE_BUDGET')
            source=collection_records(store,collection,collection_kind='reference_records',purpose=purpose,
                limits=CollectionLimits(max_total_bytes=max_total_bytes-used),on_read=lambda ref,role,raw:charge(len(raw)))
            rows=con.execute('SELECT record_key,schema_id,raw,raw_sha256,raw_size FROM records ORDER BY record_key')
            def matched():
                for item,row in zip_longest(source,rows):
                    if item is None or row is None:raise ReadIndexError('INDEX_SOURCE_ROW_MISMATCH')
                    wanted=(item.key,'IG_REPLAY_REFERENCE_RECORD_V1',item.raw,hashlib.sha256(item.raw).hexdigest(),str(len(item.raw)))
                    if item.profile!='LEGACY_REFERENCE_RECORD_V1' or row!=wanted:raise ReadIndexError('INDEX_SOURCE_ROW_MISMATCH')
                    yield item
            actual=inventory(matched());rows.close();reader._unchanged()
            if m['binding']['input_inventory']!=actual:raise ReadIndexError('INDEX_SOURCE_INVENTORY_MISMATCH')
        finally:con.set_progress_handler(None,0)
    if file_ref(manifest)!=manifest_ref:raise ReadIndexError('INDEX_MANIFEST_CHANGED')
    expected_report={'schema_id':'IG_NATIVE_INDEX_VALIDATION_REPORT_V1','status':'PASS',
        'scope':'EXACT_NATIVE_COLLECTION_AND_CLOSED_INDEX','release_root':release_root,
        'source_collections':[collection['root']],'index_content_ref':d['index_content_ref'],
        'index_manifest_ref':manifest_ref,'builder_ref':d['builder_ref'],'index_schema_ref':d['index_schema_ref'],
        'input_inventory':actual,'scientific_acceptance':'NOT_GRANTED'}
    if report!=expected_report or report_raw!=canonical_bytes(report):raise ReadIndexError('INDEX_VALIDATION_REPORT_MISMATCH')
    return {'status':'INDEX_DESCRIPTOR_VERIFIED','scope':'EXACT_NATIVE_COLLECTION_AND_CLOSED_INDEX',
        'descriptor_ref':content_ref(descriptor_raw),'inventory':actual,'bytes_checked':used,
        'budget_scope':'INPUT_BYTES_COUNTED_ONCE_PLUS_COLLECTION_READS',
        'production_release_verified':False,'builder_execution_attested':False,
        'dependency_closure_verified':False,'scientific_acceptance':'NOT_GRANTED'}
