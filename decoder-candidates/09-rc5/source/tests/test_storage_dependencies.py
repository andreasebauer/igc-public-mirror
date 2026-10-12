"""Direct dependency roles and extension refusal; no full closure claim."""
from copy import deepcopy
from pathlib import Path
import pytest
from infinity_grid.storage_schema import canonical_bytes, strict_loads, validate_record_bytes, StorageSchemaError
from infinity_grid.storage_dependencies import record_dependencies
from infinity_grid.storage_collections import ContentDirectory, verify_collection, build_collection_index
from test_storage_collections import fixture,put,seal

FIX=Path(__file__).parent/'fixtures/storage_contract_v1/positive'


def specimen(name):return strict_loads((FIX/name).read_bytes())
def deps(name):return record_dependencies((FIX/name).read_bytes())
def extension(required):
    ref=specimen('Object_S.json')['canonicalization_ref']
    return {'extension_id':'UNREGISTERED_ENGINEERING_EXTENSION','version':'1','required_to_interpret':required,'definition_ref':ref,'payload_ref':ref}


def test_required_extension_refused_for_every_storage_shape():
    for p in FIX.glob('*.json'):
        obj=strict_loads(p.read_bytes());obj['extensions']=[extension(True)]
        with pytest.raises(StorageSchemaError,match='UNSUPPORTED_REQUIRED_EXTENSION'):validate_record_bytes(canonical_bytes(obj))


def test_optional_extension_preserved_and_skipping_reported():
    obj=specimen('Object_S.json');obj['extensions']=[extension(False)];raw=canonical_bytes(obj)
    assert validate_record_bytes(raw)==obj
    out=record_dependencies(raw)
    assert out['skipped_optional_extensions']==[{'path':'/extensions/0','extension_id':'UNREGISTERED_ENGINEERING_EXTENSION','version':'1'}]
    roles={r['role'] for r in out['stored_content']}
    assert {'OPTIONAL_EXTENSION_DEFINITION','OPTIONAL_EXTENSION_PAYLOAD'}<=roles


def test_unknown_required_extension_cannot_enter_collection(tmp_path):
    root,rows,page,ref=fixture(tmp_path);rows[0]['extensions']=[extension(True)]
    page['entries'][0]['content_ref']=put(root,b''.join(canonical_bytes(r)+b'\n' for r in rows));seal(root,page,ref)
    with pytest.raises(StorageSchemaError,match='UNSUPPORTED_REQUIRED_EXTENSION'):
        verify_collection(ContentDirectory(root),ref,collection_kind='occurrences',purpose='SCHEMA_FIXTURE')
    with pytest.raises(StorageSchemaError,match='UNSUPPORTED_REQUIRED_EXTENSION'):
        build_collection_index(ContentDirectory(root),ref,tmp_path/'index',release_root=ref['root'],collection_kind='occurrences',purpose='SCHEMA_FIXTURE')
    assert not (tmp_path/'index').exists()


def test_all_shape_content_references_are_accounted_exactly_once():
    def leaves(obj,path=''):
        if isinstance(obj,dict):
            if set(obj)=={'sha256','size_bytes'}:return [(path,obj)]
            return [row for k,v in obj.items() for row in leaves(v,path+'/'+k.replace('~','~0').replace('/','~1'))]
        if isinstance(obj,list):return [row for i,v in enumerate(obj) for row in leaves(v,path+'/'+str(i))]
        return []
    for p in FIX.glob('*.json'):
        obj=strict_loads(p.read_bytes());out=record_dependencies(p.read_bytes())
        observed=[(r['path'],r['ref']) for group in ['stored_content','decoded_checks','lineage'] for r in out[group]]
        assert sorted(observed,key=lambda x:x[0])==sorted(leaves(obj),key=lambda x:x[0]),p.name
        assert len({r[0] for r in observed})==len(observed)
        assert out['verification_scope']=='DIRECT_SCHEMA_REFERENCES' and out['scientific_acceptance']=='NOT_GRANTED'


def test_decoded_commitment_not_a_stored_dependency():
    for name in ['Payload_Whole.json','Payload_Blocks.json','Block_0.json']:
        out=deps(name);assert [r['path'] for r in out['decoded_checks']]==['/decoded_content']
        assert '/decoded_content' not in [r['path'] for r in out['stored_content']]
    paths={r['path'] for r in deps('Block_0.json')['stored_content']}
    assert {'/stored_content','/codec_ref'}<=paths


def test_release_predecessor_is_lineage_only():
    obj=specimen('ReleaseRoot.json');obj['previous_release_root']=obj['source_ref']
    out=record_dependencies(canonical_bytes(obj))
    assert out['lineage']==[{'path':'/previous_release_root','ref':obj['source_ref']}]
    assert not any(r['path']=='/previous_release_root' for r in out['stored_content'])
    assert len(out['collections'])==8


def test_native_identity_and_reference_roles_stay_distinct():
    out=deps('Occurrence_left.json');refs={r['path']:r for r in out['native_references']}
    assert refs['/occurrence_ref']['role']=='IDENTITY' and refs['/object_ref']['role']=='REFERENCE'
    assert refs['/occurrence_ref']['ref']['native_id']=='P/left'
    paths={r['path'] for r in out['stored_content']}
    assert '/object_ref/profile_ref' in paths and '/occurrence_ref/scope_ref' in paths


def test_legacy_semantic_seal_retained_separately():
    out=deps('PublicationSeal.json');assert out['reference_records']
    for row in out['reference_records']:
        assert row['requires_legacy_seal_check'] is True
        assert 'record_id' in row['ref'] and 'record_sha256' in row['ref']
    assert not any(r['path'].endswith('/record_sha256') for r in out['stored_content'])


def test_unknown_optional_field_stays_unresolved():
    obj=specimen('Object_S.json');obj['structure_ref']={'state':'UNKNOWN','content_ref':None,'explanation':'not yet determined'}
    out=record_dependencies(canonical_bytes(obj));assert out['unresolved_fields']==[{'path':'/structure_ref','explanation':'not yet determined'}]
    assert not any(r['path'].startswith('/structure_ref') for r in out['stored_content'])


def test_page_shards_and_child_pages_have_distinct_roles():
    obj=specimen('CollectionPage.json');out=record_dependencies(canonical_bytes(obj))
    assert any(r['role']=='COLLECTION_SHARD' for r in out['stored_content'])
    obj['depth']=1;obj['entries'][0]['entry_type']='PAGE';out=record_dependencies(canonical_bytes(obj))
    assert any(r['role']=='COLLECTION_PAGE' for r in out['stored_content'])


def test_nested_collections_keep_full_binding():
    obj=specimen('Layer.json');out=record_dependencies(canonical_bytes(obj))
    for row in out['collections']:
        lane=row['path'].split('/')[-1];assert row['ref']==obj['lane_evidence'][lane]
        assert any(r['path']==row['path']+'/root' for r in out['stored_content'])


def test_shared_content_keeps_multiple_dependency_roles():
    obj=specimen('Object_S.json');obj['carrier_schema_ref']=obj['canonicalization_ref']
    out=record_dependencies(canonical_bytes(obj));rows=[r for r in out['stored_content'] if r['ref']==obj['carrier_schema_ref']]
    assert {'/carrier_schema_ref','/canonicalization_ref'}<={r['path'] for r in rows}


def test_unknown_schema_and_required_extension_never_yield_inventory():
    obj=specimen('Object_S.json');obj['schema_id']='UNKNOWN_SCHEMA'
    with pytest.raises(StorageSchemaError):record_dependencies(canonical_bytes(obj))
    obj=specimen('Object_S.json');obj['extensions']=[extension(True)]
    with pytest.raises(StorageSchemaError,match='UNSUPPORTED_REQUIRED_EXTENSION'):record_dependencies(canonical_bytes(obj))
