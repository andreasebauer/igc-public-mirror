"""Synthetic page-tree qualification, never publication authority."""
from copy import deepcopy
from pathlib import Path
import hashlib
import pytest
from infinity_grid.storage_schema import canonical_bytes, strict_loads
from infinity_grid.storage_collections import (ContentDirectory, CollectionLimits, CollectionReadError,
    verify_collection, collection_records, build_collection_index)

FIX=Path(__file__).parent/'fixtures/storage_contract_v1/positive'


def put(root,raw):
    ref={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}
    (root/(ref['sha256']+'.blob')).write_bytes(raw)
    return ref


def key(record):return canonical_bytes([record['occurrence_ref']['native_id']]).hex()


def fixture(tmp_path):
    root=tmp_path/'content';root.mkdir()
    rows=[strict_loads((FIX/n).read_bytes()) for n in ['Occurrence_left.json','Occurrence_right.json']]
    profile={'schema_id':'IG_STORAGE_COLLECTION_KEY_PROFILE_V1','format':'CANONICAL_STORAGE_JSONL_V1',
        'key_encoding':'CANONICAL_STRING_ARRAY_UTF8_HEX','order':'ASCII_BYTEWISE','bounds':'INCLUSIVE_DISJOINT',
        'key_fields':[['occurrence_ref','native_id']]}
    pref=put(root,canonical_bytes(profile))
    shard=put(root,b''.join(canonical_bytes(r)+b'\n' for r in rows))
    page={'schema_id':'IG_STORAGE_COLLECTIONPAGE_V1','contract_version':'1.0.0','purpose':'SCHEMA_FIXTURE',
        'collection_kind':'occurrences','record_schema_id':'IG_STORAGE_OCCURRENCE_V1','key_definition_ref':pref,
        'depth':0,'first_key':key(rows[0]),'last_key':key(rows[-1]),'record_count':'2','entries':[
        {'entry_type':'SHARD','content_ref':shard,'first_key':key(rows[0]),'last_key':key(rows[-1]),'record_count':'2'}],'extensions':[]}
    ref={'root':put(root,canonical_bytes(page)),'record_schema_id':page['record_schema_id'],
        'row_count':'2','key_definition_ref':pref,'cardinality_semantics':'OCCURRENCE_LOG'}
    return root,rows,page,ref


def seal(root,page,ref):
    ref['root']=put(root,canonical_bytes(page));return ref


def verify(root,ref,**kwargs):
    return verify_collection(ContentDirectory(root),ref,collection_kind='occurrences',purpose='SCHEMA_FIXTURE',**kwargs)


def test_full_collection_preserves_repeated_object_occurrences(tmp_path):
    root,rows,page,ref=fixture(tmp_path);report=verify(root,ref)
    got=list(collection_records(ContentDirectory(root),ref,collection_kind='occurrences',purpose='SCHEMA_FIXTURE'))
    assert [r.raw for r in got]==[canonical_bytes(r) for r in rows]
    assert rows[0]['object_ref']==rows[1]['object_ref'] and got[0].key!=got[1].key
    assert report['inventory']['record_count']=='2' and report['scientific_acceptance']=='NOT_GRANTED'
    assert report['verification_scope']=='COLLECTION_PAGES_AND_RECORD_BYTES'


def test_empty_leaf_and_nonempty_internal_tree(tmp_path):
    root,rows,page,ref=fixture(tmp_path)
    parent=deepcopy(page);parent['depth']=1
    parent['entries']=[dict(page['entries'][0],entry_type='PAGE',content_ref=ref['root'])]
    assert verify(root,seal(root,parent,ref))['status']=='PASS'
    page.update(entries=[],record_count='0',first_key=None,last_key=None);ref['row_count']='0'
    assert verify(root,seal(root,page,ref))['inventory']['record_count']=='0'


def test_parent_child_depth_mismatch_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);parent=deepcopy(page);parent['depth']=2
    parent['entries']=[dict(page['entries'][0],entry_type='PAGE',content_ref=ref['root'])]
    with pytest.raises(CollectionReadError,match='DEPTH_MISMATCH'):verify(root,seal(root,parent,ref))


def test_parent_child_count_mismatch_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);parent=deepcopy(page);parent.update(depth=1,record_count='3')
    parent['entries']=[dict(page['entries'][0],entry_type='PAGE',content_ref=ref['root'],record_count='3')];ref['row_count']='3'
    with pytest.raises(CollectionReadError,match='SUMMARY_MISMATCH'):verify(root,seal(root,parent,ref))


def test_page_total_mismatch_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);page['record_count']='3';ref['row_count']='3'
    with pytest.raises(CollectionReadError,match='TOTAL_OR_BOUNDS'):verify(root,seal(root,page,ref))


def test_overlap_and_duplicate_shard_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);page['entries']*=2;page['record_count']='4';ref['row_count']='4'
    with pytest.raises(CollectionReadError,match='KEY_RANGE'):verify(root,seal(root,page,ref))


def test_shard_count_claim_is_checked(tmp_path):
    root,rows,page,ref=fixture(tmp_path);page['entries'][0]['record_count']='3';page['record_count']='3';ref['row_count']='3'
    with pytest.raises(CollectionReadError,match='SHARD_SUMMARY'):verify(root,seal(root,page,ref))


def test_wrong_shard_bounds_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);page['last_key']='ff';page['entries'][0]['last_key']='ff'
    with pytest.raises(CollectionReadError,match='SHARD_SUMMARY'):verify(root,seal(root,page,ref))


def test_duplicate_physical_keys_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path)
    page['entries'][0]['content_ref']=put(root,(canonical_bytes(rows[0])+b'\n')*2)
    with pytest.raises(CollectionReadError,match='KEY_ORDER_OR_DUPLICATE'):verify(root,seal(root,page,ref))


def test_missing_and_corrupt_exact_content_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);p=root/(page['entries'][0]['content_ref']['sha256']+'.blob');raw=p.read_bytes();p.unlink()
    with pytest.raises(CollectionReadError,match='MISSING_CONTENT'):verify(root,ref)
    p.write_bytes(b'x'*len(raw))
    with pytest.raises(CollectionReadError,match='DIGEST_MISMATCH'):verify(root,ref)


def test_symlink_content_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);p=root/(ref['root']['sha256']+'.blob');other=tmp_path/'other';p.rename(other);p.symlink_to(other)
    with pytest.raises(CollectionReadError,match='UNSAFE_CONTENT'):verify(root,ref)


def test_resource_limits_refuse_before_large_read(tmp_path,monkeypatch):
    root,rows,page,ref=fixture(tmp_path);store=ContentDirectory(root);calls=[];read=store.read
    def observed(ref,bound):calls.append(ref);return read(ref,bound)
    monkeypatch.setattr(store,'read',observed)
    with pytest.raises(CollectionReadError,match='BYTE_BUDGET'):
        verify_collection(store,ref,collection_kind='occurrences',purpose='SCHEMA_FIXTURE',limits=CollectionLimits(max_total_bytes=1))
    assert calls==[]
    with pytest.raises(CollectionReadError,match='RECORD_COUNT_BUDGET'):verify(root,ref,limits=CollectionLimits(max_records=1))
    with pytest.raises(CollectionReadError,match='RECORD_BYTE_BUDGET'):verify(root,ref,limits=CollectionLimits(max_record_bytes=1))


def test_unsupported_key_profile_is_not_guessed(tmp_path):
    root,rows,page,ref=fixture(tmp_path);bad=put(root,canonical_bytes({'format':'pickle'}))
    page['key_definition_ref']=ref['key_definition_ref']=bad
    with pytest.raises(CollectionReadError,match='UNSUPPORTED_KEY'):verify(root,seal(root,page,ref))


def test_page_and_record_scope_binding_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);page['purpose']='SCIENCE'
    with pytest.raises(CollectionReadError,match='PAGE_BINDING'):verify(root,seal(root,page,ref))
    page['purpose']='SCHEMA_FIXTURE';rows[0]['purpose']='SCIENCE';page['entries'][0]['content_ref']=put(root,b''.join(canonical_bytes(r)+b'\n' for r in rows))
    with pytest.raises(CollectionReadError,match='RECORD_BINDING'):verify(root,seal(root,page,ref))


def test_noncanonical_and_missing_line_terminator_refused(tmp_path):
    root,rows,page,ref=fixture(tmp_path);raw=b''.join(canonical_bytes(r)+b'\n' for r in rows)
    page['entries'][0]['content_ref']=put(root,raw[:-1])
    with pytest.raises(CollectionReadError,match='SHARD_FRAMING'):verify(root,seal(root,page,ref))
    page['entries'][0]['content_ref']=put(root,b' '+raw)
    with pytest.raises(ValueError,match='NONCANONICAL'):verify(root,seal(root,page,ref))


def test_index_build_consumes_verified_canonical_rows(tmp_path):
    root,rows,page,ref=fixture(tmp_path);before={p.name:p.read_bytes() for p in root.iterdir()}
    out=build_collection_index(ContentDirectory(root),ref,tmp_path/'index',release_root=ref['root'],collection_kind='occurrences',purpose='SCHEMA_FIXTURE')
    assert out['index']['binding']['input_inventory']['record_count']=='2'
    assert out['index']['binding']['source_collections']==[ref['root']]
    assert out['scientific_acceptance']=='NOT_GRANTED'
    assert {p.name:p.read_bytes() for p in root.iterdir()}==before


def test_failed_collection_never_publishes_index(tmp_path):
    root,rows,page,ref=fixture(tmp_path);page['entries'][0]['record_count']='3';page['record_count']='3';ref['row_count']='3';seal(root,page,ref)
    with pytest.raises(CollectionReadError):
        build_collection_index(ContentDirectory(root),ref,tmp_path/'index',release_root=ref['root'],collection_kind='occurrences',purpose='SCHEMA_FIXTURE')
    assert not (tmp_path/'index').exists()


def test_changed_bytes_between_passes_refuse_index(tmp_path):
    root,rows,page,ref=fixture(tmp_path);store=ContentDirectory(root);read=store.read;seen=0
    def changed(r,bound):
        nonlocal seen
        if r==ref['root']:
            seen+=1
            if seen==2:(root/(r['sha256']+'.blob')).write_bytes(b'x'*int(r['size_bytes']))
        return read(r,bound)
    store.read=changed
    with pytest.raises(CollectionReadError,match='DIGEST_MISMATCH'):
        build_collection_index(store,ref,tmp_path/'index',release_root=ref['root'],collection_kind='occurrences',purpose='SCHEMA_FIXTURE')
    assert not (tmp_path/'index').exists()
