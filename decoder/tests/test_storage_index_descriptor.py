import json,sqlite3
from pathlib import Path
import pytest
from test_storage_native_collection import fixture as native_fixture,build
from test_storage_legacy import put
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory,CollectionReadError
from infinity_grid.storage_catalog import file_ref,ReadIndexError
from infinity_grid.storage_index_descriptor import registered_profiles,verify_index_descriptor


def report(root,d,dest):
    m=json.loads((dest/'INDEX.json').read_bytes())
    return put(root,canonical_bytes({'schema_id':'IG_NATIVE_INDEX_VALIDATION_REPORT_V1','status':'PASS',
        'scope':'EXACT_NATIVE_COLLECTION_AND_CLOSED_INDEX','release_root':d['release_root'],
        'source_collections':d['source_collections'],'index_content_ref':d['index_content_ref'],
        'index_manifest_ref':file_ref(dest/'INDEX.json'),'builder_ref':d['builder_ref'],
        'index_schema_ref':d['index_schema_ref'],'input_inventory':m['binding']['input_inventory'],
        'scientific_acceptance':'NOT_GRANTED'}))


def fixture(tmp_path):
    root,e,raws,col=native_fixture(tmp_path);dest=tmp_path/'index';out=build(root,col,dest)
    builder,schema=registered_profiles()
    d={'schema_id':'IG_STORAGE_INDEXDESCRIPTOR_V1','contract_version':'1.0.0','purpose':'SCHEMA_FIXTURE',
        'authority':'REBUILDABLE_PROJECTION','release_root':col['root'],'source_collections':[col['root']],
        'index_content_ref':out['index']['index_ref'],'builder_ref':put(root,builder),
        'index_schema_ref':put(root,schema),'validation_report_ref':put(root,b'{}'),'extensions':[]}
    d['validation_report_ref']=report(root,d,dest)
    return root,col,d,dest


def run(root,col,d,dest,**kw):
    return verify_index_descriptor(ContentDirectory(root),canonical_bytes(d),dest,
        collection=col,release_root=col['root'],purpose='SCHEMA_FIXTURE',**kw)


def reseal(root,d,dest):
    m=json.loads((dest/'INDEX.json').read_bytes());m['index_ref']=file_ref(dest/'index.sqlite')
    (dest/'INDEX.json').chmod(0o644);(dest/'INDEX.json').write_bytes(canonical_bytes(m));d['index_content_ref']=m['index_ref'];d['validation_report_ref']=report(root,d,dest)


def test_native_descriptor_all_133_rows_read_only(tmp_path):
    root,col,d,dest=fixture(tmp_path);before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in dest.iterdir()};out=run(root,col,d,dest)
    assert out['status']=='INDEX_DESCRIPTOR_VERIFIED' and out['inventory']['record_count']=='133'
    assert not out['production_release_verified'] and not out['builder_execution_attested'] and not out['dependency_closure_verified']
    assert out['scientific_acceptance']=='NOT_GRANTED'
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in dest.iterdir()}


def test_descriptor_wrong_release_binding(tmp_path):
    root,col,d,dest=fixture(tmp_path);d['release_root']=put(root,b'other')
    with pytest.raises(ReadIndexError,match='DESCRIPTOR_SCOPE_MISMATCH'):run(root,col,d,dest)


def test_descriptor_wrong_source_binding(tmp_path):
    root,col,d,dest=fixture(tmp_path);d['source_collections']=[put(root,b'other')]
    with pytest.raises(ReadIndexError,match='DESCRIPTOR_SCOPE_MISMATCH'):run(root,col,d,dest)


def test_descriptor_wrong_index_bytes_ref(tmp_path):
    root,col,d,dest=fixture(tmp_path);d['index_content_ref']=put(root,b'other')
    with pytest.raises(ReadIndexError,match='DESCRIPTOR_INDEX_BINDING_MISMATCH'):run(root,col,d,dest)


def test_descriptor_unregistered_builder(tmp_path):
    root,col,d,dest=fixture(tmp_path);d['builder_ref']=put(root,b'{}')
    with pytest.raises(ReadIndexError,match='BUILDER_PROFILE_MISMATCH'):run(root,col,d,dest)


def test_descriptor_unregistered_schema(tmp_path):
    root,col,d,dest=fixture(tmp_path);d['index_schema_ref']=put(root,b'{}')
    with pytest.raises(ReadIndexError,match='INDEX_SCHEMA_PROFILE_MISMATCH'):run(root,col,d,dest)


def test_descriptor_report_cannot_grant_acceptance(tmp_path):
    root,col,d,dest=fixture(tmp_path);p=root/(d['validation_report_ref']['sha256']+'.blob');r=json.loads(p.read_bytes());r['scientific_acceptance']='GRANTED';d['validation_report_ref']=put(root,canonical_bytes(r))
    with pytest.raises(ReadIndexError,match='INDEX_VALIDATION_REPORT_MISMATCH'):run(root,col,d,dest)


def test_descriptor_self_consistent_index_missing_row(tmp_path):
    root,col,d,dest=fixture(tmp_path);db=dest/'index.sqlite';db.chmod(0o644)
    with sqlite3.connect(db) as con:con.execute('DELETE FROM records WHERE record_key=(SELECT MIN(record_key) FROM records)')
    reseal(root,d,dest)
    with pytest.raises(ReadIndexError,match='INDEX_SOURCE_ROW_MISMATCH'):run(root,col,d,dest)


def test_descriptor_extra_sql_schema_refused(tmp_path):
    root,col,d,dest=fixture(tmp_path);db=dest/'index.sqlite';db.chmod(0o644)
    with sqlite3.connect(db) as con:con.execute('CREATE TABLE hidden(value TEXT)')
    reseal(root,d,dest)
    with pytest.raises(ReadIndexError,match='INDEX_DDL_MISMATCH'):run(root,col,d,dest)


def test_descriptor_hidden_projection_refused(tmp_path):
    root,col,d,dest=fixture(tmp_path);db=dest/'index.sqlite';db.chmod(0o644)
    with sqlite3.connect(db) as con:con.execute("INSERT INTO graph_edges VALUES('x','x','x','x','x')")
    reseal(root,d,dest)
    with pytest.raises(ReadIndexError,match='UNEXPECTED_NATIVE_PROJECTION'):run(root,col,d,dest)


def test_descriptor_missing_source_blob_refused(tmp_path):
    root,col,d,dest=fixture(tmp_path);(root/(col['root']['sha256']+'.blob')).unlink()
    with pytest.raises(CollectionReadError,match='MISSING_CONTENT'):run(root,col,d,dest)


def test_descriptor_budget_and_closed_index_required(tmp_path):
    root,col,d,dest=fixture(tmp_path);out=run(root,col,d,dest)
    with pytest.raises((ReadIndexError,CollectionReadError),match='BYTE_BUDGET'):run(root,col,d,dest,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(ReadIndexError,match='INVALID_DESCRIPTOR_BUDGET'):run(root,col,d,dest,max_total_bytes=True)
    (dest/'index.sqlite-wal').write_bytes(b'')
    with pytest.raises(ReadIndexError,match='INDEX_NOT_CLOSED'):run(root,col,d,dest)
