"""Native, synthetic engineering tests. Not scientific acceptance or scale qualification."""
from pathlib import Path
import copy
import hashlib
import json
import sqlite3
import pytest
from infinity_grid.storage_schema import (validate_record_bytes, canonical_bytes,
    StorageSchemaError, strict_loads, SCHEMA_SHA256, _compiled)
from infinity_grid.storage_catalog import (IndexRecord, ReadIndex, ReadIndexError,
    build_index, inventory, file_ref, switch_alias)
from infinity_grid.sqlite_attestation import (observe_runtime, assess_wal_build,
    UPSTREAM_FIXED_SOURCE, require_wal_fix, SQLiteAdmissionError, verify_runtime_binding)

FIXTURES = Path(__file__).parent/'fixtures/storage_contract_v1/positive'
NAMES = sorted(p.name for p in FIXTURES.glob('*.json'))
ROOT = {'sha256': '1'*64, 'size_bytes': '1'}
COLLECTION = {'sha256': '2'*64, 'size_bytes': '2'}

def specimen(name): return json.loads((FIXTURES/name).read_text())
def fixture_records():
    return [IndexRecord(name, canonical_bytes(specimen(name))) for name in NAMES]

def index(tmp_path, extra=()):
    records = sorted(fixture_records()+list(extra), key=lambda x:x.key)
    dest = tmp_path/'snapshot'
    result = build_index(records, dest, release_root=ROOT, source_collections=[COLLECTION], expected_inventory=inventory(records))
    return dest,result

@pytest.mark.parametrize('name', NAMES)
def test_pinned_schema_positive(name):
    raw = canonical_bytes(specimen(name))
    assert validate_record_bytes(raw) == specimen(name)

@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e9999}',
    b'{"a":"\\ud800"}', b'\xef\xbb\xbf{}', b'\xff'])
def test_strict_parser_rejects(raw):
    with pytest.raises(StorageSchemaError): strict_loads(raw)

def test_schema_references_and_conditionals_enforced():
    invalid=specimen('Object_S.json'); invalid['object_ref']['scope_ref']['sha256']='bad'
    with pytest.raises(StorageSchemaError): validate_record_bytes(canonical_bytes(invalid))
    invalid=specimen('Payload_Whole.json'); invalid['mode']='BLOCKS'
    with pytest.raises(StorageSchemaError): validate_record_bytes(canonical_bytes(invalid))
    invalid=specimen('Coverage.json'); invalid['distinct_tested_count']='30'
    with pytest.raises(StorageSchemaError): validate_record_bytes(canonical_bytes(invalid))
    invalid=specimen('Object_S.json'); invalid['schema_id']='UNREGISTERED'
    with pytest.raises(StorageSchemaError): validate_record_bytes(canonical_bytes(invalid))

def test_schema_no_network_resolution():
    from infinity_grid.storage_schema import _deny_resource
    from referencing.exceptions import NoSuchResource
    with pytest.raises(NoSuchResource): _deny_resource('https://example.invalid/missing.json')
    schema=Path(__file__).parents[1]/'infinity_grid/resources/storage/IG_STORAGE_CONTRACT_V1.schema.json'
    assert hashlib.sha256(schema.read_bytes()).hexdigest()==SCHEMA_SHA256

def test_record_budget_and_canonical_language():
    with pytest.raises(StorageSchemaError): strict_loads(b'{}',max_bytes=1)
    with pytest.raises(StorageSchemaError): validate_record_bytes(canonical_bytes(specimen('Coverage.json'))+b'\n')
    with pytest.raises(StorageSchemaError): strict_loads(b'[[[0]]]',max_depth=1)

def test_runtime_attestation_binds_actual_runtime():
    observed=observe_runtime()
    assert observed['sqlite_version']==sqlite3.sqlite_version
    assert observed['sqlite_source_id']
    assert observed['sqlite_compile_options']
    assert verify_runtime_binding(observed)['sqlite_source_id']==observed['sqlite_source_id']
    wrong=copy.deepcopy(observed); wrong['sqlite_source_id']='changed'
    with pytest.raises(SQLiteAdmissionError): verify_runtime_binding(wrong)

def test_wal_gate_refuses_unreviewed_or_unsafe_sources():
    assert assess_wal_build('3.51.3',UPSTREAM_FIXED_SOURCE)['approved_for_wal_fix_gate']
    for version,source in [('3.46.1','unknown'),('3.51.3','changed'),('9.0.0','unknown')]:
        assert not assess_wal_build(version,source)['approved_for_wal_fix_gate']
    observed=observe_runtime()
    if not observed['wal_fix_gate']['approved_for_wal_fix_gate']:
        with pytest.raises(SQLiteAdmissionError): require_wal_fix()
    else: assert require_wal_fix()['sqlite_version']=='3.51.3'

def test_read_only_no_files_or_bytes_change(tmp_path):
    dest,result=index(tmp_path)
    before={p.name:(file_ref(p),p.stat().st_mtime_ns) for p in dest.iterdir()}
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        page=reader.records(schema_id='IG_STORAGE_OBJECT_V1')
        assert len(page['rows'])==2 and page['scientific_acceptance']=='NOT_GRANTED'
        with pytest.raises(sqlite3.DatabaseError): reader._con.execute('DELETE FROM records')
        with pytest.raises(sqlite3.DatabaseError): reader._con.execute("ATTACH DATABASE ':memory:' AS x")
        with pytest.raises(sqlite3.DatabaseError): reader._con.execute('PRAGMA user_version=99')
        with pytest.raises(sqlite3.DatabaseError): reader._con.execute("SELECT load_extension('x')")
    after={p.name:(file_ref(p),p.stat().st_mtime_ns) for p in dest.iterdir()}
    assert before==after

def test_pagination_keeps_occurrences_and_rejects_wrong_query(tmp_path):
    dest,result=index(tmp_path)
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        page=reader.records(schema_id='IG_STORAGE_OCCURRENCE_V1',limit=1)
        assert page['truncated'] and page['next_cursor']
        second=reader.records(schema_id='IG_STORAGE_OCCURRENCE_V1',limit=1,after=page['next_cursor'])
        assert not second['truncated']
        a,b=page['rows'][0]['record'],second['rows'][0]['record']
        assert a['object_ref']==b['object_ref'] and a['occurrence_ref']!=b['occurrence_ref']
        with pytest.raises(ReadIndexError,match='CURSOR_BINDING'):
            reader.records(schema_id='IG_STORAGE_OBJECT_V1',after=page['next_cursor'])
        with pytest.raises(ReadIndexError): reader.records(schema_id='x',after='not a cursor')

def test_joint_incidence_both_lookup_directions(tmp_path):
    dest,result=index(tmp_path)
    relation=specimen('Relation.json')['relation_ref']
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        rows=reader.incidences(relation=relation)['rows']
        assert len(rows)==3
        for row in rows:
            endpoint=row['record']['endpoint']
            found=reader.incidences(endpoint=endpoint)['rows']
            assert row['key'] in [x['key'] for x in found]
        assert len(reader.records(schema_id='IG_STORAGE_RELATION_V1')['rows'])==1

def test_stale_index_and_cursor_release_binding(tmp_path):
    dest,result=index(tmp_path)
    with pytest.raises(ReadIndexError,match='STALE_INDEX'):
        ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=COLLECTION)
    with pytest.raises(ReadIndexError,match='MANIFEST_MISMATCH'):
        ReadIndex(dest,expected_manifest_ref=COLLECTION,release_root=ROOT)

def test_budget_errors_not_silent_truncation(tmp_path):
    dest,result=index(tmp_path)
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        for limit in (0,1025,True):
            with pytest.raises(ReadIndexError): reader.records(schema_id='x',limit=limit)
        with pytest.raises(ReadIndexError,match='ONE_RECORD'):
            reader.records(schema_id='IG_STORAGE_OBJECT_V1',max_bytes=1)

def test_interrupted_build_does_not_replace_good_index(tmp_path):
    dest,result=index(tmp_path); before=file_ref(dest/'index.sqlite')
    with pytest.raises(ReadIndexError,match='DESTINATION_EXISTS'):
        build_index(fixture_records(),dest,release_root=ROOT,source_collections=[COLLECTION],expected_inventory=inventory(fixture_records()))
    bad=inventory(fixture_records()); bad['record_count']='999'
    with pytest.raises(ReadIndexError,match='INVENTORY_MISMATCH'):
        build_index(fixture_records(),tmp_path/'bad',release_root=ROOT,source_collections=[COLLECTION],expected_inventory=bad)
    assert not (tmp_path/'bad').exists() and file_ref(dest/'index.sqlite')==before
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        assert reader.records(schema_id='IG_STORAGE_OBJECT_V1')['rows']

def test_alias_compare_and_swap(tmp_path):
    p=tmp_path/'LATEST.json'; switch_alias(p,expected=None,target=ROOT)
    with pytest.raises(ReadIndexError,match='ALIAS_CONFLICT'): switch_alias(p,expected=None,target=COLLECTION)
    assert json.loads(p.read_bytes())==ROOT
    switch_alias(p,expected=ROOT,target=COLLECTION)
    assert json.loads(p.read_bytes())==COLLECTION

def test_wal_sidecar_not_ignored(tmp_path):
    dest,result=index(tmp_path); (dest/'index.sqlite-wal').write_bytes(b'incomplete')
    with pytest.raises(ReadIndexError,match='INDEX_NOT_CLOSED'):
        ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT)

def test_modified_database_rejected(tmp_path):
    dest,result=index(tmp_path)
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        p=dest/'index.sqlite';p.chmod(0o644)
        with p.open('ab') as f:f.write(b'x')
        with pytest.raises(ReadIndexError,match='INDEX_CHANGED'): reader.records(schema_id='x')

def test_source_identity_collision_not_deduplicated(tmp_path):
    rows=fixture_records(); raw=canonical_bytes(specimen('Occurrence_left.json'))
    rows=sorted(rows+[IndexRecord('Z_duplicate_native.json',raw)],key=lambda x:x.key)
    with pytest.raises(sqlite3.IntegrityError):
        build_index(rows,tmp_path/'dup',release_root=ROOT,source_collections=[COLLECTION],expected_inventory=inventory(rows))

def test_native_schema_dispatch_enforces_new_contract():
    from infinity_grid.schema import validate, ValidationError
    schema=json.loads((Path(__file__).parents[1]/'infinity_grid/resources/storage/IG_STORAGE_CONTRACT_V1.schema.json').read_text())
    assert validate(schema,specimen('Object_S.json'))==[]
    invalid=specimen('Object_S.json'); invalid['object_ref']['profile_ref']['sha256']='bad'
    with pytest.raises(ValidationError):validate(schema,invalid)
    altered=copy.deepcopy(schema);altered.pop('oneOf')
    with pytest.raises(ValidationError):validate(altered,{})

def test_runtime_policy_does_not_authorize_stage(tmp_path):
    from infinity_grid.sqlite_attestation import admit_capture_runtime
    obs=observe_runtime(); spec={'schema_id':'IG_CAPTURE_SQLITE_BINDING_V1','policy':'ENGINEERING_VALIDATION_ONLY','observation':obs}
    raw=canonical_bytes(spec);h=hashlib.sha256(raw).hexdigest()
    p=tmp_path/'runtime/intake/artifacts';p.mkdir(parents=True);(p/(h+'.bin')).write_bytes(raw)
    cap={'environment':{'artifacts':[{'logical_name':'sqlite_runtime_binding','sha256':h}]},'job':{'execution':{'kind':'VALIDATION'}}}
    assert admit_capture_runtime(tmp_path,cap)['sqlite_source_id']==obs['sqlite_source_id']
    cap['job']['execution']['kind']='STAGE'
    with pytest.raises(SQLiteAdmissionError,match='NOT_FOR_SCIENCE'):admit_capture_runtime(tmp_path,cap)
    cap['environment']['artifacts']=[]
    with pytest.raises(SQLiteAdmissionError,match='BINDING_REQUIRED'):admit_capture_runtime(tmp_path,cap)

def test_indexed_legacy_adjacency_without_global_filescan(tmp_path,monkeypatch):
    from infinity_grid.graph_store import build_edge,GraphStore
    records=[]
    for n in range(5000):
        e=build_edge(edge_type='REQUIRES',source_node_id='S'+str(n),target_node_id='T'+str(n),created_utc='2026-09-28T00:00:00+00:00')
        records.append(IndexRecord('edge/%06d'%n,canonical_bytes(e),'LEGACY_GRAPH_EDGE_V1'))
    dest=tmp_path/'graph';result=build_index(records,dest,release_root=ROOT,source_collections=[COLLECTION],expected_inventory=inventory(records))
    def no_scan(*a,**k):raise AssertionError('GLOBAL_GRAPH_SCAN')
    monkeypatch.setattr(GraphStore,'list_edges',no_scan)
    with ReadIndex(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        for direction,node in [('OUT','S4999'),('IN','T4999')]:
            page=reader.graph_neighbours(node,direction=direction,edge_type='REQUIRES',max_vm_steps=1000)
            assert len(page['rows'])==1
            assert page['rows'][0]['record']['edge_id']==json.loads(records[-1].raw)['edge_id']
        # Plan evidence must show an indexed endpoint search, not a table scan.
        plan=reader._con.execute('EXPLAIN QUERY PLAN SELECT record_key FROM graph_edges WHERE source_node_id=? AND edge_type=? AND record_key>? ORDER BY record_key LIMIT ?',('S4999','REQUIRES','',2)).fetchall()
        assert any('SEARCH' in str(row) and 'edge_out_type_key' in str(row) for row in plan)

def test_facade_uses_read_path_without_migration(tmp_path,monkeypatch):
    from infinity_grid.catalog import Catalogue
    from infinity_grid.paths import IGPaths
    dest,result=index(tmp_path)
    def no_write(*a,**k):raise AssertionError('READ_CALLED_WRITER')
    monkeypatch.setattr(Catalogue,'connect',no_write);monkeypatch.setattr(Catalogue,'_open_and_migrate',no_write)
    cat=Catalogue(IGPaths(tmp_path/'nonexistent-workspace'))
    with cat.open_read_index(dest,expected_manifest_ref=result['manifest_ref'],release_root=ROOT) as reader:
        assert len(reader.records(schema_id='IG_STORAGE_OBJECT_V1')['rows'])==2
    assert not (tmp_path/'nonexistent-workspace').exists()
