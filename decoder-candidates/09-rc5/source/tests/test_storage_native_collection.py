"""Native P9 collection/index qualification; the release binding is a fixture."""
import hashlib,json,zipfile
from pathlib import Path
import pytest
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory,CollectionLimits,collection_records,verify_collection,build_collection_index
from infinity_grid.storage_catalog import ReadIndex,ReadIndexError,IndexRecord,build_index,inventory,file_ref
from test_storage_legacy import put

SID='IG_REPLAY_REFERENCE_RECORD_V1'


def seal(root,entries,profile=None):
    profile=profile or {'schema_id':'IG_STORAGE_COLLECTION_KEY_PROFILE_V1','format':'REFERENCE_RECORD_REFS_JSONL_V1','key_encoding':'CANONICAL_STRING_ARRAY_UTF8_HEX','order':'ASCII_BYTEWISE','bounds':'INCLUSIVE_DISJOINT','key_fields':[['record_id']]}
    pref=put(root,canonical_bytes(profile));blocks=[]
    for start in range(0,len(entries),32):
        group=entries[start:start+32];keys=[canonical_bytes([r['record_id']]).hex() for r in group]
        shard=put(root,b''.join(canonical_bytes(r)+b'\n' for r in group))
        blocks.append({'entry_type':'SHARD','content_ref':shard,'first_key':keys[0],'last_key':keys[-1],'record_count':str(len(group))})
    page={'schema_id':'IG_STORAGE_COLLECTIONPAGE_V1','contract_version':'1.0.0','purpose':'SCHEMA_FIXTURE','collection_kind':'reference_records','record_schema_id':SID,'key_definition_ref':pref,'depth':0,'first_key':blocks[0]['first_key'] if blocks else None,'last_key':blocks[-1]['last_key'] if blocks else None,'record_count':str(len(entries)),'entries':blocks,'extensions':[]}
    return {'root':put(root,canonical_bytes(page)),'record_schema_id':SID,'row_count':str(len(entries)),'key_definition_ref':pref,'cardinality_semantics':'SET'}


def fixture(tmp_path):
    root=tmp_path/'content';root.mkdir();S=Path(__file__).resolve().parents[1]
    archive=S/'tests/fixtures/dev79_index_reproduction/P9_CAPTURED_STATE_OBJECT_V1.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()=='b360397d2593cfee3086ac4b23560c6ab8de2418d6d607d89844463a00766cfc'
    raws={};entries=[]
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            if '/replay_reference_data/records/' not in n or not n.endswith('.json'):continue
            raw=z.read(n);r=json.loads(raw);raws[r['record_id']]=raw;entries.append({'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':put(root,raw)})
    entries.sort(key=lambda r:canonical_bytes([r['record_id']]).hex())
    return root,entries,raws,seal(root,entries)


def stream(root,ref,**kw):return collection_records(ContentDirectory(root),ref,collection_kind='reference_records',purpose='SCHEMA_FIXTURE',**kw)
def verify(root,ref,**kw):return verify_collection(ContentDirectory(root),ref,collection_kind='reference_records',purpose='SCHEMA_FIXTURE',**kw)
def build(root,ref,dest):return build_collection_index(ContentDirectory(root),ref,dest,release_root=ref['root'],collection_kind='reference_records',purpose='SCHEMA_FIXTURE')


def test_all_133_native_records_preserve_original_bytes_and_seals(tmp_path):
    root,entries,raws,ref=fixture(tmp_path);items=list(stream(root,ref));assert len(items)==133
    assert [r.raw for r in items]==[raws[e['record_id']] for e in entries]
    assert {r.profile for r in items}=={'LEGACY_REFERENCE_RECORD_V1'}
    assert verify(root,ref)['inventory']['record_count']=='133'


def test_native_index_pagination_exact_read_only_and_scoped(tmp_path):
    root,entries,raws,ref=fixture(tmp_path);dest=tmp_path/'index';out=build(root,ref,dest);index=out['index'];before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in dest.iterdir()}
    rows=[];cursor=None
    with ReadIndex(dest,expected_manifest_ref=index['manifest_ref'],release_root=ref['root']) as reader:
        while True:
            page=reader.records(schema_id=SID,limit=17,after=cursor);rows.extend(page['rows']);cursor=page['next_cursor']
            assert page['scientific_acceptance']=='NOT_GRANTED'
            if cursor is None:break
        db={key:raw for key,raw in reader._con.execute('SELECT record_key,raw FROM records')}
    assert len(rows)==133 and [row['record']['record_id'] for row in rows]==[e['record_id'] for e in entries]
    for row,e in zip(rows,entries):assert row['content_ref']==e['content_ref'] and row['record']['record_sha256']==e['record_sha256'] and db[row['key']]==raws[e['record_id']]
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in dest.iterdir()}
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_wrong_native_entry_id_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path);e[0]['record_id']='IGRD/0/WRONG'
    with pytest.raises(ValueError,match='BINDING_MISMATCH'):verify(root,seal(root,e))


def test_wrong_native_entry_semantic_seal_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path);e[0]['record_sha256']='0'*64
    with pytest.raises(ValueError,match='BINDING_MISMATCH'):verify(root,seal(root,e))


def test_missing_native_blob_refused_before_index_publish(tmp_path):
    root,e,raws,ref=fixture(tmp_path);(root/(e[-1]['content_ref']['sha256']+'.blob')).unlink();dest=tmp_path/'index'
    with pytest.raises(ValueError,match='MISSING_CONTENT'):build(root,ref,dest)
    assert not dest.exists()


def test_native_blob_raw_digest_mismatch_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path);p=root/(e[0]['content_ref']['sha256']+'.blob');p.write_bytes(b'x'*p.stat().st_size)
    with pytest.raises(ValueError,match='DIGEST_MISMATCH'):verify(root,ref)


def test_provider_cannot_lie_about_native_blob_bytes(tmp_path):
    root,e,raws,ref=fixture(tmp_path);store=ContentDirectory(root);target=e[0]['content_ref']
    class Liar:
        def read(self,r,bound):return b'x'*int(r['size_bytes']) if r==target else store.read(r,bound)
    with pytest.raises(ValueError,match='PROVIDER_CONTENT_MISMATCH'):list(collection_records(Liar(),ref,collection_kind='reference_records',purpose='SCHEMA_FIXTURE'))


def test_noncanonical_reference_entry_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path);p=json.loads((root/(ref['root']['sha256']+'.blob')).read_bytes());b=p['entries'][0];raw=(root/(b['content_ref']['sha256']+'.blob')).read_bytes();b['content_ref']=put(root,b' '+raw);ref['root']=put(root,canonical_bytes(p))
    with pytest.raises(ValueError,match='NONCANONICAL_NATIVE_ENTRY'):verify(root,ref)


def test_duplicate_native_entry_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path)
    with pytest.raises(ValueError,match='ORDER_OR_DUPLICATE'):verify(root,seal(root,[e[0],e[0]]))


def test_reordered_native_entries_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path)
    with pytest.raises(ValueError):verify(root,seal(root,list(reversed(e))))


def test_native_profile_cannot_select_other_key_or_schema(tmp_path):
    root,e,raws,ref=fixture(tmp_path);p=json.loads((root/(ref['key_definition_ref']['sha256']+'.blob')).read_bytes());p['key_fields']=[['record_sha256']]
    with pytest.raises(ValueError,match='NATIVE_KEY_FIELDS_REQUIRED'):verify(root,seal(root,e,p))
    ref['record_schema_id']='IG_STORAGE_OBJECT_V1'
    with pytest.raises(ValueError,match='NATIVE_COLLECTION_BINDING_MISMATCH'):verify(root,ref)


def test_native_blob_reads_count_toward_byte_budget(tmp_path):
    root,e,raws,ref=fixture(tmp_path);reads=[];list(stream(root,ref,on_read=lambda r,role,b:reads.append((role,len(b)))))
    assert sum(1 for role,n in reads if role=='NATIVE_REFERENCE_RECORD')==133
    with pytest.raises(ValueError,match='BYTE_BUDGET'):verify(root,ref,limits=CollectionLimits(max_total_bytes=sum(n for role,n in reads)-1))
    with pytest.raises(ValueError,match='BYTE_BUDGET'):verify(root,ref,limits=CollectionLimits(max_record_bytes=512))


def test_native_index_key_cannot_relabel_record(tmp_path):
    root,e,raws,ref=fixture(tmp_path);rows=[IndexRecord('wrong-key',raws[e[0]['record_id']],'LEGACY_REFERENCE_RECORD_V1')]
    with pytest.raises(ReadIndexError,match='NATIVE_INDEX_KEY_MISMATCH'):build_index(rows,tmp_path/'index',release_root=ref['root'],source_collections=[ref['root']],expected_inventory=inventory(rows))
    assert not (tmp_path/'index').exists()


def test_native_index_stale_root_and_wrong_query_cursor_refused(tmp_path):
    root,e,raws,ref=fixture(tmp_path);dest=tmp_path/'index';out=build(root,ref,dest)['index']
    with pytest.raises(ReadIndexError,match='STALE_INDEX'):ReadIndex(dest,expected_manifest_ref=out['manifest_ref'],release_root=e[0]['content_ref'])
    with ReadIndex(dest,expected_manifest_ref=out['manifest_ref'],release_root=ref['root']) as reader:
        cursor=reader.records(schema_id=SID,limit=1)['next_cursor']
        with pytest.raises(ReadIndexError,match='CURSOR_BINDING_MISMATCH'):reader.records(schema_id='OTHER',after=cursor)
