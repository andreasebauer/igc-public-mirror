"""Bounded canonical page/shard verification; never release acceptance.

JSONL profile V1 consists of canonical storage records, each followed by LF.
Keys encode the canonical JSON array of selected nonempty string fields as
lowercase UTF-8 hex; order is bytewise ASCII, with inclusive disjoint bounds.
No external code, codecs, locale collation or payload decoding is inferred.
"""
from dataclasses import dataclass, asdict
import hashlib
import os
from pathlib import Path
import stat

from .storage_schema import canonical_bytes, strict_loads, validate_record_bytes, _compiled
from .storage_catalog import IndexRecord, inventory, build_index, _ref


class CollectionReadError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionLimits:
    max_page_bytes: int = 262144
    max_shard_bytes: int = 4194304
    max_record_bytes: int = 65536
    max_total_bytes: int = 67108864
    max_pages: int = 4096
    max_shards: int = 4096
    max_records: int = 100000
    max_entries: int = 1024
    max_depth: int = 32

    def __post_init__(self):
        ceilings = (262144,4194304,65536,67108864,4096,4096,100000,1024,32)
        for value, ceiling in zip(asdict(self).values(), ceilings):
            if type(value) is not int or not 1 <= value <= ceiling:
                raise CollectionReadError('INVALID_COLLECTION_BUDGET')


class ContentDirectory:
    """Read only digest-named .blob files; exact raw hash/length before decode."""
    def __init__(self, directory):
        self.directory = Path(directory)
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise CollectionReadError('UNSAFE_CONTENT_DIRECTORY')

    def read(self, ref, max_bytes):
        _ref(ref)
        size = int(ref['size_bytes'])
        if size > max_bytes: raise CollectionReadError('CONTENT_BYTE_BUDGET')
        path = self.directory / (ref['sha256']+'.blob')
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError as exc:
            raise CollectionReadError('MISSING_CONTENT') from exc
        except OSError as exc:
            raise CollectionReadError('UNSAFE_CONTENT') from exc
        with os.fdopen(fd,'rb') as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode): raise CollectionReadError('UNSAFE_CONTENT')
            if st.st_size != size: raise CollectionReadError('CONTENT_LENGTH_MISMATCH')
            raw = f.read(size+1)
        if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise CollectionReadError('CONTENT_DIGEST_MISMATCH')
        return raw


def _collection_ref(ref):
    validator = _compiled()
    schema = {'$ref':'#/$defs/CollectionRef','$defs':validator.schema['$defs']}
    if not validator.evolve(schema=schema).is_valid(ref):
        raise CollectionReadError('INVALID_COLLECTION_REF')


def _profile(raw):
    p = strict_loads(raw,max_bytes=16384)
    fields = {'schema_id','format','key_encoding','order','bounds','key_fields'}
    if (not isinstance(p,dict) or set(p)!=fields or canonical_bytes(p)!=raw
        or p['schema_id']!='IG_STORAGE_COLLECTION_KEY_PROFILE_V1'
        or p['format'] not in {'CANONICAL_STORAGE_JSONL_V1','REFERENCE_RECORD_REFS_JSONL_V1'}
        or p['key_encoding']!='CANONICAL_STRING_ARRAY_UTF8_HEX'
        or p['order']!='ASCII_BYTEWISE' or p['bounds']!='INCLUSIVE_DISJOINT'):
        raise CollectionReadError('UNSUPPORTED_KEY_OR_SHARD_PROFILE')
    paths = p['key_fields']
    if not isinstance(paths,list) or not 1 <= len(paths) <= 8:
        raise CollectionReadError('INVALID_KEY_FIELDS')
    for path in paths:
        if not isinstance(path,list) or not 1 <= len(path) <= 8 or any(type(x) is not str or not x for x in path):
            raise CollectionReadError('INVALID_KEY_FIELDS')
    if len({tuple(p) for p in paths})!=len(paths):raise CollectionReadError('DUPLICATE_KEY_FIELDS')
    if p['format']=='REFERENCE_RECORD_REFS_JSONL_V1' and paths!=[['record_id']]:
        raise CollectionReadError('NATIVE_KEY_FIELDS_REQUIRED')
    return p


def _record_key(record, profile):
    values=[]
    for path in profile['key_fields']:
        value=record
        for field in path:
            if not isinstance(value,dict) or field not in value:raise CollectionReadError('KEY_FIELD_MISSING')
            value=value[field]
        if type(value) is not str or not value:raise CollectionReadError('KEY_FIELD_NOT_STRING')
        values.append(value)
    key=canonical_bytes(values).hex()
    if len(key)>4096:raise CollectionReadError('KEY_BYTE_BUDGET')
    return key


def collection_records(store, collection, *, collection_kind, purpose, limits=None, on_read=None):
    """Provisional stream: only full exhaustion verifies the collection.

    Consumers must not commit partial output. verify_collection and the explicit
    build_collection_index writer below enforce full exhaustion before success.
    Record dependencies/payloads are deliberately not followed by this API.
    """
    limits=limits or CollectionLimits()
    if not isinstance(limits,CollectionLimits):raise CollectionReadError('INVALID_COLLECTION_BUDGET')
    _collection_ref(collection)
    if type(collection_kind) is not str or not collection_kind or purpose not in {'SCIENCE','SCHEMA_FIXTURE'}:
        raise CollectionReadError('INVALID_COLLECTION_SCOPE')
    used=pages=shards=records=0
    def read(ref,bound,role):
        nonlocal used
        _ref(ref);size=int(ref['size_bytes'])
        if size>bound or used+size>limits.max_total_bytes:raise CollectionReadError('CONTENT_BYTE_BUDGET')
        used+=size
        raw=store.read(ref,bound)
        if type(raw) is not bytes or len(raw)!=size or hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise CollectionReadError('PROVIDER_CONTENT_MISMATCH')
        if on_read is not None:on_read(ref,role,raw)
        return raw
    profile=_profile(read(collection['key_definition_ref'],16384,'COLLECTION_KEY_PROFILE'))
    sid=collection['record_schema_id']
    supported={v.get('properties',{}).get('schema_id',{}).get('const') for v in _compiled().schema['$defs'].values()}
    native=profile['format']=='REFERENCE_RECORD_REFS_JSONL_V1'
    if native:
        if sid!='IG_REPLAY_REFERENCE_RECORD_V1' or collection_kind!='reference_records' or collection['cardinality_semantics']!='SET':
            raise CollectionReadError('NATIVE_COLLECTION_BINDING_MISMATCH')
    elif sid not in supported:raise CollectionReadError('UNSUPPORTED_RECORD_SCHEMA')

    def walk(ref, expected_depth=None, expected=None, root=False):
        nonlocal pages,shards,records
        pages+=1
        if pages>limits.max_pages:raise CollectionReadError('PAGE_BUDGET')
        page=validate_record_bytes(read(ref,limits.max_page_bytes,'COLLECTION_PAGE'),max_bytes=limits.max_page_bytes)
        if page['schema_id']!='IG_STORAGE_COLLECTIONPAGE_V1':raise CollectionReadError('NOT_COLLECTION_PAGE')
        if (page['purpose']!=purpose or page['collection_kind']!=collection_kind or page['record_schema_id']!=sid
            or page['key_definition_ref']!=collection['key_definition_ref']):raise CollectionReadError('PAGE_BINDING_MISMATCH')
        depth=page['depth'];entries=page['entries'];count=int(page['record_count'])
        if depth>limits.max_depth:raise CollectionReadError('DEPTH_BUDGET')
        if expected_depth is not None and depth!=expected_depth:raise CollectionReadError('PAGE_DEPTH_MISMATCH')
        if len(entries)>limits.max_entries:raise CollectionReadError('ENTRY_BUDGET')
        summary=(page['first_key'],page['last_key'],count)
        if expected is not None and summary!=expected:raise CollectionReadError('PAGE_SUMMARY_MISMATCH')
        if root and count!=int(collection['row_count']):raise CollectionReadError('ROOT_COUNT_MISMATCH')
        if count>limits.max_records:raise CollectionReadError('RECORD_COUNT_BUDGET')
        if not entries:
            if not root or depth!=0 or summary!=(None,None,0):raise CollectionReadError('INVALID_EMPTY_PAGE')
            return
        last=None;total=0
        for entry in entries:
            first,end,n=entry['first_key'],entry['last_key'],int(entry['record_count'])
            if (n<1 or not isinstance(first,str) or not isinstance(end,str) or first>end
                or len(first)>4096 or len(end)>4096 or (last is not None and first<=last)):
                raise CollectionReadError('KEY_RANGE_ORDER_OR_COUNT')
            if entry['entry_type']!=('PAGE' if depth else 'SHARD'):raise CollectionReadError('ENTRY_TYPE_DEPTH_MISMATCH')
            total+=n;last=end
        if total!=count or page['first_key']!=entries[0]['first_key'] or page['last_key']!=entries[-1]['last_key']:
            raise CollectionReadError('PAGE_TOTAL_OR_BOUNDS_MISMATCH')
        for entry in entries:
            expected=(entry['first_key'],entry['last_key'],int(entry['record_count']))
            if depth:
                yield from walk(entry['content_ref'],depth-1,expected)
                continue
            shards+=1
            if shards>limits.max_shards:raise CollectionReadError('SHARD_BUDGET')
            raw=read(entry['content_ref'],limits.max_shard_bytes,'COLLECTION_SHARD')
            if not raw or not raw.endswith(b'\n'):raise CollectionReadError('SHARD_FRAMING')
            prior=None;first=None;n=0;pos=0
            while pos<len(raw):
                end=raw.find(b'\n',pos)
                if end<0 or end-pos>limits.max_record_bytes:raise CollectionReadError('RECORD_BYTE_BUDGET')
                line=raw[pos:end];pos=end+1
                if native:
                    from .storage_legacy import reference
                    from .storage_native_records import native_record
                    entry_ref=strict_loads(line,max_bytes=limits.max_record_bytes)
                    if canonical_bytes(entry_ref)!=line:raise CollectionReadError('NONCANONICAL_NATIVE_ENTRY')
                    reference(entry_ref)
                    native_raw=read(entry_ref['content_ref'],limits.max_record_bytes,'NATIVE_REFERENCE_RECORD')
                    record,key=native_record(native_raw,entry=entry_ref,max_bytes=limits.max_record_bytes)
                else:
                    record=validate_record_bytes(line,max_bytes=limits.max_record_bytes)
                    if record['schema_id']!=sid or record['purpose']!=purpose:raise CollectionReadError('RECORD_BINDING_MISMATCH')
                    key=_record_key(record,profile)
                if prior is not None and key<=prior:raise CollectionReadError('RECORD_KEY_ORDER_OR_DUPLICATE')
                if first is None:first=key
                prior=key;n+=1;records+=1
                if records>limits.max_records:raise CollectionReadError('RECORD_COUNT_BUDGET')
                yield IndexRecord(key,native_raw,'LEGACY_REFERENCE_RECORD_V1') if native else IndexRecord(key,line)
            if (first,prior,n)!=expected:raise CollectionReadError('SHARD_SUMMARY_MISMATCH')
    yield from walk(collection['root'],root=True)
    if records!=int(collection['row_count']):raise CollectionReadError('COLLECTION_COUNT_MISMATCH')


def verify_collection(store, collection, *, collection_kind, purpose, limits=None):
    limits=limits or CollectionLimits()
    result=inventory(collection_records(store,collection,collection_kind=collection_kind,purpose=purpose,limits=limits))
    return {'status':'PASS','verification_scope':'COLLECTION_PAGES_AND_RECORD_BYTES',
            'scientific_acceptance':'NOT_GRANTED','collection':collection,'purpose':purpose,
            'collection_kind':collection_kind,'resource_limits':asdict(limits),'inventory':result}


def build_collection_index(store, collection, destination, *, release_root, collection_kind, purpose, limits=None):
    """Explicit two-pass writer. Full release closure/acceptance is NOT checked."""
    report=verify_collection(store,collection,collection_kind=collection_kind,purpose=purpose,limits=limits)
    built=build_index(collection_records(store,collection,collection_kind=collection_kind,purpose=purpose,limits=limits),
        destination,release_root=release_root,source_collections=[collection['root']],expected_inventory=report['inventory'])
    return {'collection_verification':report,'index':built,'scientific_acceptance':'NOT_GRANTED'}
