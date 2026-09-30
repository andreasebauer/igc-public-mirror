"""Native IGRD byte/identity admission for collections and rebuildable indexes."""
import hashlib
from .storage_schema import strict_loads,canonical_bytes
from .replay_reference_data import verify_reference_record


def native_record(raw,*,entry=None,max_bytes=65536):
    record=verify_reference_record(strict_loads(raw,max_bytes=max_bytes))
    if record['schema_id']!='IG_REPLAY_REFERENCE_RECORD_V1':
        raise ValueError('UNSUPPORTED_NATIVE_RECORD_SCHEMA')
    key=canonical_bytes([record['record_id']]).hex()
    if len(key)>4096:raise ValueError('NATIVE_KEY_BYTE_BUDGET')
    if entry is not None:
        from .storage_legacy import reference
        reference(entry)
        wanted={'record_id':record['record_id'],'record_sha256':record['record_sha256'],
            'content_ref':{'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))}}
        if entry!=wanted:raise ValueError('NATIVE_REFERENCE_BINDING_MISMATCH')
    return record,key
