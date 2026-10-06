"""Verbatim bounded JSON and content readers extracted from the pinned engine.
Only the ContentRef error name is adapted; no scientific engine dependency.
"""
import json,math,hashlib,os,stat
from pathlib import Path
from typing import Any
DEFAULT_MAX_BYTES=2*1024*1024
class StorageSchemaError(ValueError):
    pass

def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StorageSchemaError('DUPLICATE_JSON_KEY')
        result[key] = value
    return result

def strict_loads(raw: bytes, *, max_bytes: int = DEFAULT_MAX_BYTES,
                 max_depth: int = 64, max_nodes: int = 100000) -> Any:
    if type(raw) is not bytes or type(max_bytes) is not int or max_bytes < 1:
        raise StorageSchemaError('INVALID_JSON_INPUT_OR_BUDGET')
    if len(raw) > max_bytes:
        raise StorageSchemaError('RECORD_BUDGET_EXCEEDED')
    if raw.startswith(b'\xef\xbb\xbf'):
        raise StorageSchemaError('JSON_BOM_FORBIDDEN')
    try:
        text = raw.decode('utf-8', errors='strict')
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=lambda x: (_ for _ in ()).throw(
                               StorageSchemaError('NONFINITE_JSON')))
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise StorageSchemaError('INVALID_JSON:' + str(exc)[:120]) from exc
    stack = [(value, 0)]; seen = 0
    while stack:
        node, depth = stack.pop(); seen += 1
        if depth > max_depth or seen > max_nodes:
            raise StorageSchemaError('JSON_STRUCTURE_BUDGET_EXCEEDED')
        if isinstance(node, str):
            try: node.encode('utf-8', errors='strict')
            except UnicodeError as exc: raise StorageSchemaError('LONE_SURROGATE') from exc
        elif type(node) is float and not math.isfinite(node):
            raise StorageSchemaError('NONFINITE_JSON')
        elif isinstance(node, dict):
            stack.extend((v, depth + 1) for pair in node.items() for v in pair)
        elif isinstance(node, list):
            stack.extend((v, depth + 1) for v in node)
    return value

def canonical_bytes(value: Any) -> bytes:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise StorageSchemaError('CANONICAL_JSON_INVALID') from exc
    return raw


class ReadIndexError(ValueError):pass
def _ref(ref):
    if not isinstance(ref, dict) or set(ref) != {'sha256', 'size_bytes'}:
        raise ReadIndexError('CONTENT_REF_INVALID')
    import re
    if not isinstance(ref['sha256'], str) or re.fullmatch('[0-9a-f]{64}', ref['sha256']) is None:
        raise ReadIndexError('CONTENT_REF_INVALID')
    if not isinstance(ref['size_bytes'], str) or re.fullmatch('0|[1-9][0-9]*', ref['size_bytes']) is None:
        raise ReadIndexError('CONTENT_REF_INVALID')
    return ref


class CollectionReadError(ValueError):pass
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


