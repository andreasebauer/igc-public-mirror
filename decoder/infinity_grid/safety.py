from __future__ import annotations
import re
from pathlib import Path, PurePosixPath

SAFE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')

class PathSafetyError(ValueError): pass

def validate_identifier(value: str, *, field: str='identifier') -> str:
    if not isinstance(value,str) or not SAFE_ID_RE.fullmatch(value) or value in {'.','..'}:
        raise PathSafetyError(f'unsafe {field}: {value!r}')
    return value

def validate_relative_path(value: str, *, field: str='relative_path') -> str:
    if not isinstance(value,str) or not value or '\\' in value:
        raise PathSafetyError(f'unsafe {field}: {value!r}')
    p=PurePosixPath(value)
    if p.is_absolute() or any(part in {'','.', '..'} for part in p.parts):
        raise PathSafetyError(f'unsafe {field}: {value!r}')
    norm=p.as_posix()
    if norm!=value:
        raise PathSafetyError(f'non-canonical {field}: {value!r}')
    return norm

def contained_path(root: str|Path, *parts: str, field: str='path') -> Path:
    base=Path(root).resolve()
    cur=base
    for part in parts:
        if '/' in part or '\\' in part:
            validate_relative_path(part,field=field)
            cur=cur/Path(part)
        else:
            validate_identifier(part,field=field)
            cur=cur/part
    resolved=cur.resolve(strict=False)
    try: resolved.relative_to(base)
    except ValueError: raise PathSafetyError(f'{field} escapes root: {resolved}')
    # Existing symlinked ancestors are included by resolve(); the relative check above rejects escape.
    return resolved

def safe_archive_members(names: list[str]) -> tuple[list[str],list[str]]:
    seen=set(); duplicates=[]; unsafe=[]
    for name in names:
        try: validate_relative_path(name,field='archive member')
        except Exception: unsafe.append(name)
        if name in seen: duplicates.append(name)
        seen.add(name)
    return sorted(set(duplicates)), sorted(set(unsafe))
