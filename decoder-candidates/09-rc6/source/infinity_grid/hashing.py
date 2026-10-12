from __future__ import annotations
import hashlib, os
from pathlib import Path

def sha256_file(path: str | Path) -> str:
    p=Path(path); h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_identity(path: str | Path) -> dict:
    p=Path(path); return {"sha256":sha256_file(p),"size_bytes":p.stat().st_size}

def tree_manifest(root: str | Path, *, exclude=()) -> dict[str,dict]:
    root=Path(root); ex=set(exclude); out={}
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel=p.relative_to(root).as_posix()
        if rel in ex or any(rel.startswith(x.rstrip("/")+"/") for x in ex): continue
        out[rel]=file_identity(p)
    return out
