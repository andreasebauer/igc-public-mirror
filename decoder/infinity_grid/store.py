from __future__ import annotations
import mimetypes, os, shutil, tempfile
from pathlib import Path
from typing import BinaryIO
from .hashing import sha256_file, sha256_bytes
from .canon import write_json_atomic

class StoreError(RuntimeError): pass
class BlobCorrupt(StoreError): pass

class ArtifactStore:
    def __init__(self, root: str|Path):
        self.root=Path(root); (self.root/"sha256").mkdir(parents=True,exist_ok=True); (self.root/"artifacts").mkdir(parents=True,exist_ok=True)
    def blob_path(self, sha: str) -> Path:
        if len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha): raise ValueError("invalid sha256")
        return self.root/"sha256"/sha[:2]/sha
    def verify(self, sha: str, size_bytes: int|None=None) -> dict:
        p=self.blob_path(sha)
        if not p.is_file(): return {"status":"FAIL","reason":"MISSING","sha256":sha,"path":str(p)}
        st=p.stat().st_size
        if size_bytes is not None and st!=size_bytes: return {"status":"FAIL","reason":"SIZE_MISMATCH","sha256":sha,"expected_size":size_bytes,"observed_size":st}
        observed=sha256_file(p)
        return {"status":"PASS" if observed==sha else "FAIL","reason":None if observed==sha else "HASH_MISMATCH","sha256":sha,"observed_sha256":observed,"size_bytes":st,"path":str(p)}
    def _publish_temp(self, tmp: Path, sha: str, size: int) -> Path:
        dst=self.blob_path(sha); dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            v=self.verify(sha,size)
            tmp.unlink(missing_ok=True)
            if v["status"]!="PASS": raise BlobCorrupt(f"existing blob failed verification {v}")
            return dst
        # No-clobber atomic publication.  A racing writer may win; in that case
        # we verify the already-published immutable blob and discard our temp.
        try:
            os.link(tmp,dst)
            tmp.unlink(missing_ok=True)
        except FileExistsError:
            tmp.unlink(missing_ok=True)
            v=self.verify(sha,size)
            if v['status']!='PASS': raise BlobCorrupt(f'racing existing blob failed verification {v}')
            return dst
        try:
            os.chmod(dst,0o444)
            fd=os.open(str(dst.parent),os.O_RDONLY)
            try: os.fsync(fd)
            finally: os.close(fd)
        except OSError: pass
        v=self.verify(sha,size)
        if v["status"]!="PASS": raise BlobCorrupt(f"published blob failed verification {v}")
        return dst
    def put_file(self, path: str|Path, *, logical_role="RUN_RESULT", source_name: str|None=None, created_by_run_id: str|None=None, media_type: str|None=None) -> dict:
        src=Path(path); sha=sha256_file(src); size=src.stat().st_size
        dst=self.blob_path(sha)
        if dst.exists():
            v=self.verify(sha,size)
            if v["status"]!="PASS": raise BlobCorrupt(f"unsafe reuse {v}")
        else:
            dst.parent.mkdir(parents=True,exist_ok=True)
            fd,tmpname=tempfile.mkstemp(prefix=".put-",dir=dst.parent)
            try:
                with os.fdopen(fd,"wb") as out, src.open("rb") as inp:
                    shutil.copyfileobj(inp,out,1024*1024); out.flush(); os.fsync(out.fileno())
                self._publish_temp(Path(tmpname),sha,size)
            except Exception:
                Path(tmpname).unlink(missing_ok=True); raise
        rec={"schema_id":"IG_ARTIFACT_RECORD_V0_17","sha256":sha,"size_bytes":size,"media_type":media_type or mimetypes.guess_type(src.name)[0] or "application/octet-stream","logical_role":logical_role,"source_name":source_name or src.name,"created_by_run_id":created_by_run_id}
        # canonical primary metadata; does not overwrite an existing different primary record
        rp=self.root/"artifacts"/f"{sha}.json"
        if not rp.exists(): write_json_atomic(rp,rec)
        return rec
    def put_bytes(self, data: bytes, *, logical_role="RUN_RESULT", source_name="bytes.bin", created_by_run_id=None, media_type="application/octet-stream") -> dict:
        sha=sha256_bytes(data); size=len(data); dst=self.blob_path(sha); dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            v=self.verify(sha,size)
            if v["status"]!="PASS": raise BlobCorrupt(f"unsafe reuse {v}")
        else:
            fd,tmpname=tempfile.mkstemp(prefix=".put-",dir=dst.parent)
            try:
                with os.fdopen(fd,"wb") as out: out.write(data); out.flush(); os.fsync(out.fileno())
                self._publish_temp(Path(tmpname),sha,size)
            except Exception:
                Path(tmpname).unlink(missing_ok=True); raise
        rec={"schema_id":"IG_ARTIFACT_RECORD_V0_17","sha256":sha,"size_bytes":size,"media_type":media_type,"logical_role":logical_role,"source_name":source_name,"created_by_run_id":created_by_run_id}
        rp=self.root/"artifacts"/f"{sha}.json"
        if not rp.exists(): write_json_atomic(rp,rec)
        return rec
    def materialize(self, sha: str, destination: str|Path, *, expected_size: int|None=None) -> Path:
        v=self.verify(sha,expected_size)
        if v["status"]!="PASS": raise BlobCorrupt(f"cannot materialize unverified blob {v}")
        dst=Path(destination); dst.parent.mkdir(parents=True,exist_ok=True)
        tmp=dst.with_name(dst.name+".tmp")
        shutil.copy2(self.blob_path(sha),tmp); os.replace(tmp,dst)
        if sha256_file(dst)!=sha: raise BlobCorrupt("materialized copy hash mismatch")
        return dst
