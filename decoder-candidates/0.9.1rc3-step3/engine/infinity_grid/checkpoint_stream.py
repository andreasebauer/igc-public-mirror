"""Bounded, file-backed checkpoint assembly; no complete state byte dictionary."""
from collections.abc import Mapping
from contextlib import closing, contextmanager
from pathlib import Path
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import zipfile

from . import submission as sub


class FileMapping(Mapping):
    def __init__(self,paths):self.paths=paths
    def __iter__(self):return iter(self.paths)
    def __len__(self):return len(self.paths)
    def __getitem__(self,key):return self.paths[key].read_bytes()


def bounded_tree(root, maximum):
    size=0
    for path in Path(root).rglob('*'):
        if path.is_symlink():raise sub.SubmissionError('CHECKPOINT_SYMLINK',str(path))
        if path.is_file():
            size+=path.stat().st_size
            if maximum is not None and size>maximum:raise sub.SubmissionError('CHECKPOINT_TREE_RAW_LIMIT',str(size))
    return size


def archive_paths(paths,target):
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(paths.items()):
            sub._relative(name)
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o100644<<16
            with z.open(info,'w') as out,path.open('rb') as source:shutil.copyfileobj(source,out,1024*1024)


def put_file(outbox,path):
    """Stream a content-addressed object into the immutable local outbox."""
    from .immutable_publication import sync_directory
    root=Path(outbox)/'objects';root.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.checkpoint-',dir=root);temp=Path(tmp)
    h=hashlib.sha256();size=0
    try:
        with os.fdopen(fd,'wb') as dst,path.open('rb') as src:
            for chunk in iter(lambda:src.read(1024*1024),b''):
                h.update(chunk);size+=len(chunk);dst.write(chunk)
            dst.flush();os.fsync(dst.fileno())
        digest=h.hexdigest();target=root/(digest+'.bin')
        if target.exists():
            if target.is_symlink():raise sub.SubmissionError('OUTBOX_OBJECT_MISMATCH',digest)
            check=hashlib.sha256();n=0
            with target.open('rb') as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b''):check.update(chunk);n+=len(chunk)
            if n!=size or check.hexdigest()!=digest:raise sub.SubmissionError('OUTBOX_OBJECT_MISMATCH',digest)
        else:os.link(temp,target)
        sync_directory(root)
        return {'sha256':digest,'size_bytes':size}
    finally:temp.unlink(missing_ok=True)


@contextmanager
def snapshot_state(workspace,limits):
    from .v05_controller_event_loop import _snapshot_files
    from .preservation import _verify_terminal_state_evidence
    workspace=Path(workspace);selected=[];total=0;shards=[]
    quiescent=any(any((workspace/'runtime/intake'/name).glob('*.json')) for name in ('completed','prepared_completions'))
    legacy=any('evidence_protocol' not in sub._read(p) for name in ('completed','prepared_completions')
               for p in (workspace/'runtime/intake'/name).glob('*.json'))
    for path,name in _snapshot_files(workspace):
        if (name.startswith(('source/','coordination/','runtime/intake/artifacts/','durability/outbox/','durability/base_objects/'))
            or name=='PROJECT_LOCATION.json' or (name.endswith(('-wal','-shm')) and not name.startswith('runtime/sealed/'))):continue
        size=path.stat().st_size
        shard='bulk_shards' in Path(name).parts and path.suffix=='.bin'
        if shard:
            if (limits['max_shard_bytes'] is not None and size>limits['max_shard_bytes'] or limits['max_state_shards'] is not None and len(shards)>=limits['max_state_shards']):
                raise sub.SubmissionError('CHECKPOINT_SHARD_LIMIT',name)
            shards.append((path,name))
        else:
            total+=size
            if (limits['max_state_file_bytes'] is not None and size>limits['max_state_file_bytes'] or limits['max_state_bytes'] is not None and total>limits['max_state_bytes']):
                raise sub.SubmissionError('CHECKPOINT_STATE_RAW_LIMIT',str(total))
        selected.append((path,name,shard))
        if limits['max_state_files'] is not None and len(selected)>limits['max_state_files']:raise sub.SubmissionError('CHECKPOINT_FILE_COUNT_LIMIT')
    # A missing referenced shard must fail before checkpoint publication.
    for path,name,shard in selected:
        if path.name!='BULK_SHARDS.json':continue
        manifest=json.loads(path.read_text())
        if manifest.get('schema_id')!='IG_STAGE_BULK_SHARDS_V1':raise sub.SubmissionError('CHECKPOINT_SHARD_MANIFEST')
        for row in manifest['shards'].values():
            rel=sub._relative(row['path']);target=path.parent/rel
            if rel.as_posix()!='bulk_shards/'+row['sha256']+'.bin' or not target.is_file() or target.is_symlink():
                raise sub.SubmissionError('CHECKPOINT_SHARD_REFERENCE_MISSING',str(target))
            if target.stat().st_size!=row['size_bytes']:raise sub.SubmissionError('CHECKPOINT_SHARD_REFERENCE_SIZE')
    with tempfile.TemporaryDirectory(prefix='decoder-state-') as tmp:
        temporary=Path(tmp);paths={};modes={};shard_rows=[];actual=0
        for number,(path,name,shard) in enumerate(selected):
            before=path.stat();dest=temporary/str(number)
            with path.open('rb') as src:prefix=src.read(16)
            if prefix==b'SQLite format 3\x00' and not (legacy or name.startswith('runtime/sealed/')):
                with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as src,closing(sqlite3.connect(dest)) as dst:
                    page_size=src.execute('PRAGMA page_size').fetchone()[0]
                    def progress(status,remaining,pages):
                        if limits['max_state_file_bytes'] is not None and pages*page_size>limits['max_state_file_bytes']:
                            raise sub.SubmissionError('CHECKPOINT_DATABASE_RAW_LIMIT',name)
                    src.backup(dst,pages=256,progress=progress)
                    if dst.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise sub.SubmissionError('CHECKPOINT_DATABASE_INVALID')
                    dst.commit()
            else:
                with path.open('rb') as src,dest.open('wb') as dst:
                    remaining=before.st_size
                    while remaining:
                        chunk=src.read(min(1024*1024,remaining))
                        if not chunk:raise sub.SubmissionError('CHECKPOINT_FILE_CHANGED',name)
                        dst.write(chunk);remaining-=len(chunk)
            after=path.stat()
            if (before.st_ino,before.st_mode)!=(after.st_ino,after.st_mode):raise sub.SubmissionError('CHECKPOINT_FILE_MODE_CHANGED',name)
            if prefix!=b'SQLite format 3\x00' and (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                if path.suffix!='.log' or after.st_size<before.st_size:raise sub.SubmissionError('CHECKPOINT_FILE_CHANGED',name)
            modes[name]=after.st_mode&0o777
            if shard:
                digest=hashlib.sha256(dest.read_bytes()).hexdigest()
                if path.stem!=digest:raise sub.SubmissionError('CHECKPOINT_SHARD_HASH',name)
                shard_rows.append({'path':name,'sha256':digest,'size_bytes':dest.stat().st_size,'local':dest})
            else:
                actual+=dest.stat().st_size
                if limits['max_state_bytes'] is not None and actual>limits['max_state_bytes']:raise sub.SubmissionError('CHECKPOINT_STATE_RAW_LIMIT',str(actual))
                paths[name]=dest
        files=FileMapping(paths)
        if quiescent:
            all_paths=dict(paths,**{r['path']:r['local'] for r in shard_rows})
            _verify_terminal_state_evidence(FileMapping(all_paths))
        yield files,modes,shard_rows,temporary
