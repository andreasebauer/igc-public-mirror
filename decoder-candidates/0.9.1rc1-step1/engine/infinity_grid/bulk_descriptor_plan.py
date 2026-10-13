"""Frozen-domain audit and independent-window registration plan, no science execution."""
import io
import gzip
import hashlib
import json
import zipfile
from .structural_encoding import structural_canonical_bytes as encode

RAW_SHA = '9b2824df5eeeced822c12ba582e778bd7fb1007f2ec32efa2c4f90209ebeb75f'
PREFIX_SHA = 'c218306314006329ca9cdcedfec0319a0c6d9f6cd2ff3c247793f89b483877ad'
DOMAIN = 652366
PREFIX = 56832

class DescriptorPlanError(RuntimeError):
    pass


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):h.update(chunk)
    return h.hexdigest()


def checked_archive(path, expected):
    # Keep one descriptor for the verified bytes and ZIP parsing. Reopening
    # a pathname could read a replacement that was never hashed.
    import os
    from contextlib import contextmanager
    @contextmanager
    def opened():
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = os.fstat(stream.fileno())
            if ((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise DescriptorPlanError('FROZEN_ARCHIVE_CHANGED')
            if digest != expected:
                raise DescriptorPlanError('FROZEN_ARCHIVE_MISMATCH')
            stream.seek(0)
            with zipfile.ZipFile(stream) as z:
                if len(z.namelist())!=len(set(z.namelist())):
                    raise DescriptorPlanError('DUPLICATE_ARCHIVE_MEMBER')
                yield z
            final = os.fstat(stream.fileno())
            if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (final.st_size, final.st_mtime_ns, final.st_ctime_ns):
                raise DescriptorPlanError('FROZEN_ARCHIVE_CHANGED')
    return opened()


def audit_prefix(path):
    rows={};files=[];end=0;semantic=hashlib.sha256()
    with checked_archive(path,PREFIX_SHA) as z:
        ledger=json.loads(z.read('data/FULL_LEDGER.json'))
        for i,item in enumerate(ledger):
            name='data/'+('official_rows/' if i<4 else 'full_rows/')+item['file'];raw=z.read(name)
            if item['chunk']!=i or item['start']!=end or item['stop']-end!=item['records'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
                raise DescriptorPlanError('PRESERVED_PREFIX_FILE_MISMATCH')
            batch=[json.loads(line) for line in raw.splitlines()]
            if [x['ordinal'] for x in batch]!=list(range(end,item['stop'])):raise DescriptorPlanError('PRESERVED_PREFIX_COVERAGE')
            for row in batch:
                rows[row['ordinal']]=row['literal_descriptor_sha256'];semantic.update(encode(row));semantic.update(b'\n')
            files.append({'path':name,**item,'size_bytes':len(raw)});end=item['stop']
    if end!=PREFIX:raise DescriptorPlanError('PRESERVED_PREFIX_LENGTH')
    return rows,{'schema_id':'IG_PRESERVED_MAPPING_PREFIX_V1','archive_sha256':PREFIX_SHA,'records':end,'files':files,'rows_canonical_stream_sha256':semantic.hexdigest(),'recomputed_rows':0}


def iter_descriptors(path):
    with checked_archive(path,RAW_SHA) as z:
        ledger=json.loads(z.read('CHUNK_LEDGER.json'));end=0
        for item in ledger:
            compressed=z.read('chunks/'+item['file'])
            if item['start']!=end or hashlib.sha256(compressed).hexdigest()!=item['sha256']:raise DescriptorPlanError('DESCRIPTOR_PART_BINDING')
            digest=hashlib.sha256();count=0
            with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
                for line in stream:
                    digest.update(line);r=json.loads(line)
                    if type(r[0]) is not int or r[0]!=end or len(r[4])!=14:raise DescriptorPlanError('DESCRIPTOR_COVERAGE')
                    yield {'ordinal':r[0],'edge':r[4],'literal_descriptor_sha256':hashlib.sha256(line).hexdigest()}
                    count+=1;end+=1
            if count!=item['records'] or end!=item['stop'] or digest.hexdigest()!=item['raw_stream_sha256']:raise DescriptorPlanError('DESCRIPTOR_PART_STREAM')
        if end!=DOMAIN:raise DescriptorPlanError('DESCRIPTOR_DOMAIN_LENGTH')


def build_plan(raw_path,prefix_path,*,window_rows=16384):
    if type(window_rows) is not int or window_rows<=0:raise DescriptorPlanError('WINDOW_SIZE')
    preserved,prefix=audit_prefix(prefix_path);all_hash=hashlib.sha256();pending_hash=hashlib.sha256();windows=[];current=[];start=PREFIX
    for row in iter_descriptors(raw_path):
        line=encode(row)+b'\n';all_hash.update(line)
        if row['ordinal']<PREFIX:
            if preserved[row['ordinal']]!=row['literal_descriptor_sha256']:raise DescriptorPlanError('PREFIX_LITERAL_DESCRIPTOR_MISMATCH')
            continue
        pending_hash.update(line);current.append(row)
        if len(current)==window_rows:
            windows.append({'start':start,'stop':row['ordinal']+1,'rows':len(current),'descriptor_stream_sha256':hashlib.sha256(b''.join(encode(r)+b'\n' for r in current)).hexdigest()});start=row['ordinal']+1;current=[]
    if current:windows.append({'start':start,'stop':DOMAIN,'rows':len(current),'descriptor_stream_sha256':hashlib.sha256(b''.join(encode(r)+b'\n' for r in current)).hexdigest()})
    if sum(x['rows'] for x in windows)!=DOMAIN-PREFIX or windows[0]['start']!=PREFIX or windows[-1]['stop']!=DOMAIN:raise DescriptorPlanError('PENDING_WINDOW_COVERAGE')
    result={'schema_id':'IG_HET4_ROLLING_MAPPING_PLAN_V1','domain_rows':DOMAIN,'preserved_rows':PREFIX,'missing_rows':DOMAIN-PREFIX,'raw_archive_sha256':RAW_SHA,'all_descriptors_canonical_stream_sha256':all_hash.hexdigest(),'missing_descriptors_canonical_stream_sha256':pending_hash.hexdigest(),'prefix':prefix,'window_rows_tuning':window_rows,'windows':windows,'execution_model':'ONE_NATIVE_CAPTURE_AND_PORTABLE_PROJECT_PER_WINDOW','data_restart':False,'scientific_adoption':False}
    result['plan_sha256']=hashlib.sha256(encode(result)).hexdigest();return result


def window_descriptors(path, window):
    rows=[]
    for row in iter_descriptors(path):
        if window['start']<=row['ordinal']<window['stop']:rows.append(row)
    if len(rows)!=window['rows'] or hashlib.sha256(b''.join(encode(r)+b'\n' for r in rows)).hexdigest()!=window['descriptor_stream_sha256']:
        raise DescriptorPlanError('WINDOW_DESCRIPTOR_BINDING')
    return rows
