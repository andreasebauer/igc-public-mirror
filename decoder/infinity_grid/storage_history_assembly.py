"""Read-only assembly of exactly the preserved P4-P9 history profile."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile
from .storage_dataset_snapshot import verify_dataset_snapshot


class HistoryAssemblyError(ValueError):
    pass


def assemble_preserved_p9_history(archives, *, max_total_bytes=67108864):
    """Return original dataset bytes and verification; never install or recover."""
    pins=json.loads(Path(__file__).with_name('storage_history_pins.json').read_text())
    if type(archives) is not dict or set(archives)!=set(pins):
        raise HistoryAssemblyError('ARCHIVE_INVENTORY_MISMATCH')
    if type(max_total_bytes) is not int or not 1<=max_total_bytes<=67108864:
        raise HistoryAssemblyError('INVALID_HISTORY_BUDGET')
    if any(type(b) is not bytes for b in archives.values()):
        raise HistoryAssemblyError('ARCHIVE_BYTES_REQUIRED')
    spent=sum(map(len,archives.values()))
    if spent>max_total_bytes:raise HistoryAssemblyError('HISTORY_BYTE_BUDGET')
    members={k:{} for k in ('records','transactions','commits')};manifest=None;origins=[]
    for name in sorted(pins):
        raw=archives[name]
        if hashlib.sha256(raw).hexdigest()!=pins[name]:raise HistoryAssemblyError('ARCHIVE_HASH_MISMATCH')
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist()
            if len(infos)>4096 or len({i.filename for i in infos})!=len(infos):raise HistoryAssemblyError('ARCHIVE_MEMBER_INVENTORY')
            for i in infos:
                path=PurePosixPath(i.filename)
                if path.is_absolute() or '..' in path.parts or '\\' in i.filename:raise HistoryAssemblyError('UNSAFE_ARCHIVE_MEMBER')
                if i.is_dir():continue
                spent+=i.file_size
                if i.file_size>4194304 or spent>max_total_bytes:raise HistoryAssemblyError('HISTORY_BYTE_BUDGET')
                b=z.read(i) # checks CRC; bytes retained unchanged
                parts=path.parts
                if 'replay_reference_data' not in parts:continue
                tail=parts[parts.index('replay_reference_data')+1:]
                if tail==('MANIFEST.json',) and name.startswith('P9_'):manifest=b
                elif len(tail)==2 and tail[0] in members and tail[1].endswith('.json'):
                    group=tail[0]
                    if group=='records' and not name.startswith('P9_'):continue
                    prior=members[group].get(tail[1])
                    if prior is not None and prior!=b:raise HistoryAssemblyError('CONFLICTING_ORIGINAL_HISTORY')
                    members[group][tail[1]]=b
                    origins.append({'archive_sha256':pins[name],'member':i.filename,'sha256':hashlib.sha256(b).hexdigest()})
    if manifest is None:raise HistoryAssemblyError('P9_MANIFEST_MISSING')
    inputs={'manifest_raw':manifest,**{k:list(v.values()) for k,v in members.items()}}
    remaining=max_total_bytes-spent
    if remaining<1:raise HistoryAssemblyError('HISTORY_BYTE_BUDGET')
    result=verify_dataset_snapshot(**inputs,max_total_bytes=remaining)
    return {'status':'PRESERVED_P9_HISTORY_VERIFIED','dataset_inputs':inputs,'dataset_verification':result,
        'origins':origins,'bytes_checked':spent+result['bytes_checked'],
        'execution_authorized':False,'scientific_acceptance':'NOT_GRANTED',
        'production_recovery_verified':False,'recovery_performed':False}
