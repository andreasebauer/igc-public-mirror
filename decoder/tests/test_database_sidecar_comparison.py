"""New registered comparison, not a lowered acceptance threshold for dev135."""
from pathlib import Path
from contextlib import closing
import hashlib
import json
import sqlite3
import struct
import sys
import zipfile
from infinity_grid import preservation as pr
from infinity_grid.canon import canonical_sha256


def sha(raw):return hashlib.sha256(raw).hexdigest()


def inventory(db):
    return {s:{'sha256':sha(Path(str(db)+s).read_bytes()),'size_bytes':Path(str(db)+s).stat().st_size}
            for s in ('','-wal','-shm') if Path(str(db)+s).exists()}


def read(db,table,column,immutable):
    uri=db.as_uri()+'?mode=ro'+('&immutable=1' if immutable else '')
    with closing(sqlite3.connect(uri,uri=True)) as c:
        assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
        rows=[r[0] for r in c.execute('SELECT '+column+' FROM '+table+' ORDER BY '+column)]
    assert len(rows)==len(set(rows))
    return rows


def header_observations(main,wal,shm):
    # Diagnostic raw header fields only; NOT a WAL checksum validator or recovery implementation.
    result={'main_read_write_versions':list(main[18:20]),'main_page_count_u32be':int.from_bytes(main[28:32],'big'),
            'wal_header_u32be':list(struct.unpack('>8I',wal[:32])) if len(wal)>=32 else [],
            'shm_first_two_headers_equal':shm[:48]==shm[48:96],
            'shm_native_byteorder':sys.byteorder,'shm_u32_at_16':int.from_bytes(shm[16:20],sys.byteorder),
            'shm_u32_at_96':int.from_bytes(shm[96:100],sys.byteorder),'shm_u32_at_128':int.from_bytes(shm[128:132],sys.byteorder),
            'wal_checksums_validated':False}
    if len(wal)>=32:
        pagesize=result['wal_header_u32be'][2];assert pagesize>0
        result['physical_complete_frames']=(len(wal)-32)//(24+pagesize)
        result['trailing_bytes']=(len(wal)-32)%(24+pagesize)
        result['physical_commit_marker_frames']=[i+1 for i in range(result['physical_complete_frames'])
            if int.from_bytes(wal[32+i*(24+pagesize)+4:32+i*(24+pagesize)+8],'big')]
    return result


def compare(tmp_path,label,tail,table,column,allowed):
    base=Path(__file__).parent/'fixtures/a12_sidecars';archive=base/'incident.zip';raw=archive.read_bytes()
    assert sha(raw)=='744eb4e91231b40f1d058eee5fcd30a6c8daca3c7a5caa0df4b5526a0afff969'
    with zipfile.ZipFile(archive) as z:files={n:z.read(n) for n in z.namelist()}
    name=next(n for n in files if n.endswith(tail));out=[]
    variants=[('MAIN_IMMUTABLE',('',),'immutable'),('MAIN_NATIVE',('',),'native'),
              ('MAIN_WAL_NATIVE',('','-wal'),'native'),('EXACT_PAIR_NATIVE',('','-wal','-shm'),'native'),
              ('EXACT_PAIR_DIRECT',('','-wal','-shm'),'direct')]
    for variant,suffixes,operation in variants:
        root=tmp_path/variant;root.mkdir();db=root/'copy.sqlite3'
        for suffix in suffixes:Path(str(db)+suffix).write_bytes(files[name+suffix])
        before=inventory(db);assert set(before)==set(suffixes)
        assert all(before[s]['sha256']==sha(files[name+s]) for s in suffixes)
        if operation=='native':
            snapshot=pr._stable_file(db);snap=root/'snapshot.sqlite3';snap.write_bytes(snapshot)
            rows=read(snap,table,column,True);snapshot_sha=sha(snapshot)
        else:
            rows=read(db,table,column,operation=='immutable');snapshot_sha=None
        assert set(rows)<=set(allowed)
        out.append({'variant':variant,'operation':operation,'before':before,'after':inventory(db),
                    'count':len(rows),'ids':rows,'ids_sha256':canonical_sha256(rows),'snapshot_sha256':snapshot_sha})
    assert out[0]['ids']==out[1]['ids'],'Main-only native backup differs from immutable main-only view'
    original=out[0]['ids']
    for row in out:
        row['missing_relative_to_main']=[i for i in original if i not in row['ids']]
        row['additional_relative_to_main']=[i for i in row['ids'] if i not in original]
    assert sha(archive.read_bytes())==sha(raw)
    result={'database':label,'archive_sha256':sha(raw),'original_path':name,'variants':out,
            'raw_headers':header_observations(files[name],files[name+'-wal'],files[name+'-shm']),
            'native_vs_direct_full_pair_equal':out[3]['ids']==out[4]['ids'],
            'wal_only_vs_full_pair_equal':out[2]['ids']==out[3]['ids'],
            'originals_opened_with_sqlite':False,'repair_qualified':False,'historical_cause_established':False}
    print('A12_DATABASE_SIDECAR_COMPARISON='+json.dumps(result,sort_keys=True))


def test_control_database_file_combinations(tmp_path):
    compare(tmp_path,'CONTROL','/CONTROL/partition.sqlite3','task_results','task_id',[f'v{i:03d}' for i in range(96)])


def test_held_write_database_file_combinations(tmp_path):
    compare(tmp_path,'HELD_WRITE','/HELD_WRITE/partition.sqlite3','task_results','task_id',[f'v{i:03d}' for i in range(96)])


def test_project_database_file_combinations(tmp_path):
    compare(tmp_path,'PROJECT','/project_writer.sqlite3','probe','id',[1,2])
