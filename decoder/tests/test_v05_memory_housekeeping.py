from __future__ import annotations

import json
from pathlib import Path


def test_completed_phase_file_cache_reclaim_targets_only_large_complete_store(tmp_path, monkeypatch):
    import infinity_grid.v05_stage_runtime as rt
    root=tmp_path
    complete=root/'science_chains'/'intent-a'/'decoder_stage_runtime'/'G6__S7'/'phases'/'P0'
    running=root/'science_chains'/'intent-a'/'decoder_stage_runtime'/'G6__S7'/'phases'/'P1'
    complete.mkdir(parents=True); running.mkdir(parents=True)
    (complete/'RUNTIME_STATUS.json').write_text(json.dumps({'status':'COMPLETE'}))
    (running/'RUNTIME_STATUS.json').write_text(json.dumps({'status':'RUNNING'}))
    (complete/'state_store.sqlite3').write_bytes(b'x'*128)
    (running/'state_store.sqlite3').write_bytes(b'y'*128)
    seen=[]
    def fake(path, *, min_bytes):
        seen.append(Path(path))
        size=Path(path).stat().st_size if Path(path).exists() else 0
        return (size>=min_bytes,size,None)
    monkeypatch.setattr(rt,'_file_cache_dontneed',fake)
    got=rt.reclaim_completed_phase_file_cache(root,min_bytes=64)
    assert got['status']=='PASS'
    assert got['files_advised']==1
    assert got['bytes_advised']==128
    assert complete/'state_store.sqlite3' in seen
    assert running/'state_store.sqlite3' not in seen


def test_cgroup_memory_snapshot_is_resource_only_mapping():
    import infinity_grid.v05_stage_runtime as rt
    got=rt.cgroup_memory_snapshot()
    assert got['schema_id']=='IG_DECODER_CGROUP_MEMORY_SNAPSHOT_V1'
    assert isinstance(got,dict)
