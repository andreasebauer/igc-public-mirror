from pathlib import Path
import hashlib
import pytest
from infinity_grid.storage_history_assembly import assemble_preserved_p9_history,HistoryAssemblyError


def originals():
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'infinity_grid/resources/replay').glob('*CAPTURED_STATE*.zip'))+[root/'tests/fixtures/dev79_index_reproduction/P9_CAPTURED_STATE_OBJECT_V1.zip']
    return {p.name:p.read_bytes() for p in paths}


def test_real_p9_complete_original_history_verified():
    out=assemble_preserved_p9_history(originals());d=out['dataset_verification']
    assert d['status']=='DATASET_SNAPSHOT_VERIFIED' and len(d['record_ids'])==133
    assert d['manifest_sha256']=='35dd38d46f9c6cdd2a867e8789846872ee8d77ea4d6d6681de017bc4b324fbb3'
    assert all(len(out['dataset_inputs'][k])==133 for k in ('records','transactions','commits'))
    assert not out['execution_authorized'] and not out['production_recovery_verified'] and not out['recovery_performed']
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_original_archives_and_member_bytes_unchanged():
    import zipfile,io
    a=originals();before={k:hashlib.sha256(v).hexdigest() for k,v in a.items()};out=assemble_preserved_p9_history(a)
    assert before=={k:hashlib.sha256(v).hexdigest() for k,v in a.items()}
    preserved=set()
    for raw in a.values():
        with zipfile.ZipFile(io.BytesIO(raw)) as z:preserved.update(z.read(n) for n in z.namelist() if '/replay_reference_data/' in n and n.endswith('.json'))
    assert all(b in preserved for k in ('records','transactions','commits') for b in out['dataset_inputs'][k])


def test_missing_inherited_archive_refused():
    a=originals();a.pop(next(k for k in a if k.startswith('P4_')))
    with pytest.raises(HistoryAssemblyError,match='INVENTORY'):assemble_preserved_p9_history(a)


def test_unrelated_archive_refused():
    a=originals();a['other.zip']=b''
    with pytest.raises(HistoryAssemblyError,match='INVENTORY'):assemble_preserved_p9_history(a)


def test_altered_original_archive_refused():
    a=originals();k=next(iter(a));a[k]+=b'changed'
    with pytest.raises(HistoryAssemblyError,match='HASH_MISMATCH'):assemble_preserved_p9_history(a)


def test_archive_transport_order_is_irrelevant():
    a=originals();x=assemble_preserved_p9_history(dict(reversed(list(a.items()))))
    assert x['dataset_verification']['record_ids'][-1]=='IGRD/O1_O3/FRONTIER/AUTOMATIC_PASS'


def test_history_budget_and_bytes_contract():
    a=originals()
    with pytest.raises(HistoryAssemblyError,match='BYTE_BUDGET'):assemble_preserved_p9_history(a,max_total_bytes=1)
    with pytest.raises(HistoryAssemblyError,match='INVALID_HISTORY_BUDGET'):assemble_preserved_p9_history(a,max_total_bytes=True)
    a[next(iter(a))]=bytearray(b'bad')
    with pytest.raises(HistoryAssemblyError,match='BYTES_REQUIRED'):assemble_preserved_p9_history(a)
