import json
import pytest
from test_storage_legacy import fixture as legacy_fixture
from infinity_grid.canon import canonical_bytes,canonical_sha256
from infinity_grid.replay_reference_data import ReplayReferenceDataStore,empty_manifest
from infinity_grid.storage_dataset_snapshot import verify_dataset_snapshot,DatasetSnapshotError


def fixture(tmp_path,empty=False):
    root=tmp_path/'dataset';store=ReplayReferenceDataStore.initialize(root)
    if not empty:
        _,_,records,_,_=legacy_fixture(tmp_path)
        for r in records:store.put(r)
    return {'manifest_raw':(root/'MANIFEST.json').read_bytes(),
        **{key:[p.read_bytes() for p in (root/key).glob('*.json')] for key in ['records','transactions','commits']}}


def test_dataset_empty_snapshot_verified(tmp_path):
    out=verify_dataset_snapshot(**fixture(tmp_path,True));assert out['record_ids']==[] and out['records_checked']==1


def test_dataset_populated_and_complete_history_verified(tmp_path):
    s=fixture(tmp_path);out=verify_dataset_snapshot(**s)
    assert out['status']=='DATASET_SNAPSHOT_VERIFIED' and len(out['record_ids'])==3
    assert out['records_checked']==10 and out['recovery_performed'] is False
    assert out['scientific_acceptance']=='NOT_GRANTED' and out['dependency_closure_verified'] is False


def test_dataset_transport_list_order_does_not_change_chain(tmp_path):
    s=fixture(tmp_path)
    for key in ['records','transactions','commits']:s[key].reverse()
    assert len(verify_dataset_snapshot(**s)['record_ids'])==3


def test_dataset_raw_and_semantic_manifest_identity_differ(tmp_path):
    out=verify_dataset_snapshot(**fixture(tmp_path));row=out['inventory'][-1]
    assert row['content_ref']['sha256']!=row['semantic_sha256']


def test_dataset_missing_record_refused(tmp_path):
    s=fixture(tmp_path);s['records'].pop()
    with pytest.raises(Exception,match='reference record unavailable'):verify_dataset_snapshot(**s)


def test_dataset_old_manifest_cannot_hide_retained_records(tmp_path):
    s=fixture(tmp_path);s['manifest_raw']=canonical_bytes(empty_manifest())
    with pytest.raises(Exception,match='REFERENCE_INDEX_ROLLBACK'):verify_dataset_snapshot(**s)


def test_dataset_missing_transaction_refused(tmp_path):
    s=fixture(tmp_path);s['transactions'].pop()
    with pytest.raises(FileNotFoundError):verify_dataset_snapshot(**s)


def test_dataset_pending_transaction_is_not_recovered(tmp_path):
    s=fixture(tmp_path);s['commits'].pop();before=(tmp_path/'dataset/MANIFEST.json').read_bytes()
    with pytest.raises(DatasetSnapshotError,match='PENDING_OR_DUPLICATE_COMMIT'):verify_dataset_snapshot(**s)
    assert (tmp_path/'dataset/MANIFEST.json').read_bytes()==before


def test_dataset_populated_without_history_refused(tmp_path):
    s=fixture(tmp_path);s['transactions']=[];s['commits']=[]
    with pytest.raises(DatasetSnapshotError,match='HISTORY_INVENTORY_MISMATCH'):verify_dataset_snapshot(**s)


def test_dataset_resealed_broken_parent_chain_refused(tmp_path):
    s=fixture(tmp_path);tx=json.loads(s['transactions'][0]);old=tx.pop('transaction_sha256');tx['parent_manifest_sha256']='0'*64;tx['transaction_sha256']=canonical_sha256(tx);s['transactions'][0]=canonical_bytes(tx)
    for i,raw in enumerate(s['commits']):
        c=json.loads(raw)
        if c['transaction_sha256']==old:
            c['transaction_sha256']=tx['transaction_sha256'];c.pop('commit_sha256');c['commit_sha256']=canonical_sha256(c);s['commits'][i]=canonical_bytes(c)
    with pytest.raises(DatasetSnapshotError,match='TRANSACTION_CHAIN_MISMATCH'):verify_dataset_snapshot(**s)


def test_dataset_manifest_count_tampering_refused(tmp_path):
    s=fixture(tmp_path);m=json.loads(s['manifest_raw']);m['record_type_counts']['CANONICAL_OBJECT']+=1;m.pop('manifest_sha256');m['manifest_sha256']=canonical_sha256(m);s['manifest_raw']=canonical_bytes(m)
    with pytest.raises(Exception,match='manifest type counts'):verify_dataset_snapshot(**s)


def test_dataset_duplicate_members_refused(tmp_path):
    s=fixture(tmp_path);s['records'].append(s['records'][0])
    with pytest.raises(DatasetSnapshotError,match='DUPLICATE_DATASET_MEMBER'):verify_dataset_snapshot(**s)


def test_dataset_member_identity_cannot_escape_staging(tmp_path):
    s=fixture(tmp_path);r=json.loads(s['records'][0]);r['record_sha256']='../escape';s['records'][0]=canonical_bytes(r)
    with pytest.raises(DatasetSnapshotError,match='INVALID_DATASET_MEMBER_IDENTITY'):verify_dataset_snapshot(**s)


def test_dataset_shared_byte_and_record_bounds(tmp_path):
    s=fixture(tmp_path);out=verify_dataset_snapshot(**s)
    with pytest.raises(DatasetSnapshotError,match='BYTE_BUDGET'):verify_dataset_snapshot(**s,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(DatasetSnapshotError,match='RECORD_BUDGET'):verify_dataset_snapshot(**s,max_records=out['records_checked']-1)
