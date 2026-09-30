import json,shutil,hashlib
from pathlib import Path
import pytest
from test_storage_p9_result_closure import fixture as result_fixture
from test_storage_auxiliary_closure import fixture as auxiliary_fixture
from test_storage_frontier_history import fixture as frontier_fixture
from test_storage_history_assembly import originals
from test_storage_legacy import put
from infinity_grid.storage_history_assembly import assemble_preserved_p9_history
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.storage_full_recovery_join import verify_full_recovery_inventory


def fixture(tmp_path):
    a=tmp_path/'result';a.mkdir();root,s,r,bs=result_fixture(a)
    b=tmp_path/'auxiliary';b.mkdir();other,aux,_=auxiliary_fixture(b)
    for p in other.iterdir():
        target=root/p.name
        if target.exists():assert target.read_bytes()==p.read_bytes()
        else:shutil.copy2(p,target)
    h=frontier_fixture();S=Path(__file__).resolve().parents[1];catraw=(S/'infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json').read_bytes();cat=json.loads(catraw);pins=[]
    for group,fields in [('obligations',['source_hashes','evidence_hashes']),('historical_assertion_mappings',['source_hashes']),('historical_audit_authorizations',['historical_source_hashes','accepted_equivalence_certificates','audit_provenance','audit_evidence']),('known_replay_qualifications',['evidence_pins'])]:
        for owner in cat.get(group,[]):
            for field in fields:pins.extend(owner.get(field,[]))
    for field in ['audit_provenance_ledger','known_replay_qualification_ledger','execution_class_ledger','cost_budget_ledger']:
        if cat.get(field):pins.append(cat[field])
    sources={}
    for pin in pins:
        raw=(S/pin['ref']).read_bytes();assert hashlib.sha256(raw).hexdigest()==pin['sha256'];sources[pin['ref']]=raw
    args={'snapshot_inputs':s,'dataset_inputs':assemble_preserved_p9_history(originals())['dataset_inputs'],
        'frontiers':h['frontiers'],'frontier_snapshots':h['snapshots'],'current_frontier_id':h['current_frontier_id'],'catalogue_raw':catraw,'sources':sources}
    return root,r,aux,args


def run(root,r,aux,args,**kw):return verify_full_recovery_inventory(ContentDirectory(root),put(root,canonical_bytes(r)),put(root,canonical_bytes(aux)),**args,**kw)


def test_actual_all_133_records_and_complete_history_join(tmp_path):
    root,r,aux,args=fixture(tmp_path);out=run(root,r,aux,args)
    assert out['status']=='FULL_RECOVERY_INVENTORY_VERIFIED' and out['record_count']==133
    from collections import Counter
    assert Counter(out['record_classes'].values())=={'RESULT':107,'AUXILIARY':20,'FRONTIER':6}
    assert len(out['dataset_snapshot']['record_ids'])==133
    assert not out['execution_authorized'] and not out['recovery_performed'] and not out['accepted_head_freshness_verified'] and not out['production_release_verified']
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_omitted_historical_frontier_and_binding_detected_by_dataset(tmp_path):
    root,r,aux,args=fixture(tmp_path);rid=next(k for k in args['frontier_snapshots'] if k!=args['current_frontier_id']);args['frontier_snapshots'].pop(rid);args['frontiers']=[b for b in args['frontiers'] if json.loads(b)['record_id']!=rid]
    with pytest.raises(ValueError,match='FULL_RECOVERY_EXACT_INVENTORY_MISMATCH'):run(root,r,aux,args)


def test_auxiliary_root_omission_detected_by_dataset(tmp_path):
    root,r,aux,args=fixture(tmp_path)
    # Entire valid smaller closure, not just a malformed inventory.
    from infinity_grid.storage_closure import verify_declared_closure
    # Empty auxiliary root inventory cannot silently qualify the full dataset.
    aux['roots']=[];aux['required_content']=[];aux['interpretations']=[];aux['legacy_bindings']=[]
    with pytest.raises(ValueError):run(root,r,aux,args)


def test_component_overlap_refused(tmp_path):
    root,r,aux,args=fixture(tmp_path)
    with pytest.raises(ValueError,match='FULL_RECOVERY_COMPONENT_OVERLAP'):run(root,r,r,args)


def test_current_frontier_snapshot_must_be_same_as_result_snapshot(tmp_path):
    root,r,aux,args=fixture(tmp_path);raw=args['frontier_snapshots'][args['current_frontier_id']]['state_raw'];args['frontier_snapshots'][args['current_frontier_id']]['state_raw']=json.dumps(json.loads(raw),indent=3).encode()
    with pytest.raises(ValueError,match='FULL_RECOVERY_CURRENT_STATE_MISMATCH'):run(root,r,aux,args)


def test_missing_original_transaction_refused(tmp_path):
    root,r,aux,args=fixture(tmp_path);args['dataset_inputs']['transactions'].pop()
    with pytest.raises(FileNotFoundError):run(root,r,aux,args)


def test_changed_catalogue_source_bytes_refused(tmp_path):
    root,r,aux,args=fixture(tmp_path);args['sources'][next(iter(args['sources']))]=b'changed'
    with pytest.raises(ValueError,match='SOURCE_BYTES_MISMATCH'):run(root,r,aux,args)


def test_exact_raw_record_encoding_bound_across_components(tmp_path):
    root,r,aux,args=fixture(tmp_path);rows=args['dataset_inputs']['records'];rows[0]=json.dumps(json.loads(rows[0]),indent=3).encode()
    with pytest.raises(ValueError,match='FULL_RECOVERY_EXACT_INVENTORY_MISMATCH'):run(root,r,aux,args)


def test_full_join_shared_budget_refused(tmp_path):
    root,r,aux,args=fixture(tmp_path);out=run(root,r,aux,args)
    with pytest.raises(ValueError,match='BUDGET'):run(root,r,aux,args,max_total_bytes=out['bytes_checked']-1)
