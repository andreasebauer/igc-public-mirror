import json
import pytest
from test_storage_full_recovery_join import fixture as recovery_fixture
from test_storage_native_collection import seal
from test_storage_legacy import put
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory,CollectionReadError
from infinity_grid.storage_collection_recovery import verify_collection_recovery
from infinity_grid.storage_closure import ClosureError


def fixture(tmp_path):
    root,r,aux,args=recovery_fixture(tmp_path);entries=[]
    for raw in args['dataset_inputs']['records']:
        rec=json.loads(raw);entries.append({'record_id':rec['record_id'],'record_sha256':rec['record_sha256'],'content_ref':put(root,raw)})
    entries.sort(key=lambda r:canonical_bytes([r['record_id']]).hex())
    return root,entries,seal(root,entries),put(root,canonical_bytes(r)),put(root,canonical_bytes(aux)),args


def run(root,ref,r,aux,args,**kw):return verify_collection_recovery(ContentDirectory(root),ref,r,aux,purpose='SCHEMA_FIXTURE',recovery_inputs=args,**kw)


def test_actual_133_collection_members_have_verified_dependency_history(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);out=run(root,ref,r,aux,args)
    assert out['status']=='COLLECTION_RECOVERY_VERIFIED' and out['record_count']==133 and out['collection_inventory']['record_count']=='133'
    assert out['recovery_verification']['record_count']==133 and out['collection_bytes_checked']>0
    assert out['bytes_checked']==out['recovery_verification']['bytes_checked']+out['collection_bytes_checked']
    assert not out['production_release_verified'] and not out['execution_authorized'] and not out['recovery_performed'] and out['scientific_acceptance']=='NOT_GRANTED'


def test_valid_collection_omitting_frontier_cannot_hide_from_recovery(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);rid=args['current_frontier_id'];e=[x for x in e if x['record_id']!=rid]
    with pytest.raises(ClosureError,match='EXACT_INVENTORY_MISMATCH'):run(root,seal(root,e),r,aux,args)


def test_valid_collection_omitting_auxiliary_record_refused(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);rid=next(json.loads(raw)['record_id'] for raw in args['dataset_inputs']['records'] if json.loads(raw)['record_type']=='GENERATION_RECIPE');e=[x for x in e if x['record_id']!=rid]
    with pytest.raises(ClosureError,match='EXACT_INVENTORY_MISMATCH'):run(root,seal(root,e),r,aux,args)


def test_same_seal_different_raw_collection_encoding_refused(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);entry=e[0];raw=(root/(entry['content_ref']['sha256']+'.blob')).read_bytes();entry['content_ref']=put(root,json.dumps(json.loads(raw),indent=3).encode())
    with pytest.raises(ClosureError,match='COLLECTION_RECOVERY_EXACT_INVENTORY_MISMATCH'):run(root,seal(root,e),r,aux,args)


def test_collection_cannot_hide_missing_observation_dependency(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);request=json.loads((root/(r['sha256']+'.blob')).read_bytes())
    for row in request['legacy_bindings']:
        b=json.loads((root/(row['binding_ref']['sha256']+'.blob')).read_bytes())
        if b['semantic_bindings']:
            (root/(b['semantic_bindings'][0]['content_ref']['sha256']+'.blob')).unlink();break
    with pytest.raises(CollectionReadError,match='MISSING_CONTENT'):run(root,ref,r,aux,args)


def test_collection_cannot_hide_missing_original_transaction(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);args['dataset_inputs']['transactions'].pop()
    with pytest.raises(FileNotFoundError):run(root,ref,r,aux,args)


def test_collection_recovery_shared_byte_budget_enforced(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);out=run(root,ref,r,aux,args)
    with pytest.raises(CollectionReadError,match='BYTE_BUDGET'):run(root,ref,r,aux,args,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(ClosureError,match='INVALID_COLLECTION_RECOVERY_BUDGET'):run(root,ref,r,aux,args,max_total_bytes=True)


def test_collection_recovery_rejects_other_schema_and_input_override(tmp_path):
    root,e,ref,r,aux,args=fixture(tmp_path);ref['record_schema_id']='IG_STORAGE_OBJECT_V1'
    with pytest.raises(ClosureError,match='NATIVE_RECOVERY_COLLECTION_REQUIRED'):run(root,ref,r,aux,args)
    args['max_total_bytes']=67108864
    with pytest.raises(ClosureError,match='INVALID_COLLECTION_RECOVERY_INPUTS'):run(root,ref,r,aux,args)
