import json
import pytest
from test_storage_collection_recovery import fixture as recovery_fixture
from test_storage_index_descriptor import report
from test_storage_legacy import put
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory,build_collection_index
from infinity_grid.storage_index_descriptor import registered_profiles
from infinity_grid.storage_candidate import assemble_reference_candidate,verify_reference_candidate
from infinity_grid.storage_closure import ClosureError
from infinity_grid.storage_catalog import ReadIndexError


def fixture(tmp_path):
    root,entries,col,r,aux,args=recovery_fixture(tmp_path);store=ContentDirectory(root)
    metadata={k:put(root,canonical_bytes({'fixture':k})) for k in
        ('contract_freeze_ref','source_ref','environment_ref','registry_ref','acceptance_policy_ref')}
    kw={'collection':col,'metadata':metadata,'result_request_ref':r,'auxiliary_request_ref':aux,'recovery_inputs':args}
    out=assemble_reference_candidate(store,lambda raw:put(root,raw),**kw)
    dest=tmp_path/'index';built=build_collection_index(store,col,dest,release_root=out['candidate_ref'],collection_kind='reference_records',purpose='SCHEMA_FIXTURE')
    b,s=registered_profiles();d={'schema_id':'IG_STORAGE_INDEXDESCRIPTOR_V1','contract_version':'1.0.0',
        'purpose':'SCHEMA_FIXTURE','authority':'REBUILDABLE_PROJECTION','release_root':out['candidate_ref'],
        'source_collections':[col['root']],'index_content_ref':built['index']['index_ref'],
        'builder_ref':put(root,b),'index_schema_ref':put(root,s),'validation_report_ref':put(root,b'{}'),'extensions':[]}
    d['validation_report_ref']=report(root,d,dest)
    return root,out,d,dest,kw


def run(f,**extra):
    root,out,d,dest,kw=f;kw=dict(kw);kw.pop('collection')
    return verify_reference_candidate(ContentDirectory(root),out['candidate_ref'],canonical_bytes(d),dest,**kw,**extra)


def change_root(f,mutate):
    root,out,d,dest,kw=f;record=json.loads(out['candidate_raw']);mutate(record);raw=canonical_bytes(record)
    out['candidate_raw']=raw;out['candidate_ref']=put(root,raw)


def test_actual_133_candidate_joins_recovery_and_index_read_only(tmp_path):
    f=fixture(tmp_path);root,out,d,dest,kw=f;before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in dest.iterdir()};checked=run(f)
    assert checked['status']=='REFERENCE_CANDIDATE_VERIFIED' and checked['inventory']['record_count']=='133'
    assert checked['recovery_verification']['record_count']==133 and checked['required_content_count']>133
    assert checked['index_verification']['inventory']==checked['inventory']
    assert all(checked[k] is False for k in ['production_release_verified','metadata_semantics_verified','execution_authorized','publication_seal_verified','recovery_performed'])
    assert checked['scientific_acceptance']=='NOT_GRANTED'
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in dest.iterdir()}


def test_candidate_assembly_deterministic_and_no_self_reference(tmp_path):
    f=fixture(tmp_path);root,out,d,dest,kw=f;again=assemble_reference_candidate(ContentDirectory(root),lambda b:put(root,b),**kw)
    assert again['candidate_raw']==out['candidate_raw'] and again['candidate_ref']==out['candidate_ref']
    obj=json.loads(out['candidate_raw']);assert out['candidate_ref']['sha256'].encode() not in out['candidate_raw']
    assert d['validation_report_ref']['sha256'].encode() not in out['candidate_raw'] and obj['previous_release_root'] is None


def test_candidate_metadata_substitution_refused(tmp_path):
    f=fixture(tmp_path);f[4]['metadata']['source_ref']=put(f[0],b'other')
    with pytest.raises(ClosureError,match='ROOT_OR_INVENTORY_MISMATCH'):run(f)


def test_candidate_required_inventory_omission_refused(tmp_path):
    f=fixture(tmp_path);change_root(f,lambda r:r['required_content_inventory'].update(row_count='0'))
    with pytest.raises(ClosureError,match='ROOT_OR_INVENTORY_MISMATCH'):run(f)


def test_candidate_required_inventory_extra_or_replaced_refused(tmp_path):
    f=fixture(tmp_path);change_root(f,lambda r:r['required_content_inventory'].update(root=put(f[0],b'other')))
    with pytest.raises(ClosureError,match='ROOT_OR_INVENTORY_MISMATCH'):run(f)


def test_candidate_missing_reconstruction_bytes_refused(tmp_path):
    f=fixture(tmp_path);raw=f[4]['recovery_inputs']['dataset_inputs']['transactions'][0]
    import hashlib
    (f[0]/(hashlib.sha256(raw).hexdigest()+'.blob')).unlink()
    with pytest.raises(ValueError,match='MISSING_CONTENT'):run(f)


def test_candidate_changed_input_tree_refused(tmp_path):
    f=fixture(tmp_path);f[4]['recovery_inputs']['dataset_inputs']['transactions'].reverse()
    with pytest.raises(ValueError,match='MISSING_CONTENT|ROOT_OR_INVENTORY_MISMATCH'):run(f)


def test_candidate_rejects_native_science_and_lineage_claims(tmp_path):
    for field in ['purpose','previous_release_root','extensions']:
        folder=tmp_path/field;folder.mkdir();f=fixture(folder)
        def change(r):
            if field=='purpose':r[field]='SCIENCE'
            elif field=='previous_release_root':r[field]=f[4]['metadata']['source_ref']
            else:r['collections']['objects']['row_count']='1'
        change_root(f,change)
        with pytest.raises(ClosureError,match='PROFILE_REQUIRED|ROOT_OR_INVENTORY_MISMATCH'):run(f)


def test_candidate_structural_inventory_bytes_required(tmp_path):
    f=fixture(tmp_path);obj=json.loads(f[1]['candidate_raw']);page=json.loads((f[0]/(obj['required_content_inventory']['root']['sha256']+'.blob')).read_bytes())
    (f[0]/(page['entries'][0]['content_ref']['sha256']+'.blob')).unlink()
    with pytest.raises(ValueError,match='MISSING_CONTENT'):run(f)


def test_candidate_index_must_bind_fixed_root(tmp_path):
    f=fixture(tmp_path);f[2]['release_root']=f[4]['collection']['root']
    with pytest.raises(ReadIndexError,match='DESCRIPTOR_SCOPE_MISMATCH'):run(f)


def test_candidate_shared_byte_budget_and_provider_refusal(tmp_path):
    f=fixture(tmp_path);out=run(f)
    with pytest.raises(ValueError,match='BUDGET'):run(f,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(ValueError,match='INVALID_CANDIDATE_BUDGET'):run(f,max_total_bytes=True)
    class Liar:
        def read(self,ref,bound):return b'x'*int(ref['size_bytes'])
    kw=dict(f[4]);kw.pop('collection')
    with pytest.raises(ClosureError,match='PROVIDER_MISMATCH'):verify_reference_candidate(Liar(),f[1]['candidate_ref'],canonical_bytes(f[2]),f[3],**kw)


def test_candidate_writer_failure_and_metadata_fields_fail_closed(tmp_path):
    root,entries,col,r,aux,args=recovery_fixture(tmp_path);kw={'collection':col,'metadata':{},'result_request_ref':r,'auxiliary_request_ref':aux,'recovery_inputs':args}
    with pytest.raises(ClosureError,match='METADATA_FIELDS'):assemble_reference_candidate(ContentDirectory(root),lambda b:put(root,b),**kw)
    kw['metadata']={k:put(root,b'fixture') for k in ('contract_freeze_ref','source_ref','environment_ref','registry_ref','acceptance_policy_ref')}
    with pytest.raises(ClosureError,match='WRITER_MISMATCH'):assemble_reference_candidate(ContentDirectory(root),lambda b:{'sha256':'0'*64,'size_bytes':str(len(b))},**kw)
