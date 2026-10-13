import copy
import json
from pathlib import Path
import pytest
from test_storage_candidate_metadata import fixture as old_fixture, run
from test_storage_capture_metadata import archive, seal, change_archive
from test_storage_legacy import put
from test_storage_index_descriptor import report
from infinity_grid.canon import canonical_sha256
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory, build_collection_index, collection_records
from infinity_grid.storage_index_descriptor import content_ref
from infinity_grid.storage_capture_metadata import verify_capture_metadata
from infinity_grid.storage_candidate import assemble_reference_candidate, capture_metadata_descriptors, REGISTRY_LABEL
from infinity_grid.storage_contract_registry import registered_contract_objects, verify_contract_registry, SOURCE_PATHS


def fixture(tmp_path):
    f=list(old_fixture(tmp_path))
    root, _, d, _, kw, _, captured, capture=f
    registry, objects=registered_contract_objects()
    entries={p: objects[p] for p in SOURCE_PATHS}
    entries['infinity_grid/__init__.py']=b'raise RuntimeError("SOURCE MUST NOT EXECUTE")\n'
    rows=[[p,content_ref(raw)['sha256']] for p,raw in sorted(entries.items())]
    sid=canonical_sha256({'schema_id':'IG_DECODER_ENGINEERING_SOURCE_TREE_V1','files':rows})
    pid=canonical_sha256({'schema_id':'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1','files':[[p[len('infinity_grid/'):],h] for p,h in rows]})
    capture['workspace'].update(source_sha256=sid, package_sha256=pid)
    capture['job']['source_sha256']=sid
    change_archive(captured,capture,archive(list(entries.items())))
    ref=seal(captured,capture)
    for raw in captured.blobs.values():put(root,raw)
    for raw in objects.values():put(root,raw)
    checked=verify_capture_metadata(ContentDirectory(root),ref,source_paths=SOURCE_PATHS)
    source,env=capture_metadata_descriptors(checked)
    kw['metadata'].update(contract_freeze_ref=put(root,objects[SOURCE_PATHS[0]]),registry_ref=put(root,registry),source_ref=put(root,source),environment_ref=put(root,env))
    kw.update(capture_ref=ref,contract_registry=True)
    out=assemble_reference_candidate(ContentDirectory(root),lambda raw:put(root,raw),**kw)
    dest=tmp_path/'registry-index'
    built=build_collection_index(ContentDirectory(root),kw['collection'],dest,release_root=out['candidate_ref'],collection_kind='reference_records',purpose='SCHEMA_FIXTURE')
    d.update(release_root=out['candidate_ref'],index_content_ref=built['index']['index_ref'])
    d['validation_report_ref']=report(root,d,dest)
    f[1]=out;f[3]=dest;f[5]=checked
    return f


def test_registry_candidate_closure_and_explicit_nonclaims(tmp_path):
    f=fixture(tmp_path);result=run(f)
    assert result['inventory']['record_count']=='133'
    assert result['frozen_contract_and_reference_registry_verified'] is True
    assert result['metadata_semantics_verified'] is False and result['execution_authorized'] is False
    r=result['contract_registry_verification']
    assert r['source_member_count']==len(SOURCE_PATHS)
    assert r['production_registry_verified'] is False and r['transitive_code_closure_verified'] is False
    assert json.loads(f[1]['candidate_raw'])['label']==REGISTRY_LABEL
    root=json.loads(f[1]['candidate_raw'])
    rows=collection_records(ContentDirectory(f[0]),root['required_content_inventory'],collection_kind='required_content_inventory',purpose='SCHEMA_FIXTURE')
    hashes={json.loads(x.raw)['content_ref']['sha256'] for x in rows}
    _,objects=registered_contract_objects()
    assert all(content_ref(raw)['sha256'] in hashes for raw in objects.values())
    assert f[1]['candidate_ref']['sha256'] not in hashes
    again=assemble_reference_candidate(ContentDirectory(f[0]),lambda raw:put(f[0],raw),**f[4])
    assert again['candidate_ref']==f[1]['candidate_ref']


def test_registry_resealed_mutations_refused(tmp_path):
    f=fixture(tmp_path);raw,_=registered_contract_objects();value=json.loads(raw)
    for change in ({'production_registry':True},{'formats':['ANY']},{'science_authority':'GRANTED'},{'objects':{}}):
        changed=dict(value,**change)
        f[4]['metadata']['registry_ref']=put(f[0],canonical_bytes(changed))
        with pytest.raises(ValueError,match='UNSUPPORTED_CONTRACT_REGISTRY'):run(f)
    f[4]['metadata']['registry_ref']=put(f[0],json.dumps(value,indent=2).encode())
    with pytest.raises(ValueError,match='UNSUPPORTED_CONTRACT_REGISTRY'):run(f)


def test_frozen_contract_substitution_refused(tmp_path):
    f=fixture(tmp_path);f[4]['metadata']['contract_freeze_ref']=put(f[0],b'new contract')
    with pytest.raises(ValueError,match='FROZEN_CONTRACT_REFERENCE_MISMATCH'):run(f)


def test_each_registry_object_required(tmp_path):
    f=fixture(tmp_path);_,objects=registered_contract_objects()
    for raw in objects.values():
        ref=content_ref(raw);p=f[0]/(ref['sha256']+'.blob');p.unlink()
        with pytest.raises(ValueError,match='MISSING_CONTENT'):run(f)
        put(f[0],raw)


def test_registry_source_pins_reject_changed_and_missing_members(tmp_path):
    f=fixture(tmp_path);checked=copy.deepcopy(f[5]);kw=f[4]
    for path in SOURCE_PATHS:
        changed=copy.deepcopy(checked);changed['selected_source_member_sha256'][path]='0'*64
        with pytest.raises(ValueError,match='REGISTRY_CAPTURE_SOURCE_MISMATCH'):
            verify_contract_registry(ContentDirectory(f[0]),kw['metadata']['contract_freeze_ref'],kw['metadata']['registry_ref'],changed)
    checked['selected_source_member_sha256'].pop(SOURCE_PATHS[0])
    with pytest.raises(ValueError,match='REGISTRY_CAPTURE_SOURCE_MEMBERS'):
        verify_contract_registry(ContentDirectory(f[0]),kw['metadata']['contract_freeze_ref'],kw['metadata']['registry_ref'],checked)


def test_registry_profile_requires_capture_and_cannot_downgrade(tmp_path):
    f=fixture(tmp_path)
    with pytest.raises(ValueError,match='PROFILE_REQUIRED'):run(f,contract_registry=False)
    with pytest.raises(ValueError,match='REGISTRY_PROFILE_REQUIRES_CAPTURE'):run(f,capture_ref=None)
    for v in (1,'yes',None):
        with pytest.raises(ValueError,match='REGISTRY_PROFILE_REQUIRES_CAPTURE'):run(f,contract_registry=v)
    kw=dict(f[4],capture_ref=None)
    with pytest.raises(ValueError,match='REGISTRY_PROFILE_REQUIRES_CAPTURE'):
        assemble_reference_candidate(ContentDirectory(f[0]),lambda raw:put(f[0],raw),**kw)


def test_registry_reads_share_budget_and_do_not_mutate_index(tmp_path):
    f=fixture(tmp_path);before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in f[3].iterdir()}
    r=run(f)
    assert r['bytes_checked']>r['contract_registry_verification']['bytes_checked']
    with pytest.raises(ValueError,match='BUDGET'):run(f,max_total_bytes=r['bytes_checked']-1)
    assert before=={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in f[3].iterdir()}


def test_selected_source_paths_validation_and_no_import(tmp_path):
    f=fixture(tmp_path)
    for paths in (['x'],('x','x'),('../x',),('/x',),('x\\y',),('x//y',),('x/./y',)):
        with pytest.raises(ValueError,match='METADATA_SOURCE_PATHS'):
            verify_capture_metadata(ContentDirectory(f[0]),f[4]['capture_ref'],source_paths=paths)
    r=verify_capture_metadata(ContentDirectory(f[0]),f[4]['capture_ref'],source_paths=SOURCE_PATHS)
    assert r['selected_source_member_sha256']==f[5]['selected_source_member_sha256']


def test_resealed_capture_with_altered_reader_rejected_by_candidate(tmp_path):
    import io
    import zipfile
    f=fixture(tmp_path);root,_,_,_,kw,checked,captured,capture=f
    with zipfile.ZipFile(io.BytesIO(captured.blobs[checked['source_ref']['sha256']])) as z:
        entries={n:z.read(n) for n in z.namelist()}
    entries['infinity_grid/storage_collections.py']+=b'\n# altered reader\n'
    rows=[[p,content_ref(raw)['sha256']] for p,raw in sorted(entries.items())]
    sid=canonical_sha256({'schema_id':'IG_DECODER_ENGINEERING_SOURCE_TREE_V1','files':rows})
    pid=canonical_sha256({'schema_id':'IG_DECODER_EXECUTABLE_SOURCE_TREE_V1','files':[[p[len('infinity_grid/'):],h] for p,h in rows]})
    capture['workspace'].update(source_sha256=sid,package_sha256=pid);capture['job']['source_sha256']=sid
    change_archive(captured,capture,archive(list(entries.items())));ref=seal(captured,capture)
    for raw in captured.blobs.values():put(root,raw)
    verified=verify_capture_metadata(ContentDirectory(root),ref,source_paths=SOURCE_PATHS)
    source,env=capture_metadata_descriptors(verified)
    kw.update(capture_ref=ref)
    kw['metadata'].update(source_ref=put(root,source),environment_ref=put(root,env))
    with pytest.raises(ValueError,match='REGISTRY_CAPTURE_SOURCE_MISMATCH'):
        assemble_reference_candidate(ContentDirectory(root),lambda raw:put(root,raw),**kw)
