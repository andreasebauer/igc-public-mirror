import json
import pytest
from test_storage_collection_recovery import fixture as recovery_fixture
from test_storage_capture_metadata import fixture as capture_fixture, seal as seal_capture
from test_storage_index_descriptor import report
from test_storage_legacy import put
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory, build_collection_index, collection_records
from infinity_grid.storage_index_descriptor import registered_profiles
from infinity_grid.storage_capture_metadata import verify_capture_metadata
from infinity_grid.storage_candidate import (assemble_reference_candidate, verify_reference_candidate,
                                            capture_metadata_descriptors, CAPTURE_LABEL, LABEL)


def fixture(tmp_path):
    root, entries, col, r, aux, args = recovery_fixture(tmp_path)
    store = ContentDirectory(root)
    captured, capture, binding, archive_ref = capture_fixture(tmp_path / 'metadata')
    capture_ref = seal_capture(captured, capture)
    for raw in captured.blobs.values():
        put(root, raw)
    checked = verify_capture_metadata(store, capture_ref)
    source_raw, env_raw = capture_metadata_descriptors(checked)
    metadata = {k: put(root, canonical_bytes({'fixture': k})) for k in
                ('contract_freeze_ref', 'registry_ref', 'acceptance_policy_ref')}
    metadata.update(source_ref=put(root, source_raw), environment_ref=put(root, env_raw))
    kw = dict(collection=col, metadata=metadata, result_request_ref=r,
              auxiliary_request_ref=aux, recovery_inputs=args, capture_ref=capture_ref)
    out = assemble_reference_candidate(store, lambda raw: put(root, raw), **kw)
    dest = tmp_path / 'index'
    built = build_collection_index(store, col, dest, release_root=out['candidate_ref'],
                                   collection_kind='reference_records', purpose='SCHEMA_FIXTURE')
    b, s = registered_profiles()
    d = {'schema_id': 'IG_STORAGE_INDEXDESCRIPTOR_V1', 'contract_version': '1.0.0',
         'purpose': 'SCHEMA_FIXTURE', 'authority': 'REBUILDABLE_PROJECTION',
         'release_root': out['candidate_ref'], 'source_collections': [col['root']],
         'index_content_ref': built['index']['index_ref'], 'builder_ref': put(root, b),
         'index_schema_ref': put(root, s), 'validation_report_ref': put(root, b'{}'), 'extensions': []}
    d['validation_report_ref'] = report(root, d, dest)
    return root, out, d, dest, kw, checked, captured, capture


def run(f, **overrides):
    root, out, d, dest, kw, *_ = f
    kw = dict(kw)
    kw.pop('collection')
    kw.update(overrides)
    return verify_reference_candidate(ContentDirectory(root), out['candidate_ref'],
                                      canonical_bytes(d), dest, **kw)


def alter_descriptor(f, field, change, canonical=True):
    root, _, _, _, kw, *_ = f
    value = json.loads((root / (kw['metadata'][field]['sha256'] + '.blob')).read_bytes())
    change(value)
    raw = canonical_bytes(value) if canonical else json.dumps(value, indent=2).encode()
    kw['metadata'][field] = put(root, raw)


def test_joined_133_candidate_includes_capture_bytes_and_retains_limits(tmp_path):
    f = fixture(tmp_path)
    root, out, _, dest, kw, checked, *_ = f
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.iterdir()}
    result = run(f)
    assert result['source_and_runtime_declarations_verified'] is True
    assert result['metadata_semantics_verified'] is False
    assert result['scientific_acceptance'] == 'NOT_GRANTED'
    assert result['production_release_verified'] is False
    assert result['scope'] == 'REFERENCE_ONLY_FIXTURE_ROOT_RECOVERY_INDEX_AND_CAPTURE_METADATA'
    assert result['capture_metadata_verification'] == checked
    assert result['inventory']['record_count'] == '133'
    rr = json.loads(out['candidate_raw'])
    assert rr['label'] == CAPTURE_LABEL
    rows = list(collection_records(ContentDirectory(root), rr['required_content_inventory'],
                                   collection_kind='required_content_inventory', purpose='SCHEMA_FIXTURE'))
    refs = {json.loads(row.raw)['content_ref']['sha256'] for row in rows}
    for key in ('capture_ref', 'source_ref', 'runtime_binding_ref'):
        assert checked[key]['sha256'] in refs
    assert out['candidate_ref']['sha256'] not in refs
    assert len(checked['unobserved_requirements']) == 12
    assert len(checked['unread_environment_artifacts']) == 3
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in dest.iterdir()}


def test_source_descriptor_identity_substitution_refused(tmp_path):
    f = fixture(tmp_path)
    alter_descriptor(f, 'source_ref', lambda v: v.update(source_sha256='0' * 64))
    with pytest.raises(ValueError, match='CAPTURE_METADATA_BINDING'):
        run(f)


def test_environment_descriptor_hash_substitution_refused(tmp_path):
    f = fixture(tmp_path)
    alter_descriptor(f, 'environment_ref', lambda v: v.update(environment_sha256='0' * 64))
    with pytest.raises(ValueError, match='CAPTURE_METADATA_BINDING'):
        run(f)


def test_valid_other_capture_cannot_satisfy_root_metadata(tmp_path):
    f = fixture(tmp_path)
    captured, capture = f[6:]
    capture['another_capture'] = True
    ref = seal_capture(captured, capture)
    put(f[0], captured.blobs[ref['sha256']])
    with pytest.raises(ValueError, match='CAPTURE_METADATA_BINDING'):
        run(f, capture_ref=ref)


def test_each_required_capture_dependency_must_exist(tmp_path):
    f = fixture(tmp_path)
    for key in ('capture_ref', 'source_ref', 'runtime_binding_ref'):
        p = f[0] / (f[5][key]['sha256'] + '.blob')
        raw = p.read_bytes()
        p.unlink()
        with pytest.raises(ValueError, match='MISSING_CONTENT'):
            run(f)
        put(f[0], raw)


def test_profile_downgrade_and_label_substitution_refused(tmp_path):
    f = fixture(tmp_path)
    with pytest.raises(ValueError, match='PROFILE_REQUIRED'):
        run(f, capture_ref=None)
    rr = json.loads(f[1]['candidate_raw'])
    rr['label'] = LABEL
    f[1]['candidate_ref'] = put(f[0], canonical_bytes(rr))
    with pytest.raises(ValueError, match='PROFILE_REQUIRED'):
        run(f)


def test_joined_inventory_omission_cannot_be_resealed(tmp_path):
    f = fixture(tmp_path)
    rr = json.loads(f[1]['candidate_raw'])
    rr['required_content_inventory']['row_count'] = '0'
    f[1]['candidate_ref'] = put(f[0], canonical_bytes(rr))
    with pytest.raises(ValueError, match='ROOT_OR_INVENTORY_MISMATCH'):
        run(f)


def test_metadata_reads_share_candidate_byte_budget(tmp_path):
    f = fixture(tmp_path)
    result = run(f)
    assert result['bytes_checked'] > result['capture_metadata_verification']['bytes_checked']
    with pytest.raises(ValueError, match='BUDGET'):
        run(f, max_total_bytes=result['bytes_checked'] - 1)


def test_descriptor_report_is_not_trusted_and_noncanonical_refused(tmp_path):
    f = fixture(tmp_path)
    forged = dict(f[5], source_sha256='0' * 64)
    raw, _ = capture_metadata_descriptors(forged)
    f[4]['metadata']['source_ref'] = put(f[0], raw)
    with pytest.raises(ValueError, match='CAPTURE_METADATA_BINDING'):
        run(f)
    source, _ = capture_metadata_descriptors(f[5])
    f[4]['metadata']['source_ref'] = put(f[0], source)
    alter_descriptor(f, 'environment_ref', lambda v: None, canonical=False)
    with pytest.raises(ValueError, match='CAPTURE_METADATA_BINDING'):
        run(f)


def test_assembly_requires_verified_metadata_and_is_deterministic(tmp_path):
    f = fixture(tmp_path)
    root, out, _, _, kw, *_ = f
    again = assemble_reference_candidate(ContentDirectory(root), lambda raw: put(root, raw), **kw)
    assert again['candidate_ref'] == out['candidate_ref']
    alter_descriptor(f, 'environment_ref', lambda v: v.update(purpose='SCIENCE'))
    with pytest.raises(ValueError, match='CAPTURE_METADATA_BINDING'):
        assemble_reference_candidate(ContentDirectory(root), lambda raw: put(root, raw), **kw)
