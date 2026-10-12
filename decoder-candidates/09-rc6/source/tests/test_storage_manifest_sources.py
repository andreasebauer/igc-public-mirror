import hashlib
import json
import pytest
from test_replay_obligation_compiler_p1a2 import load_source, make_authorization
from infinity_grid.canon import canonical_bytes, canonical_sha256
from infinity_grid.replay_obligation_compiler import compile_obligations
from infinity_grid.storage_manifest_sources import verify_manifest_sources, ManifestSourceError


def pin(label,raw):return {'ref':label,'sha256':hashlib.sha256(raw).hexdigest()}


def fixture():
    s=load_source();raw=b'Frozen synthetic source, not a scientific result.\n'
    for node in s['obligations']:node['source_hashes']=[pin('source.txt',raw)]
    return s,{'source.txt':raw}


def run(s,blobs,**kw):
    return verify_manifest_sources(canonical_bytes(compile_obligations(s)),canonical_bytes(s),blobs,**kw)


def test_manifest_catalogue_recompiles_and_raw_pins_match():
    s,b=fixture();out=run(s,b)
    assert out['status']=='MANIFEST_SOURCES_VERIFIED' and out['records_checked']==3
    assert out['pin_occurrences_checked']==3 and out['dependency_closure_verified'] is False
    assert out['execution_authorized'] is False and out['scientific_acceptance']=='NOT_GRANTED'


def test_manifest_pretty_json_keeps_semantic_identity():
    s,b=fixture();m=compile_obligations(s)
    out=verify_manifest_sources(json.dumps(m,indent=2).encode(),json.dumps(s,indent=2).encode(),b)
    assert out['source_catalogue_sha256']==canonical_sha256(s)
    assert out['catalogue_content_ref']['sha256']!=out['source_catalogue_sha256']


def test_manifest_missing_source_refused():
    s,b=fixture()
    with pytest.raises(ManifestSourceError,match='INVENTORY_MISMATCH'):run(s,{})


def test_manifest_extra_source_refused():
    s,b=fixture();b['extra']=b'not referenced'
    with pytest.raises(ManifestSourceError,match='INVENTORY_MISMATCH'):run(s,b)


def test_manifest_altered_source_refused():
    s,b=fixture();b['source.txt']+=b'changed'
    with pytest.raises(ManifestSourceError,match='BYTES_MISMATCH'):run(s,b)


def test_manifest_wrong_catalogue_semantic_hash_refused():
    s,b=fixture();m=compile_obligations(s);s['status']='changed'
    with pytest.raises(ManifestSourceError,match='CATALOGUE_SEMANTIC_HASH_MISMATCH'):
        verify_manifest_sources(canonical_bytes(m),canonical_bytes(s),b)


def test_manifest_resealed_but_not_compiler_output_refused():
    s,b=fixture();m=compile_obligations(s);m['nodes'][0]['observer']='ALTERED';m.pop('dag_sha256');m['dag_sha256']=canonical_sha256(m)
    with pytest.raises(ManifestSourceError,match='RECOMPILED_MANIFEST_MISMATCH'):
        verify_manifest_sources(canonical_bytes(m),canonical_bytes(s),b)


def test_manifest_conflicting_label_pin_refused():
    s,b=fixture();s['obligations'][1]['source_hashes']=[pin('source.txt',b'different')]
    with pytest.raises(ManifestSourceError,match='CONFLICTING_SOURCE_PIN'):run(s,b)


def test_manifest_obligation_evidence_pin_is_required():
    s,b=fixture();raw=b'evidence';s['obligations'][0]['evidence_hashes']=[pin('evidence.bin',raw)]
    with pytest.raises(ManifestSourceError,match='INVENTORY_MISMATCH'):run(s,b)
    b['evidence.bin']=raw;assert run(s,b)['records_checked']==4


def test_manifest_audit_pins_are_required_and_not_authority():
    s,b=fixture();node=s['obligations'][0];auth=make_authorization(node['canonical_id'])
    for field in ['historical_source_hashes','audit_provenance','accepted_equivalence_certificates']:
        b[field]=field.encode();auth[field]=[pin(field,b[field])]
    auth.pop('decision_hash');auth['decision_hash']=canonical_sha256(auth)
    s['historical_audit_authorizations']=[auth];node['audit_authorization_ids']=[auth['authorization_id']]
    assert run(s,b)['records_checked']==6
    del b['audit_provenance']
    with pytest.raises(ManifestSourceError,match='INVENTORY_MISMATCH'):run(s,b)


def test_manifest_ledgers_are_required():
    s,b=fixture()
    for field in ['audit_provenance_ledger','known_replay_qualification_ledger','execution_class_ledger','cost_budget_ledger']:
        b[field]=field.encode();s[field]=pin(field,b[field])
    assert run(s,b)['records_checked']==7
    del b['cost_budget_ledger']
    with pytest.raises(ManifestSourceError,match='INVENTORY_MISMATCH'):run(s,b)


def test_manifest_labels_are_not_filesystem_paths():
    s,b=fixture();raw=b.pop('source.txt');label='../../never-open-this';b[label]=raw
    for node in s['obligations']:node['source_hashes']=[pin(label,raw)]
    assert run(s,b)['sources'][0]['ref']==label


def test_manifest_shared_byte_and_record_budgets():
    s,b=fixture();out=run(s,b)
    with pytest.raises(ManifestSourceError,match='BYTE_BUDGET'):run(s,b,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(ManifestSourceError,match='RECORD_BUDGET'):run(s,b,max_records=2)


def test_manifest_nonbytes_provider_and_invalid_budget_refused():
    s,b=fixture();b['source.txt']='not bytes'
    with pytest.raises(ManifestSourceError,match='BYTES_REQUIRED'):run(s,b)
    s,b=fixture()
    with pytest.raises(ManifestSourceError,match='INVALID_MANIFEST_SOURCE_BUDGET'):run(s,b,max_records=True)
