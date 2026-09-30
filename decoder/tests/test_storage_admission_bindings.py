from pathlib import Path
import json
import pytest
from test_storage_audit import audit
from test_storage_legacy import put
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_admission_bindings import review_admission_bindings,AdmissionBindingError,SCOPE


def fixture(tmp_path,obligations=None):
    store,req,decision,raw,b=audit(tmp_path,'CERTIFY_AND_ADVANCE')
    policy={'schema_id':'IG_STORAGE_ADMISSION_BINDING_POLICY_V1','purpose':'SCHEMA_FIXTURE',
        'verification_scope':SCOPE,'required_obligation_ids':obligations or ['FIXTURE_OBLIGATION'],'required_decision':'CERTIFY_AND_ADVANCE'}
    pref=put(store,canonical_bytes(policy))
    root=json.loads((Path(__file__).parent/'fixtures/storage_contract_v1/positive/ReleaseRoot.json').read_bytes());root['acceptance_policy_ref']=pref;rref=put(store,canonical_bytes(root))
    report={'status':'REFERENCE_CANDIDATE_VERIFIED','scope':SCOPE,'candidate_ref':rref,
        'scientific_acceptance':'NOT_GRANTED','production_release_verified':False,
        'metadata_semantics_verified':False,'execution_authorized':False,'publication_seal_verified':False,'recovery_performed':False}
    report_ref=put(store,canonical_bytes(report))
    scope={'release_root':rref,'acceptance_policy_ref':pref,'verification_report_ref':report_ref,'purpose':'SCHEMA_FIXTURE','verification_scope':SCOPE}
    decision['scope']=scope;decision['payload']['authorized_scope']=scope.copy();decision['payload']['evidence_hashes']=[{'ref':'fixture-only-report','sha256':report_ref['sha256']}]
    f={'store':store,'root':root,'root_ref':rref,'policy':policy,'policy_ref':pref,'report':report,'report_ref':report_ref,'decision':decision};reseal(f);return f


def reseal(f):
    d=seal_reference_record(f['decision']);f['decision']=d;raw=json.dumps(d,indent=2).encode()+b'\n';f['decisions']=[{'record_id':d['record_id'],'record_sha256':d['record_sha256'],'content_ref':put(f['store'],raw)}]


def run(f,**kw):return review_admission_bindings(ContentDirectory(f['store']),f['root_ref'],f['policy_ref'],f['report_ref'],f['decisions'],**kw)


def test_exact_bindings_remain_unauthenticated_and_unaccepted(tmp_path):
    f=fixture(tmp_path);before={p.name:p.stat().st_mtime_ns for p in f['store'].iterdir()};out=run(f)
    assert out['status']=='BINDINGS_REVIEWED_AUTHORITY_NOT_VERIFIED' and out['scientific_acceptance']=='NOT_GRANTED'
    for k in ['authority_authenticated','verification_report_truth_verified','production_release_verified','execution_authorized','publication_authorized']:assert out[k] is False
    assert out['decisions'][0]['limitations']==['NO_CURRENT_ACCEPTANCE']
    assert before=={p.name:p.stat().st_mtime_ns for p in f['store'].iterdir()}


def test_wrong_root_policy_pin_refused(tmp_path):
    f=fixture(tmp_path);f['policy_ref']=put(f['store'],b'other')
    with pytest.raises(AdmissionBindingError,match='ROOT_POLICY'):run(f)


def test_fixture_cannot_be_used_as_science_root(tmp_path):
    f=fixture(tmp_path);f['root']['purpose']='SCIENCE';f['root_ref']=put(f['store'],canonical_bytes(f['root']))
    with pytest.raises(AdmissionBindingError,match='ROOT_POLICY'):run(f)


def test_report_other_root_refused(tmp_path):
    f=fixture(tmp_path);f['report']['candidate_ref']=f['policy_ref'];f['report_ref']=put(f['store'],canonical_bytes(f['report']))
    with pytest.raises(AdmissionBindingError,match='REPORT_ROOT'):run(f)


def test_report_authority_claim_refused(tmp_path):
    f=fixture(tmp_path);f['report']['execution_authorized']=True;f['report_ref']=put(f['store'],canonical_bytes(f['report']))
    with pytest.raises(AdmissionBindingError,match='REPORT_ROOT'):run(f)


def test_decision_other_root_policy_and_report_refused(tmp_path):
    for field in ['release_root','acceptance_policy_ref','verification_report_ref']:
        p=tmp_path/field;p.mkdir();f=fixture(p);f['decision']['payload']['authorized_scope'][field]=put(f['store'],b'other');reseal(f)
        with pytest.raises(AdmissionBindingError,match='DECISION_ROOT_POLICY_REPORT_SCOPE'):run(f)


def test_continue_does_not_satisfy_certification_policy(tmp_path):
    f=fixture(tmp_path);f['decision']['payload']['decision']='CONTINUE';reseal(f)
    with pytest.raises(AdmissionBindingError,match='DECISION_ROOT_POLICY_REPORT_SCOPE'):run(f)


def test_exact_decision_inventory_required(tmp_path):
    f=fixture(tmp_path);f['decisions']=[]
    with pytest.raises(AdmissionBindingError,match='DECISION_INVENTORY'):run(f)
    reseal(f);f['decisions']*=2
    with pytest.raises(AdmissionBindingError,match='DUPLICATE_DECISION'):run(f)
    p=tmp_path/'partial';p.mkdir();g=fixture(p,['FIXTURE_OBLIGATION','SECOND'])
    with pytest.raises(AdmissionBindingError,match='MISSING_REQUIRED_DECISION'):run(g)


def test_wrong_obligation_and_missing_report_evidence_refused(tmp_path):
    f=fixture(tmp_path);f['decision']['payload']['obligation_ids']=['OTHER'];reseal(f)
    with pytest.raises(AdmissionBindingError,match='OBLIGATION_OVERLAP_OR_EXTRA'):run(f)
    f['decision']['payload']['obligation_ids']=['FIXTURE_OBLIGATION'];f['decision']['payload']['evidence_hashes']=[{'ref':'unrelated','sha256':'0'*64}];reseal(f)
    with pytest.raises(AdmissionBindingError,match='REPORT_EVIDENCE_REQUIRED'):run(f)


def test_unsupported_policy_and_noncanonical_report_refused(tmp_path):
    f=fixture(tmp_path);f['report_ref']=put(f['store'],json.dumps(f['report'],indent=2).encode())
    with pytest.raises(AdmissionBindingError,match='NONCANONICAL'):run(f)
    f['policy']['verification_scope']='FULL_REQUIRED_CLOSURE';f['policy_ref']=put(f['store'],canonical_bytes(f['policy']));f['root']['acceptance_policy_ref']=f['policy_ref'];f['root_ref']=put(f['store'],canonical_bytes(f['root']))
    with pytest.raises(AdmissionBindingError,match='UNSUPPORTED_ADMISSION_POLICY'):run(f)


def test_shared_budget_and_provider_lie_refused(tmp_path):
    f=fixture(tmp_path);out=run(f)
    with pytest.raises(AdmissionBindingError,match='BYTE_BUDGET'):run(f,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(AdmissionBindingError,match='INVALID_ADMISSION_BINDING_BUDGET'):run(f,max_total_bytes=True)
    class Liar:
        def read(self,ref,bound):return b'x'*int(ref['size_bytes'])
    with pytest.raises(AdmissionBindingError,match='PROVIDER_MISMATCH'):review_admission_bindings(Liar(),f['root_ref'],f['policy_ref'],f['report_ref'],f['decisions'])


def test_raw_and_semantic_decision_pins_both_required(tmp_path):
    f=fixture(tmp_path);f['decisions'][0]['record_sha256']='0'*64
    with pytest.raises(ValueError,match='BINDING_MISMATCH'):run(f)
    reseal(f);r=f['decisions'][0]['content_ref'];p=f['store']/(r['sha256']+'.blob');p.write_bytes(b'x'*int(r['size_bytes']))
    with pytest.raises(ValueError,match='DIGEST_MISMATCH'):run(f)
