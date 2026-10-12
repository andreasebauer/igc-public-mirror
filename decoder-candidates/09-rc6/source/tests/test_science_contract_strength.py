"""A08 data fixtures; execution only inside the registered qualification."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from infinity_grid import result_contracts as rc, submission as sub
from infinity_grid.canon import canonical_sha256


def contract():
    return {'schema_id':rc.SCIENCE_SCHEMA,'claim':'SCIENCE',
            'required_artifacts':[{'path':'answer.json','json_checks':[{'pointer':'/rows','count':0}, {'pointer':'/established','equals':False}]}],
            'scientific_content':['answer.json'],'result_checks':[{'pointer':'/status','equals':'PASS'}],
            'outcome':{'artifact':'answer.json','pointer':'/outcome'},'prerequisites':[],'preservation':{}}


def normalized(c=None):
    return rc.normalize(c or contract(),{'kind':'SCRIPT'},{'outcomes':['NO_EFFECT','NOT_ESTABLISHED']})


def answer(root, **kw):
    value={'rows':[],'established':False,'outcome':'NOT_ESTABLISHED'};value.update(kw)
    (root/'answer.json').write_text(json.dumps(value))


def test_negative_conformance_has_no_science_authority(tmp_path):
    answer(tmp_path)
    result=rc.verify(normalized(),{'status':'PASS'},tmp_path)
    assert result['status']=='VERIFIED' and result['scientific_outcome']=='NOT_ESTABLISHED'
    assert result['schema_id']=='IG_DECODER_RESULT_VERIFICATION_V2'
    assert result['science_qualification_authority']=='NONE'
    assert result['conformance_scope']=='DECLARED_CONTENT_ONLY'
    answer(tmp_path,outcome='NO_EFFECT')
    assert rc.verify(normalized(),{'status':'PASS'},tmp_path)['scientific_outcome']=='NO_EFFECT'


def test_v1_science_requires_explicit_new_admission(tmp_path):
    c=contract();c['schema_id']=rc.SCHEMA;c.pop('scientific_content')
    before=canonical_sha256(c)
    with pytest.raises(sub.SubmissionError,match='SCIENCE_V2'):normalized(c)
    n=dict(c,declared_outcomes=['NOT_ESTABLISHED'])
    answer(tmp_path)
    assert rc.verify(n,{'status':'PASS'},tmp_path)['status']=='REJECTED'
    assert canonical_sha256(c)==before


def test_presence_and_process_envelope_cannot_be_content():
    for checks in ([],[{'pointer':'/status','equals':'PASS'}],[{'pointer':'/outcome','equals':'PROCESS_COMPLETED'}], [{'pointer':'/return_code','equals':0}]):
        c=contract();c['required_artifacts'][0]['json_checks']=checks
        with pytest.raises(sub.SubmissionError,match='SCIENTIFIC_CONTENT_CHECK_REQUIRED'):normalized(c)
    c=contract();c['scientific_content']=[]
    with pytest.raises(sub.SubmissionError,match='SCIENTIFIC_CONTENT_PATHS'):normalized(c)


def test_all_present_files_do_not_bypass_typed_or_missing_assertions(tmp_path):
    c=normalized()
    for changes in ({'established':0},{'rows':[1]}):
        answer(tmp_path,**changes)
        r=rc.verify(c,{'status':'PASS'},tmp_path)
        assert r['status']=='REJECTED' and any(x['reason']=='VALUE_MISMATCH' for x in r['failures'])
    (tmp_path/'answer.json').write_text('{"outcome":"NO_EFFECT","rows":[]}')
    assert any(x['reason']=='VALUE_MISSING' for x in rc.verify(c,{'status':'PASS'},tmp_path)['failures'])


def test_expected_hash_is_independent_of_presence(tmp_path):
    answer(tmp_path);c=contract();row=c['required_artifacts'][0];row.pop('json_checks');row['sha256']=hashlib.sha256((tmp_path/'answer.json').read_bytes()).hexdigest()
    n=normalized(c);assert rc.verify(n,{'status':'PASS'},tmp_path)['status']=='VERIFIED'
    answer(tmp_path,rows=[1]);r=rc.verify(n,{'status':'PASS'},tmp_path)
    assert r['status']=='REJECTED' and any(x['reason']=='ARTIFACT_HASH' for x in r['failures'])


def test_supporting_artifacts_are_explicit_and_still_required(tmp_path):
    answer(tmp_path);(tmp_path/'notes.txt').write_text('support only')
    c=contract();c['required_artifacts'].append({'path':'notes.txt'});n=normalized(c)
    r=rc.verify(n,{'status':'PASS'},tmp_path)
    assert r['status']=='VERIFIED'
    assert r['artifact_roles']==[{'path':'answer.json','role':'SCIENTIFIC_CONTENT'},{'path':'notes.txt','role':'SUPPORTING'}]
    (tmp_path/'notes.txt').unlink();assert rc.verify(n,{'status':'PASS'},tmp_path)['status']=='REJECTED'
    c['scientific_content'].append('notes.txt')
    with pytest.raises(sub.SubmissionError,match='SCIENTIFIC_CONTENT_CHECK_REQUIRED'):normalized(c)


def test_missing_and_unregistered_outcome_refuse(tmp_path):
    for value in ({'rows':[],'established':False},{'rows':[],'established':False,'outcome':'UNREGISTERED'}):
        (tmp_path/'answer.json').write_text(json.dumps(value))
        r=rc.verify(normalized(),{'status':'PASS'},tmp_path)
        assert r['status']=='REJECTED' and r['scientific_outcome'] is None
    c=rc.normalize(contract(),{'kind':'SCRIPT'},{'outcomes':['PROCESS_COMPLETED','PROCESS_FAILED','NO_EFFECT']})
    answer(tmp_path,outcome='PROCESS_COMPLETED')
    assert rc.verify(c,{'status':'PASS'},tmp_path)['scientific_outcome'] is None
    assert rc.verify(c,{'status':'PASS'},tmp_path)['status']=='REJECTED'
    for outcomes in ([],['PASS'],['NO_EFFECT','NO_EFFECT']):
        with pytest.raises(sub.SubmissionError):rc.normalize(contract(),{'kind':'SCRIPT'},{'outcomes':outcomes})


def test_missing_symlink_and_unsafe_content_refuse(tmp_path):
    n=normalized();assert rc.verify(n,{'status':'PASS'},tmp_path)['status']=='REJECTED'
    other=tmp_path/'other';other.mkdir();answer(other);(tmp_path/'answer.json').symlink_to(other/'answer.json')
    assert rc.verify(n,{'status':'PASS'},tmp_path)['status']=='REJECTED'
    c=contract();c['required_artifacts'][0]['path']='../escape'
    with pytest.raises(sub.SubmissionError):normalized(c)


def test_strict_json_rejects_duplicate_nonfinite_overflow_and_malformed(tmp_path):
    for raw in (b'{"rows":[],"established":true,"established":false,"outcome":"NO_EFFECT"}',
                b'{"rows":[],"established":false,"outcome":"NO_EFFECT","x":NaN}',
                b'{"rows":[],"established":false,"outcome":"NO_EFFECT","x":Infinity}',
                b'{"rows":[],"established":false,"outcome":"NO_EFFECT","x":1e999}', b'not JSON',b'\xff'):
        (tmp_path/'answer.json').write_bytes(raw);r=rc.verify(normalized(),{'status':'PASS'},tmp_path)
        assert r['status']=='REJECTED' and any(x['reason']=='ARTIFACT_JSON' for x in r['failures'])


def test_forged_prenormalized_contract_cannot_bypass_verifier(tmp_path):
    answer(tmp_path);n=normalized();n['required_artifacts'][0].pop('json_checks')
    r=rc.verify(n,{'status':'PASS'},tmp_path)
    assert r['status']=='REJECTED' and r['failures'][0]['reason']=='SCIENTIFIC_CONTRACT_POLICY'
    n=normalized();n.pop('declared_outcomes')
    assert rc.verify(n,{'status':'PASS'},tmp_path)['status']=='REJECTED'


def test_stored_contract_rechecked_and_question_binding_enforced(tmp_path,monkeypatch):
    n=normalized();rec={'result_contract':n,'job':{'question':{'outcomes':n['declared_outcomes']}}}
    monkeypatch.setattr(sub,'capture_record',lambda _:rec)
    assert rc.contract_for(tmp_path)==n
    rec['job']['question']['outcomes']=['OTHER']
    with pytest.raises(sub.SubmissionError,match='SCIENTIFIC_OUTCOME_BINDING'):rc.contract_for(tmp_path)
    n['required_artifacts'][0].pop('json_checks')
    with pytest.raises(sub.SubmissionError,match='SCIENTIFIC_CONTENT_CHECK_REQUIRED'):rc.contract_for(tmp_path)


def test_restore_preserves_strong_contract_and_reapplies_policy(tmp_path):
    from tests.test_decoder06_capture import _spec
    spec=_spec(tmp_path);spec['output_contract']=contract();spec['question']['outcomes']=['PROCESS_COMPLETED','PROCESS_FAILED','NO_EFFECT','NOT_ESTABLISHED']
    captured=sub.capture(tmp_path/'store',spec);w=Path(captured['workspace']);raw=(w/'CAPTURE.json').read_bytes()
    restored=sub.restore_capture(w/'CAPTURE.json',tmp_path/'store/objects',tmp_path/'restored')
    r=Path(restored['workspace']);assert (r/'CAPTURE.json').read_bytes()==raw
    assert rc.contract_for(r)==rc.contract_for(w)
    out=tmp_path/'results';out.mkdir();answer(out)
    assert rc.verify(rc.contract_for(r),{'status':'PASS'},out)['status']=='VERIFIED'
    assert not (r/'runtime/attempts').exists()


def test_capture_refuses_weak_science_before_attempt(tmp_path):
    from tests.test_decoder06_capture import _spec
    spec=_spec(tmp_path);c=contract();c['required_artifacts'][0].pop('json_checks');spec['output_contract']=c;spec['question']['outcomes']=['NO_EFFECT']
    with pytest.raises(sub.SubmissionError,match='SCIENTIFIC_CONTENT_CHECK_REQUIRED'):sub.capture(tmp_path/'store',spec)
    assert not list(tmp_path.rglob('attempts'))


def test_controller_seals_rejected_values_without_scientific_outcome(tmp_path,monkeypatch):
    from infinity_grid import completion_evidence as ce, preservation as pr
    n=normalized();monkeypatch.setattr(sub,'capture_record',lambda _: {'result_contract':n,'job':{'question':{'outcomes':n['declared_outcomes']}}})
    monkeypatch.setattr(pr,'check_budget',lambda *a,**kw:None)
    working=tmp_path/'runtime/runs/unit';working.mkdir(parents=True);answer(working,established=0)
    sealed,r=ce.prepare_evidence({'workspace':tmp_path},working,{'status':'PASS'})
    assert r['status']=='REJECTED' and r['scientific_outcome'] is None
    assert json.loads((sealed/'RESULT_VERIFICATION.json').read_text())==r
    assert (sealed/'answer.json').read_bytes()==(working/'answer.json').read_bytes()


def test_validation_and_execution_only_keep_v1_report_identity(tmp_path):
    for kind in ('VALIDATION','SCRIPT'):
        c=rc.normalize({'description':'legacy fixture'}, {'kind':kind},{})
        r=rc.verify(c,{'status':'PASS'},tmp_path)
        assert r=={'schema_id':'IG_DECODER_RESULT_VERIFICATION_V1','contract_sha256':canonical_sha256(c),'claim':'VALIDATION' if kind=='VALIDATION' else 'EXECUTION_ONLY','status':'VERIFIED' if kind=='VALIDATION' else 'NOT_DECLARED','scientific_outcome':None,'artifacts':[],'failures':[]}


def test_malformed_content_declarations_refuse():
    for value in (None,'answer.json',['answer.json','answer.json'],['absent'],[{}]):
        c=contract();c['scientific_content']=value
        with pytest.raises(sub.SubmissionError):normalized(c)
    c=contract();c['required_artifacts'][0]['json_checks']=[{'pointer':'rows','equals':[]}]
    with pytest.raises(sub.SubmissionError,match='RESULT_JSON_POINTER'):normalized(c)
