import json,zipfile,hashlib
from pathlib import Path
import pytest
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_oracle_witnesses import verify_oracle_negative_witnesses,OracleWitnessError


def fixture():
    root=Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root/'tests/fixtures/dev79_index_reproduction/P9_CAPTURED_STATE_OBJECT_V1.zip') as z:
        raw=next(z.read(n) for n in z.namelist() if '/records/' in n and n.endswith('.json') and json.loads(z.read(n)).get('record_id')=='IGRD/L2J3/NEGATIVE/F')
    r=json.loads(raw);files={w['file']:(root/'tests/fixtures/storage_witnesses'/Path(w['file']).name).read_bytes() for w in r['payload']['witnesses']}
    return {'record_raw':raw,'oracle_raw':(root/'infinity_grid/resources/oracle/V026_SCIENTIFIC_ORACLE_MANIFEST.json').read_bytes(),'files':files}


def reseal(s,change):
    r=json.loads(s['record_raw']);o=json.loads(s['oracle_raw']);w=r['payload']['witnesses'][0];old=canonical_bytes(w);change(w)
    for i,a in enumerate(o['assertions']):
        if canonical_bytes(a)==old:o['assertions'][i]=w
    o.pop('oracle_sha256');o['oracle_sha256']=hashlib.sha256(canonical_bytes(o)).hexdigest();s['oracle_raw']=canonical_bytes(o)
    r['provenance']['source_hashes'][0]['sha256']=hashlib.sha256(s['oracle_raw']).hexdigest();r.pop('record_sha256');r['record_sha256']=hashlib.sha256(canonical_bytes(r)).hexdigest();s['record_raw']=canonical_bytes(r)


def test_actual_eight_l2j3_witnesses_verified():
    out=verify_oracle_negative_witnesses(**fixture());assert out['witnesses_checked']==8 and out['files_checked']==4
    assert out['status']=='ORACLE_NEGATIVE_WITNESSES_VERIFIED' and not out['execution_authorized'] and not out['dependency_closure_verified']
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_missing_witness_file_refused():
    s=fixture();s['files'].pop(next(iter(s['files'])))
    with pytest.raises(OracleWitnessError,match='INVENTORY_MISMATCH'):verify_oracle_negative_witnesses(**s)


def test_extra_witness_file_refused():
    s=fixture();s['files']['extra']=b'{}'
    with pytest.raises(OracleWitnessError,match='INVENTORY_MISMATCH'):verify_oracle_negative_witnesses(**s)


def test_changed_witness_source_bytes_refused():
    s=fixture();k=next(iter(s['files']));s['files'][k]+=b' '
    with pytest.raises(OracleWitnessError,match='PIN_MISMATCH'):verify_oracle_negative_witnesses(**s)


def test_changed_oracle_raw_binding_refused():
    s=fixture();s['oracle_raw']+=b' '
    with pytest.raises(OracleWitnessError,match='RAW_BINDING_MISMATCH'):verify_oracle_negative_witnesses(**s)


def test_integer_cannot_match_boolean_observation():
    s=fixture();r=json.loads(s['record_raw']);w=next(w for w in r['payload']['witnesses'] if w['equals'] is True)
    # Keep all four file references while making first oracle assertion claim 1
    # for the same source/pointer whose observed value is true.
    reseal(s,lambda x:x.update(file=w['file'],pointer=w['pointer'],equals=1))
    with pytest.raises(OracleWitnessError,match='VALUE_MISMATCH'):verify_oracle_negative_witnesses(**s)


def test_missing_object_pointer_refused():
    s=fixture();reseal(s,lambda w:w.update(pointer='/absent'))
    with pytest.raises(OracleWitnessError,match='POINTER_MISSING'):verify_oracle_negative_witnesses(**s)


def test_invalid_pointer_escape_refused():
    s=fixture();reseal(s,lambda w:w.update(pointer='/bad~2escape'))
    with pytest.raises(OracleWitnessError,match='UNSUPPORTED_OBJECT_POINTER'):verify_oracle_negative_witnesses(**s)


def test_witness_absent_from_pinned_oracle_refused():
    s=fixture();r=json.loads(s['record_raw']);r['payload']['witnesses'][0]['equals']='altered';r.pop('record_sha256');r['record_sha256']=hashlib.sha256(canonical_bytes(r)).hexdigest();s['record_raw']=canonical_bytes(r)
    with pytest.raises(OracleWitnessError,match='NOT_IN_ORACLE'):verify_oracle_negative_witnesses(**s)


def test_witness_budget_and_original_input_preservation():
    import copy
    s=fixture();before=copy.deepcopy(s);out=verify_oracle_negative_witnesses(**s);assert s==before
    with pytest.raises(OracleWitnessError,match='BYTE_BUDGET'):verify_oracle_negative_witnesses(**s,max_total_bytes=out['bytes_checked']-1)
