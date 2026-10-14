"""Engineering fixtures only; these objects carry no scientific authority."""
from copy import deepcopy
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import pytest
from infinity_grid import carrier_reuse as cr
from infinity_grid import maturation_parallel as mp
from infinity_grid.canon import canonical_sha256

IDENTITY=dict(source_sha256='1'*64,registry_sha256='2'*64,
    producer_experiment_id='ENGINEERING_FIXTURE_NO_SCIENTIFIC_AUTHORITY',
    authority_stage='ENGINEERING_ONLY',authority_science_sha256='3'*64)

@dataclass
class FixtureCarrier:
    construction_digest: str
    value: int
    execution_policy: str = 'NATIVE_V3'

def expected():return {'expected_'+k:v for k,v in IDENTITY.items()}
def write(root,manifest):
    manifest=deepcopy(manifest)
    manifest['artifact_sha256']=canonical_sha256({k:v for k,v in manifest.items() if k!='artifact_sha256'})
    (root/'CERTIFIED_CARRIER_SET_V1.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')

def fixture(tmp_path,monkeypatch):
    states=[FixtureCarrier('a'*64,2**200+17),FixtureCarrier('b'*64,23),FixtureCarrier('a'*64,2**200+17)]
    dag={'schema_id':'IG_MATURATION_STATE_DAG_V2','roots':[s.construction_digest for s in states],
         'nodes':{},'execution_policy':'NATIVE_V3'}
    dag['science_sha256']=canonical_sha256(dag)
    monkeypatch.setattr(mp,'state_dag_wire',lambda roots:deepcopy(dag))
    calls=[]
    def restore(obj):
        calls.append(obj)
        return None,states
    monkeypatch.setattr(mp,'_states_from_dag',restore)
    carriers={f'ROOT.{i:03d}':(s,{'value':s.value,'engineering_only':True}) for i,s in enumerate(states)}
    manifest=cr.save_certified_carrier_set(tmp_path,carriers=carriers,**IDENTITY)
    return manifest,dag,calls,carriers

def forbidden_load(monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('Authority load attempted before identity checks')
    monkeypatch.setattr(cr,'_load_exact_state_bundle',fail)

def test_valid_manifest_and_bundle_bytes_remain_unchanged(tmp_path,monkeypatch):
    manifest,dag,calls,carriers=fixture(tmp_path,monkeypatch)
    rows=[dict(carrier_id=cid,public_state=pub,public_state_sha256=canonical_sha256(pub),
        exact_carrier_payload='EXACT_STATE_DAG_V1.json',exact_carrier_root_digest=s.construction_digest,
        construction_digest=s.construction_digest,**IDENTITY,certification_status='CERTIFIED_PASS') for cid,(s,pub) in sorted(carriers.items())]
    original=dict(schema_id='CERTIFIED_CARRIER_SET_V1',exact_payload_schema='IG_MATURATION_STATE_DAG_V1',
        status='CERTIFIED_PASS',carrier_count=3,carriers=rows)
    original['artifact_sha256']=canonical_sha256(original)
    assert manifest==original
    assert (tmp_path/'CERTIFIED_CARRIER_SET_V1.json').read_bytes()==(json.dumps(original,sort_keys=True,indent=2)+'\n').encode()
    assert json.loads((tmp_path/'EXACT_STATE_DAG_V1.json').read_text())==dag

def test_duplicate_root_occurrences_preserve_all_carrier_ids(tmp_path,monkeypatch):
    manifest,dag,calls,_=fixture(tmp_path,monkeypatch)
    loaded,observed=cr.load_certified_carrier_set(tmp_path,**expected())
    assert len(loaded)==3 and observed==manifest and len(calls)==1
    assert loaded['ROOT.000'].value==loaded['ROOT.002'].value==2**200+17

def test_counts_refuse_exactly_before_loading(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for count in [True,False,3.0,'3',-1,0]:
        bad=deepcopy(manifest);bad['carrier_count']=count;write(tmp_path,bad)
        with pytest.raises(cr.CertifiedCarrierReuseError,match='EXACT_POSITIVE_COUNT'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_row_count_and_list_shape_refuse_before_loading(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for value in [manifest['carriers'][:-1],{},None]:
        bad=deepcopy(manifest);bad['carriers']=value;write(tmp_path,bad)
        with pytest.raises(cr.CertifiedCarrierReuseError,match='count mismatch'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_all_five_expected_identity_checks_precede_loading(tmp_path,monkeypatch):
    fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for key in IDENTITY:
        args=expected();args['expected_'+key]=('f'*64 if key.endswith('sha256') else 'DIFFERENT')
        with pytest.raises(cr.CertifiedCarrierReuseError,match='identity mismatch'):cr.load_certified_carrier_set(tmp_path,**args)

def test_exact_identity_types_and_digest_shapes(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for key,value in [('source_sha256',1),('registry_sha256','bad'),('authority_science_sha256','A'*64),('producer_experiment_id',True),('authority_stage','')]:
        bad=deepcopy(manifest);bad['carriers'][0][key]=value;write(tmp_path,bad)
        with pytest.raises(cr.CertifiedCarrierReuseError,match='EXACT_'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_duplicate_or_inexact_carrier_ids_refused(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for value in ['ROOT.000',True,0,'']:
        bad=deepcopy(manifest);bad['carriers'][1]['carrier_id']=value;write(tmp_path,bad)
        with pytest.raises(cr.CertifiedCarrierReuseError):cr.load_certified_carrier_set(tmp_path,**expected())

def test_manifest_and_row_unknown_fields_refused(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for target in ['manifest','row']:
        bad=deepcopy(manifest);(bad if target=='manifest' else bad['carriers'][0])['extra']='unconsumed';write(tmp_path,bad)
        with pytest.raises(cr.CertifiedCarrierReuseError,match='FIELDS'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_payload_paths_and_schema_refuse_before_loading(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    for value in ['../outside.json','/outside.json','OTHER.json',True]:
        bad=deepcopy(manifest);bad['carriers'][0]['exact_carrier_payload']=value;write(tmp_path,bad)
        with pytest.raises(cr.CertifiedCarrierReuseError,match='PAYLOAD_NAME'):cr.load_certified_carrier_set(tmp_path,**expected())
    bad=deepcopy(manifest);bad['exact_payload_schema']='UNKNOWN';write(tmp_path,bad)
    with pytest.raises(cr.CertifiedCarrierReuseError,match='unsupported'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_public_hash_and_root_digest_mismatch_refuse_early(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    bad=deepcopy(manifest);bad['carriers'][0]['public_state']['value']+=1;write(tmp_path,bad)
    with pytest.raises(cr.CertifiedCarrierReuseError,match='public state digest'):cr.load_certified_carrier_set(tmp_path,**expected())
    bad=deepcopy(manifest);bad['carriers'][0]['exact_carrier_root_digest']='f'*64;write(tmp_path,bad)
    with pytest.raises(cr.CertifiedCarrierReuseError,match='ROOT_CONSTRUCTION'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_manifest_seal_and_status_fail_before_loading(tmp_path,monkeypatch):
    manifest,_,_,_=fixture(tmp_path,monkeypatch);forbidden_load(monkeypatch)
    bad=deepcopy(manifest);bad['artifact_sha256']='f'*64
    (tmp_path/'CERTIFIED_CARRIER_SET_V1.json').write_text(json.dumps(bad))
    with pytest.raises(cr.CertifiedCarrierReuseError,match='manifest hash'):cr.load_certified_carrier_set(tmp_path,**expected())
    bad=deepcopy(manifest);bad['status']='REVIEW_REQUIRED';write(tmp_path,bad)
    with pytest.raises(cr.CertifiedCarrierReuseError,match='schema/status'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_ordered_root_closure_before_decoder_loading(tmp_path,monkeypatch):
    _,dag,calls,_=fixture(tmp_path,monkeypatch)
    for roots in [dag['roots'][:-1],dag['roots']+['f'*64],list(reversed(dag['roots'][:2]))+[dag['roots'][2]]]:
        bad=deepcopy(dag);bad['roots']=roots;(tmp_path/'EXACT_STATE_DAG_V1.json').write_text(json.dumps(bad))
        with pytest.raises(cr.CertifiedCarrierReuseError,match='ORDERED_ROOT_OCCURRENCE'):cr.load_certified_carrier_set(tmp_path,**expected())
    assert not calls

def test_writer_validates_before_creating_payload(tmp_path,monkeypatch):
    for carriers in [{},{True:(FixtureCarrier('a'*64,1),{})},{'x':(FixtureCarrier('bad',1),{})}]:
        with pytest.raises(cr.CertifiedCarrierReuseError):cr.save_certified_carrier_set(tmp_path,carriers=carriers,**IDENTITY)
    assert not list(tmp_path.iterdir())

def test_generic_pickle_fixture_roundtrip_and_id_closure(tmp_path):
    s=FixtureCarrier('a'*64,2**200+9)
    manifest=cr.save_certified_carrier_set(tmp_path,carriers={'a':(s,{'engineering_only':True})},**IDENTITY)
    assert manifest['exact_payload_schema']=='PYTHON_PICKLE_EXACT_STATE_V1'
    loaded,_=cr.load_certified_carrier_set(tmp_path,**expected());assert loaded['a']==s
    path=tmp_path/'EXACT_STATE_PICKLE_V1.pkl';path.write_bytes(pickle.dumps({'other':s}))
    with pytest.raises(cr.CertifiedCarrierReuseError,match='PICKLE_ID_CLOSURE'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_pickle_alias_collision_cannot_discard_distinct_states(tmp_path):
    carriers={'a':(FixtureCarrier('a'*64,1),{}),'b':(FixtureCarrier('a'*64,2),{})}
    cr.save_certified_carrier_set(tmp_path,carriers=carriers,**IDENTITY)
    with pytest.raises(cr.CertifiedCarrierReuseError,match='ALIAS_COLLISION_REVIEW_REQUIRED'):cr.load_certified_carrier_set(tmp_path,**expected())

def test_stable_outer_slot_does_not_relabel_inner_historical_dag(tmp_path,monkeypatch):
    _,dag,_,_=fixture(tmp_path,monkeypatch)
    monkeypatch.undo()
    historical=deepcopy(dag);historical['schema_id']='IG_MATURATION_STATE_DAG_V1'
    (tmp_path/'EXACT_STATE_DAG_V1.json').write_text(json.dumps(historical))
    with pytest.raises(cr.CertifiedCarrierReuseError,match='DAG reconstruction failed'):
        cr._load_exact_state_bundle(tmp_path,schema='IG_MATURATION_STATE_DAG_V1',payload_name='EXACT_STATE_DAG_V1.json')
