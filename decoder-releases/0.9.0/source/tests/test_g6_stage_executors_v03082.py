from __future__ import annotations
import json
import importlib.resources as ir
import pytest

from infinity_grid.canon import canonical_sha256
from infinity_grid.g6_stage_executors import QUESTION_FILES, _basis, _join_exact_relation, _public_key_json, register_g6_chain_executors
from infinity_grid.adapters.g4_accepted import G4AcceptedAdapter
from infinity_grid.v05_chain import ScientificChainController
from infinity_grid.v05_execution_authority import ExecutionAuthorityError
from infinity_grid.v05_stage_architecture import audit_module_source

def _resource(name):
    return json.loads(ir.files('infinity_grid').joinpath('resources/g6').joinpath(name).read_text(encoding='utf-8'))

def test_all_frozen_g6_question_hashes_are_embedded_and_exact():
    for sid,name in QUESTION_FILES.items():
        obj=_resource(name); assert obj['stage_id']==sid
        assert obj['question_sha256']==canonical_sha256({k:v for k,v in obj.items() if k!='question_sha256'})

def test_historical_g6_private_executor_registration_is_inspect_only(tmp_path):
    cc=ScientificChainController(tmp_path, allow_legacy_new_registrations=True, engineering_only=True)
    with pytest.raises(ExecutionAuthorityError, match='LEGACY_EXECUTOR_DISABLED'):
        register_g6_chain_executors(cc)
    report=audit_module_source(ir.files('infinity_grid').joinpath('g6_stage_executors.py'))
    assert report['status']=='FAIL'

def test_g6_s0_frozen_authority_records_four_certified_g5_seed_carriers():
    d=_resource('authority/G6__S0.json')
    assert d['commit_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='commit_sha256'})
    x=d['result']; assert x['outcome']=='PASS_INTERFACE_AND_BASIS_FROZEN'
    assert x['basis_carrier_count']==4 and x['exact_basis_canon_count']==4 and x['public_basis_class_count']==2
    assert x['operator_count']==31 and x['public_exact_interface_mismatch_count']==0
    assert sorted(r['ref'] for r in x['basis_rows'])==['D2_BROOM','D2_PATH','D4_BROOM','D4_PATH']

def test_g6_s1_exact_relation_oracle_finds_residual_independently():
    ad=G4AcceptedAdapter(); c=_basis(); op=(0,0)
    assert _public_key_json(ad.public_read(c['D2_PATH']))==_public_key_json(ad.public_read(c['D2_BROOM']))
    _,a,_,_=_join_exact_relation(c['D2_PATH'],c['D2_PATH'],op)
    _,b,_,_=_join_exact_relation(c['D2_PATH'],c['D2_BROOM'],op)
    assert tuple(a)!=tuple(b); assert len(a)==6; assert len(b)==12

def test_historical_g6_s0_s1_chain_evidence_is_packaged_and_integrity_bound():
    s0=_resource('authority/G6__S0.json'); s1=_resource('authority/G6__S1.json')
    for d in (s0,s1): assert d['commit_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='commit_sha256'})
    assert s0['outcome']=='PASS_INTERFACE_AND_BASIS_FROZEN' and s1['outcome']=='SEPARATOR_FOUND'
    x=s1['result']; assert x['pair_context_row_count']==496 and x['public_pair_fibre_count']==124
    assert x['separator_fibre_count']==124 and x['structural_equality_used_for_decision'] is True
