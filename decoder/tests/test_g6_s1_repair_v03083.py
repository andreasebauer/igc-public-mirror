from __future__ import annotations
import json
from importlib.resources import files

from infinity_grid.canon import canonical_sha256
from infinity_grid.g6_s1_repair import _basis, _rho01, _typed_owner_orbits


def test_repair_question_hash_and_ladder():
    q=json.loads(files('infinity_grid').joinpath('resources/g6/G6_S1_REPAIR_PREREGISTRATION_V1.json').read_text())
    assert q['question_sha256']==canonical_sha256({k:v for k,v in q.items() if k!='question_sha256'})
    assert [x['candidate_id'] for x in q['candidate_ladder']]==['C1_RHO01','C2_TYPED_OWNER_ORBITS','C3_ROOTED_OWNER_RESPONSE_BAG']
    assert 'NO_AUTOMATIC_RESUME_OF_OLD_S2' in q['nonclaims']


def test_embedded_authority_commit_integrity():
    q=json.loads(files('infinity_grid').joinpath('resources/g6/G6_S1_REPAIR_PREREGISTRATION_V1.json').read_text())
    for sid in ('G6:S0','G6:S1'):
        name=sid.replace(':','__')+'.json'
        d=json.loads(files('infinity_grid').joinpath('resources/g6/authority').joinpath(name).read_text())
        assert d['commit_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='commit_sha256'})
        assert d['commit_sha256']==q['authority'][sid]['commit_sha256']


def test_frozen_basis_rho01_matches_review():
    assert {k:_rho01(t) for k,t in _basis().items()}=={'D2_BROOM':12,'D2_PATH':9,'D4_BROOM':18,'D4_PATH':12}


def test_typed_owner_orbit_structural_interpretation_matches_review():
    vals={k:_typed_owner_orbits(t) for k,t in _basis().items()}
    assert vals['D2_PATH']==(3,3,3,3,3,3,3)
    assert vals['D2_BROOM']==(4,4,4,4,4,4,4)
    assert vals['D4_PATH']==(4,4,4,4,4,4,4)
    assert vals['D4_BROOM']==(6,6,6,6,6,6,6)
