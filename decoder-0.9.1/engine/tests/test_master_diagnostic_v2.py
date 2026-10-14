"""New engineering checks; synthetic states carry no scientific authority."""
from copy import deepcopy
import ast
import hashlib
import inspect
from itertools import permutations, product
from pathlib import Path
from types import SimpleNamespace

import pytest
from infinity_grid import regime_scanner as rs
from infinity_grid import materialized_discovery as md
from infinity_grid import maturation_parallel as mp
from infinity_grid import regime_raise_audit as ra
from infinity_grid import semantic_sentinel as ss
from infinity_grid.theorem_registry import verify_all_theorems, StaleTheoremError


def state(caps=None, edges=None):
    caps = caps or [(2, 3, 0, 0, 0, 0, 0), (5, 7, 0, 0, 0, 0, 0)]
    edges = [(0, 1, 0, 0)] if edges is None else edges
    return SimpleNamespace(owner_caps=list(caps), typed_edges=list(edges),
        top_pairs=[(u,v) for u,v,a,b in edges], construction_digest='synthetic',
        skin='synthetic', motif_id='P2', lane='ENGINEERING',
        total_caps=tuple(map(sum,zip(*caps))),leaf_count=2,relation_count_total=len(edges))


def test_physical_actions_retained_while_diagnostic_orientation_folds():
    s=state(); before=deepcopy(vars(s))
    a=rs._action_aggregate_v2(s,list(product(range(2),repeat=2)))
    assert a['legal_action_labels']==4 and a['total_action_copies']==60
    assert a['type_pair_support']==3 and a['owner_pair_support']==1
    assert a['service_classes']==3 and a['resource_weighted_service_classes']==4
    assert a['action_orbits']==3 and a['automorphism_size']==2 and a['owner_orbits']==1
    assert a['schema_id']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2 and vars(s)==before


def test_asymmetric_duplicate_or_malformed_bridge_refuses_before_graph(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('graph reached before validation')
    monkeypatch.setattr(rs,'_graph_basic',forbidden)
    for pairs in [[(0,1)],[(0,0),(0,0)],[(True,True)],[(0,'0')],[(7,7)],[(0,)],None]:
        with pytest.raises(ValueError,match='DIAGNOSTIC_V2_'):
            rs._action_aggregate_v2(state(),pairs)


def test_capacity_and_topology_contract_preserves_multiplicity(monkeypatch):
    cases=[]
    for row in [(1,)*6,(True,)*7,('1',)*7,(-1,)*7]:
        s=state();s.owner_caps=[row,(1,)*7];cases.append(s)
    for edges,top in [([(0,1,0,0)],[]), ([(0,1,0,0)]*2,[(0,1)]),
                      ([(1,0,0,0)],[(0,1)]), ([(0,0,0,0)],[(0,0)]),
                      ([(0,1,7,0)],[(0,1)]), ([(0,True,0,0)],[(0,1)])]:
        s=state();s.typed_edges=edges;s.top_pairs=top;cases.append(s)
    for s in cases:
        with pytest.raises(ValueError,match='DIAGNOSTIC_V2_'):
            rs._action_aggregate_v2(s,[(0,0)])
    s=state(edges=[(0,1,0,0)]*6)
    a=rs._action_aggregate_v2(s,[(0,0)])
    assert a['total_action_copies']==10 and len(s.top_pairs)==6


def test_empty_bridge_and_single_owner_are_valid():
    a=rs._action_aggregate_v2(state(caps=[(0,)*7],edges=[]),[])
    assert a['legal_action_labels']==a['action_orbits']==a['total_action_copies']==0
    assert a['automorphism_size']==a['owner_orbits']==1


def test_new_organization_excludes_pure_magnitude_growth():
    s=state();pairs=list(product(range(2),repeat=2))
    a=rs._action_aggregate_v2(s,pairs);org=rs._state_organizational_signature_v2(s,a)
    t=deepcopy(s);t.owner_caps=[tuple(x*11 for x in row) for row in s.owner_caps]
    b=rs._action_aggregate_v2(t,pairs)
    assert b['total_action_copies']==a['total_action_copies']*121
    assert rs._state_organizational_signature_v2(t,b)==org
    with pytest.raises(ValueError,match='SCHEMA_REQUIRED'):
        rs._state_organizational_signature_v2(s,rs._action_aggregate(s,pairs))


def test_current_producers_route_v2_and_tag_results(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('legacy aggregate producer')
    monkeypatch.setattr(rs,'_action_aggregate',forbidden)
    s=state();pairs=list(product(range(2),repeat=2))
    scan,_=rs._scan_level([s],4,pairs,'synthetic-grammar')
    assert scan['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2
    assert scan['normalized_signature']['schema_id']==rs.ORGANIZATIONAL_SCHEMA_V2
    assert scan['obstruction_relief']['all_bridge_types_supported_everywhere']
    assert scan['quotient_observer']['resource_future_equivalence_basis']=='DIAGNOSTICS_ONLY_NO_FUTURE_EQUIVALENCE_AUTHORITY'
    assert rs._backbone_signature([s],pairs)['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2
    assert md._state_probe(s,pairs)['action']['schema_id']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2
    assert mp._selected_state_worker({'state':mp.state_wire(s),'bridge_pairs':pairs})['action']['schema_id']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2
    assert ra._recipe_signature(s,pairs)['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2


def test_mixed_observers_cannot_create_genealogy():
    s=state();scan,_=rs._scan_level([s],4,[(0,0)],'synthetic-grammar')
    assert all(v=='PERSISTS' for v in rs._classify_genealogy(scan,scan).values())
    legacy=deepcopy(scan);legacy.pop('diagnostic_schema')
    for a,b in [(legacy,scan),(scan,legacy),(None,dict(scan,diagnostic_schema='UNKNOWN'))]:
        with pytest.raises(ValueError,match='SCHEMA_COMPARISON_REFUSED'):
            rs._classify_genealogy(a,b)


def test_truthful_registry_keeps_diagnostics_outside_q_action_role():
    registry=ss.action_registry();rows={x['action_id']:x for x in registry['actions']}
    legacy=rows['RESOURCE_SCANNER_RELATION_ADD_AGGREGATE_V1']
    assert legacy['features']['hidden_topology_read'] and legacy['features']['orientation_order_read']
    assert set(legacy['hidden_reads'])=={'top_pairs','typed_edges'}
    for key in ['RESOURCE_SCANNER_RELATION_ADD_AGGREGATE_V1','RESOURCE_SCANNER_ACTION_DIAGNOSTIC_V2','RESOURCE_SCANNER_ORGANIZATIONAL_SIGNATURE_V2']:
        assert rows[key]['scientific_role']!='RELATION_ADD_SEMANTICS'
        assert rows[key]['successor_semantics']=='NONE_DIAGNOSTICS_ONLY'
        assert not set(rows[key]['writes']) & {'visible_p_f_reservation','hidden_outer_incidence'}
    assert [x['action_id'] for x in registry['actions'] if x['scientific_role']=='RELATION_ADD_SEMANTICS']==['EXACT_PROJECTED_RELATION_ADD_PROFILE_V1']


def test_v1_callable_and_authority_bytes_remain_frozen():
    assert ss.callable_semantic_sha256('infinity_grid.regime_scanner:_action_aggregate')=='3ebc339bf8bd0ad0db6febbe707ab664c4a34fcf7b772849dc84d2360ac46044'
    root=Path(rs.__file__).parent/'resources/decoder'
    expected={
        'O_NATIVE_SEMANTICS_BASELINE_V1.json':'7c7f211e34d54d539b9d04edf426765bd6049814b415a0b91ef7ff4edca59adf',
        'THEOREM_PREMISE_REGISTRY_V1.json':'6d6d744fad706a06aa8f3f615a8641a73a05cdf5b857efeb09135fd1cdd94dd8',
    }
    for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest


def test_all_semantic_and_theorem_guards_still_refuse():
    assert ss.verify_native_semantics(raise_on_change=False)['status']=='REOPEN_REQUIRED'
    for call in [ss.verify_native_semantics,ss.current_action_contract]:
        with pytest.raises(ss.NativeSemanticsReopenRequired):call()
    status=verify_all_theorems(raise_on_stale=False)
    assert status['status']=='STALE_REOPEN_REQUIRED' and len(status['stale_theorems'])==4
    with pytest.raises(StaleTheoremError):verify_all_theorems()
