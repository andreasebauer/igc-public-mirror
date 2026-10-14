"""Codec engineering fixtures; genuine lower-root qualification is separate."""
from copy import deepcopy
from types import SimpleNamespace
import pytest

from infinity_grid import regime_scanner as rs
from infinity_grid import maturation_parallel as codec
from infinity_grid import materialized_discovery as discovery
from infinity_grid.canon import canonical_sha256
from infinity_grid.maturation_dag_contract import validate_state_dag


class FixtureO7(rs.O7State):
    @property
    def construction_digest(self):
        return self.__dict__.get('forced_alias', canonical_sha256({
            'source': self.source_id, 'reserve': self.reserve_counts,
            'parents': [(p.exact_color, p.h6) for p in self.ctx.parents], 'edges': self.edges}))


class FixtureLift(rs.LiftState):
    @property
    def construction_digest(self):
        return self.__dict__.get('forced_alias', canonical_sha256({
            'level': self.level, 'children': [c.construction_digest for c in self.children],
            'edges': self.top_edges_full, 'lane': self.lane, 'motif': self.motif_id}))


@pytest.fixture
def types(monkeypatch):
    monkeypatch.setattr(rs, 'O7State', FixtureO7)
    monkeypatch.setattr(rs, 'LiftState', FixtureLift)
    return FixtureO7, FixtureLift


def leaf(count=1, source='AUTHORITY_A', color='COLOR_A'):
    ctx = SimpleNamespace(n=1, parents=(SimpleNamespace(exact_color=color, h6=((1, 2),)),),
                          source_order=('PARENT_' + color,))
    return FixtureO7(None, ctx, (), (count, 0, 0, 0, 0, 0, 0), source)


def fixture_graph():
    a=leaf();b=leaf(source='AUTHORITY_B',color='COLOR_B')
    x=FixtureLift(None,8,(a,b),((0,1,0,0,(0,0),(0,0)),),'BOUNDED','EDGE')
    return [x,x], [a,b]


def seal(obj):
    obj['science_sha256']=canonical_sha256({k:v for k,v in obj.items() if k!='science_sha256'})
    return obj


def test_valid_v2_bytes_and_occurrences_remain_exact(types):
    roots,bases=fixture_graph();obj=codec.state_dag_wire(roots);x=roots[0]
    expected={'schema_id':'IG_MATURATION_STATE_DAG_V2','execution_policy':'NATIVE_V3',
        'roots':[x.construction_digest,x.construction_digest], 'nodes':{}}
    for s in bases:
        expected['nodes'][s.construction_digest]={'kind':'O7','construction_digest':s.construction_digest,
            'source_id':s.source_id,'reserve_counts':list(s.reserve_counts),'lane':s.lane,
            'execution_policy':'NATIVE_V3','execution_order_key':s.construction_digest}
    expected['nodes'][x.construction_digest]={'kind':'LIFT','construction_digest':x.construction_digest,
        'level':8,'children':[s.construction_digest for s in bases],
        'top_edges_full':[[0,1,0,0,[0,0],[0,0]]],'lane':'BOUNDED','motif_id':'EDGE',
        'execution_policy':'NATIVE_V3','execution_order_key':x.construction_digest}
    assert obj==seal(expected)


def test_duplicate_child_slots_survive(types):
    a=leaf(2);x=FixtureLift(None,8,(a,a),((0,1,0,0,(0,0),(0,1)),),'BOUNDED','DUP')
    obj=codec.state_dag_wire([x,x])
    assert len(obj['nodes'])==2 and obj['roots']==[x.construction_digest]*2
    assert obj['nodes'][x.construction_digest]['children']==[a.construction_digest]*2


def test_encoder_rejects_inexact_and_negative_counters_before_alias(types):
    for value in [True,1.0,-1]:
        with pytest.raises(ValueError,match='O7_RESERVATION_EXACT_INTEGER'):
            codec.state_dag_wire([leaf(value)])


def test_encoder_rejects_inexact_level_and_witness(types):
    for level in [True,8.0,7]:
        x=FixtureLift(None,level,(leaf(),),(),'BOUNDED','LEVEL')
        with pytest.raises(ValueError,match='LIFT_LEVEL_EXACT_INTEGER'):
            codec.state_dag_wire([x])
    for value in [True,0.0]:
        x=FixtureLift(None,8,(leaf(),leaf()),((0,1,0,0,(0,value),(0,0)),),'BOUNDED','WITNESS')
        with pytest.raises(ValueError):codec.state_dag_wire([x])


def test_large_integer_roundtrip_uses_no_scalar_coercion(types,monkeypatch):
    huge=(1<<200)+7;a=leaf(huge);obj=codec.state_dag_wire([a,a])
    monkeypatch.setattr(discovery,'_get_process_o7_runtime',lambda spec:{'engine':None,'authority_base_states':[leaf()]})
    _,roots=codec._states_from_dag(obj)
    assert roots[0] is roots[1] and roots[0].reserve_counts[0]==huge
    assert codec.state_dag_wire(roots)==obj


def test_reader_rejects_resealed_inexact_values_before_loading(types,monkeypatch):
    monkeypatch.setattr(discovery,'_get_process_o7_runtime',lambda spec:pytest.fail('authority loaded before rejection'))
    roots,_=fixture_graph();original=codec.state_dag_wire(roots)
    for value in [True,1.0,-1]:
        obj=deepcopy(original);row=next(r for r in obj['nodes'].values() if r['kind']=='O7');row['reserve_counts'][0]=value
        with pytest.raises(ValueError):codec._states_from_dag(seal(obj))
    for value in [True,8.0,7]:
        obj=deepcopy(original);obj['nodes'][obj['roots'][0]]['level']=value
        with pytest.raises(ValueError):codec._states_from_dag(seal(obj))


def test_cycle_and_unreachable_rows_rejected_before_loading(types,monkeypatch):
    monkeypatch.setattr(discovery,'_get_process_o7_runtime',lambda spec:pytest.fail('authority loaded before rejection'))
    roots,_=fixture_graph();original=codec.state_dag_wire(roots)
    obj=deepcopy(original);obj['nodes'][obj['roots'][0]]['children'][0]=obj['roots'][0]
    with pytest.raises(ValueError,match='CYCLE'):codec._states_from_dag(seal(obj))
    obj=deepcopy(original);obj['nodes']['UNREACHABLE']={**next(r for r in obj['nodes'].values() if r['kind']=='O7'),'construction_digest':'UNREACHABLE'}
    with pytest.raises(ValueError,match='UNREACHABLE_NODE'):codec._states_from_dag(seal(obj))


def test_missing_children_policy_and_unknown_fields_rejected(types):
    roots,_=fixture_graph();original=codec.state_dag_wire(roots)
    for label in ['child','policy','field','seal','version']:
        obj=deepcopy(original);row=obj['nodes'][obj['roots'][0]]
        if label=='child':row['children'][0]='missing'
        elif label=='policy':row['execution_policy']='HISTORICAL_COORDINATE_V1'
        elif label=='field':row['unknown_payload']='unconsumed'
        elif label=='version':obj['schema_id']='IG_MATURATION_STATE_DAG_V1'
        else:obj['science_sha256']='wrong'
        if label!='seal':seal(obj)
        with pytest.raises(ValueError):validate_state_dag(obj)


def test_bad_recursive_witness_and_child_level_rejected(types):
    roots,_=fixture_graph();original=codec.state_dag_wire(roots)
    for value in [[1,0],[0,1],[0,True],[]]:
        obj=deepcopy(original);obj['nodes'][obj['roots'][0]]['top_edges_full'][0][4]=value
        with pytest.raises(ValueError):validate_state_dag(seal(obj))
    obj=deepcopy(original);obj['nodes'][obj['roots'][0]]['level']=9
    with pytest.raises(ValueError,match='CHILD_LEVEL'):validate_state_dag(seal(obj))


def test_encoder_cycle_rejected_before_recursive_alias(types):
    a=leaf();x=FixtureLift(None,8,(a,),(),'BOUNDED','CYCLE');x.children=(x,)
    with pytest.raises(ValueError,match='CYCLE'):codec.state_dag_wire([x])


def test_forced_alias_collision_cannot_discard_full_rows(types):
    a=leaf();b=leaf(2);a.forced_alias=b.forced_alias='COLLISION'
    with pytest.raises(ValueError,match='ALIAS_COLLISION_REVIEW_REQUIRED'):codec.state_dag_wire([a,b])


def test_equal_alias_and_row_cannot_discard_different_coordinates(types):
    a=leaf();b=leaf(color='OTHER_COLOR');a.forced_alias=b.forced_alias='COLLISION'
    with pytest.raises(ValueError,match='ALIAS_COLLISION_REVIEW_REQUIRED'):codec.state_dag_wire([a,b])


def test_same_object_different_provenance_requires_review(types):
    a=leaf();b=leaf();b.lane='DIFFERENT_SOURCE_LANE'
    with pytest.raises(ValueError,match='ALIAS_COLLISION_REVIEW_REQUIRED'):codec.state_dag_wire([a,b])


def test_ambiguous_authority_source_cannot_select_last_match(types,monkeypatch):
    a=leaf();obj=codec.state_dag_wire([a]);other=leaf(color='OTHER_COLOR')
    monkeypatch.setattr(discovery,'_get_process_o7_runtime',lambda spec:{'engine':None,'authority_base_states':[a,other]})
    with pytest.raises(ValueError,match='SOURCE_ALIAS_AMBIGUOUS_REVIEW_REQUIRED'):codec._states_from_dag(obj)


def test_identical_authority_coordinates_and_duplicate_roots_reconstruct(types,monkeypatch):
    roots,bases=fixture_graph();obj=codec.state_dag_wire(roots)
    monkeypatch.setattr(discovery,'_get_process_o7_runtime',lambda spec:{'engine':None,'authority_base_states':[bases[0],leaf(),bases[1]]})
    _,restored=codec._states_from_dag(obj)
    assert restored[0] is restored[1] and codec.state_dag_wire(restored)==obj


def test_historical_policy_remains_explicit_and_separate(types,monkeypatch):
    a=leaf();a.execution_policy='HISTORICAL_COORDINATE_V1';obj=codec.state_dag_wire([a])
    assert obj['execution_policy']=='HISTORICAL_COORDINATE_V1'
    assert obj['nodes'][a.construction_digest]['execution_order_key']==rs.execution_order_key(a)
    monkeypatch.setattr(discovery,'_get_process_o7_runtime',lambda spec:{'engine':None,'authority_base_states':[leaf()]})
    _,roots=codec._states_from_dag(obj)
    assert roots[0].execution_policy=='HISTORICAL_COORDINATE_V1' and codec.state_dag_wire(roots)==obj
