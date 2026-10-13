import pytest
from infinity_grid.step3_l2_producer import evaluate,export_rows

def test_occurrences_keep_rooted_port_and_destination_binding():
    a=(1,((1,0),(2,0),(4,0)),2,7,1);b=(2,((2,0),(4,0),(8,0)),1,8,1);c=(4,((4,0),(8,0),(16,0)),2,9,0)
    result=evaluate({'a_events':[a],'b':b,'c_events':[c],'catalog_sha256':'a'*64})
    row=result['states'][0]
    assert row['state']['record']==(7,((2,0),(4,0),(8,0),(8,0),(16,0)),1,(7,8,9),0)
    exported=list(export_rows({'state':row['state'],'occurrence_count':1,'representative_occurrence_id':'B000:0'}))
    assert [p['slot'] for p in exported[0]['ports']]==['A1','A2','B2','C1','C2']
    assert [p['target_class'] for p in exported[0]['components']]==[7,8,9]
    assert exported[1]['object_id']==exported[0]['id']
    assert exported[0]['internal_bridges']==[['A0','B0'],['B1','C0']]

def test_bridge_filters_both_directions():
    a=(1,((1,2),(1,0),(1,0)),1,0,1);b=(1,((1,0),(1,0),(1,0)),1,0,1)
    assert evaluate({'a_events':[a],'b':b,'c_events':[b],'catalog_sha256':'a'*64})['states']==[]

def test_export_refuses_unenumerated_multiplicity():
    state={'record':[1,[[1,0]]*5,1,[0,0,0],1],'object_id':'x','components':['a','b','c']}
    with pytest.raises(ValueError,match='UNIQUE_CASE'):list(export_rows({'state':state,'occurrence_count':2,'representative_occurrence_id':'x'}))


def test_production_registry_admits_evaluator_with_semantic_gate():
    from infinity_grid.v05_stage_runtime import _resolve_ref
    from infinity_grid.v05_stage_registry import semantic_audit_for_ref
    ref="infinity_grid.step3_l2_producer:evaluate"
    assert _resolve_ref(ref) is evaluate
    assert semantic_audit_for_ref(ref)["status"]=="PASS"
