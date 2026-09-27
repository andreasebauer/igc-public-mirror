"""Execute only through registered Decoder VALIDATION; fixed bounded scope."""
import importlib
import itertools
import json
import random
import runpy
import shutil
import sys
from pathlib import Path
import pytest
from infinity_grid import relation_library_backend as lib

REFERENCE = Path(__file__).parent / 'reference_two_sorted'


def relation(np, nd, rows):
    return lib.Relation(np, nd, frozenset(rows))


def oracle(monkeypatch):
    monkeypatch.syspath_prepend(str(REFERENCE))
    return importlib.import_module('ig_two_sorted_tree_wiring')


def to_ref(ref, r):
    return ref.LabeledRelation.from_rows(tuple(range(r.n_ports)), tuple(range(r.n_destinations)), [ref.Row(*x) for x in r.rows])


def from_ref(r):
    return relation(len(r.port_slots), len(r.destination_slots), [x.serial() for x in r.rows])


def test_bounded_exhaustive_oracle_and_witnesses(monkeypatch):
    ref = oracle(monkeypatch)
    rng = random.Random(20260927)
    checks = 0
    for _ in range(24):
        np, nd = rng.randrange(4), rng.randrange(4)
        rows = [(rng.randrange(3), tuple((rng.randrange(3),rng.randrange(3)) for _ in range(np)), rng.randrange(3), tuple(rng.randrange(3) for _ in range(nd)), bool(rng.randrange(2))) for _ in range(rng.randrange(5))]
        a = relation(np,nd,rows)
        for p,d in itertools.product(itertools.permutations(range(np)),itertools.permutations(range(nd))):
            b=lib.transport(a,p,d)
            witness=lib.isomorphism(a,b)
            assert witness is not None
            assert lib.transport(a,**witness)==b
            assert to_ref(ref,a).isomorphic_to(to_ref(ref,b))
            checks+=1
        if rows:
            x=rows[0]; altered=(x[0]+100,x[1],x[2],x[3],x[4])
            b=relation(np,nd,[altered]+rows[1:])
            assert (lib.isomorphism(a,b) is not None)==to_ref(ref,a).isomorphic_to(to_ref(ref,b))
    print('BOUNDED_ORACLE_POSITIVE_MAPS',checks)


@pytest.mark.parametrize('sort',['port','destination'])
def test_equal_cardinality_correlation_control(sort):
    a,b=(1,0),(2,0)
    if sort=='port':
        x=relation(2,0,[(0,(a,b),1,(),True),(1,(a,b),1,(),True)])
        y=relation(2,0,[(0,(a,b),1,(),True),(1,(b,a),1,(),True)])
    else:
        x=relation(0,2,[(0,(),1,(5,9),True),(1,(),1,(5,9),True)])
        y=relation(0,2,[(0,(),1,(5,9),True),(1,(),1,(9,5),True)])
    weak=lambda r:{(a,tuple(sorted(p)),s,tuple(sorted(d)),t) for a,p,s,d,t in r.rows}
    assert len(x.rows)==len(y.rows)==2 and weak(x)==weak(y)
    assert lib.isomorphism(x,y) is None


@pytest.mark.parametrize('field',range(5))
def test_fixed_values_and_common_color_vocabulary(field):
    row=(1,((1,2),),2,(5,),True)
    alternatives=[2,((1,3),),3,(6,),False]
    changed=list(row);changed[field]=alternatives[field]
    assert lib.isomorphism(relation(1,1,[row]),relation(1,1,[tuple(changed)])) is None


def test_empty_arity_and_duplicate_set_semantics():
    assert lib.isomorphism(relation(0,0,[]),relation(0,0,[]))=={'ports':[],'destinations':[]}
    assert lib.isomorphism(relation(2,1,[]),relation(1,2,[])) is None
    assert lib.isomorphism(relation(2,1,[]),relation(2,1,[])) is not None
    r=(0,(),0,(),False)
    assert relation(0,0,[r,r])==relation(0,0,[r])
    assert lib.isomorphism(relation(0,0,[]),relation(0,0,[r])) is None


def test_composition_transport_and_negative_controls():
    a=relation(2,2,[(1,((1,2),(4,0)),2,(5,9),True),(2,((1,2),(8,0)),1,(9,5),False)])
    b=relation(1,1,[(4,((2,1),),2,(11,),True)])
    out=lib.compose(a,b,0,0)
    assert out.rows==frozenset({(5,((4,0),),2,(5,9,11),True),(6,((8,0),),1,(9,5,11),False)})
    moved=lib.transport(a,(1,0),(1,0))
    assert lib.compose(moved,b,1,0)==lib.transport(out,(0,),(1,0,2))
    assert lib.compose(moved,b,0,0)!=out
    blocked=relation(1,0,[(0,((4,1),),2,(),True)])
    assert lib.compose(a,blocked,0,0)==relation(1,2,[])


def test_invalid_inputs_budgets_and_library_failure(monkeypatch):
    for args in [(9,0,[]),(True,0,[]),(0,-1,[]),(1,0,[(0,(),0,(),True)]),(0,0,[(0,(),0,(),1)])]:
        with pytest.raises(ValueError):relation(*args)
    r=relation(1,0,[(0,((0,0),),0,(),True)])
    with pytest.raises(ValueError):lib.transport(r,[True],[])
    with pytest.raises(ValueError):lib.compose(r,r,-1,0)
    with pytest.raises(ValueError):lib.compose(relation(8,0,[]),relation(8,0,[]),0,0)
    monkeypatch.setattr(lib,'MAX_PAIRS',0)
    with pytest.raises(ValueError):lib.compose(r,r,0,0)
    monkeypatch.setitem(sys.modules,'igraph',None)
    with pytest.raises(ModuleNotFoundError):lib.isomorphism(r,r)


def test_false_library_witness_is_rejected(monkeypatch):
    import igraph
    class FakeGraph:
        def __init__(self,**kw):pass
        def isomorphic_bliss(self,*args,**kw):return True,[0],None
    monkeypatch.setattr(igraph,'Graph',FakeGraph)
    r=relation(1,0,[(0,((0,0),),0,(),True)])
    with pytest.raises(RuntimeError,match='witness replay'):lib.isomorphism(r,r)


def test_archived_45_vectors_with_current_adapter_and_exact_composition(monkeypatch,tmp_path):
    # Archived source bytes unchanged. Adapted run, not a claim of unchanged replay.
    ref=oracle(monkeypatch)
    original_compose=ref.compose_relations
    counts={'composition_comparisons':0,'library_isomorphism_calls':0}
    def composed(left,lp,right,rp):
        expected=original_compose(left,lp,right,rp)
        got=lib.compose(from_ref(left),from_ref(right),left.port_index(lp),right.port_index(rp))
        assert got==from_ref(expected)
        counts['composition_comparisons']+=1
        return ref.LabeledRelation.from_rows(expected.port_slots,expected.destination_slots,[ref.Row(*r) for r in got.rows])
    def iso(left,right):
        counts['library_isomorphism_calls']+=1
        return lib.isomorphism(from_ref(left),from_ref(right)) is not None
    monkeypatch.setattr(ref,'compose_relations',composed)
    monkeypatch.setattr(ref.LabeledRelation,'isomorphic_to',iso)
    code=tmp_path/'04_CODE';code.mkdir();(tmp_path/'05_RESULTS').mkdir()
    src=code/'test_t_naturality_and_embedding.py';shutil.copyfile(REFERENCE/src.name,src)
    result=runpy.run_path(str(src))['result']
    assert result['summary']['tests_passed']==45
    assert counts['composition_comparisons']>=180
    assert counts['library_isomorphism_calls']>=120
    print('ADAPTED_ARCHIVED_45',json.dumps(counts,sort_keys=True))


def test_archived_cospan_three_families(monkeypatch,tmp_path):
    oracle(monkeypatch)
    code=tmp_path/'04_CODE';code.mkdir();(tmp_path/'05_RESULTS').mkdir()
    src=code/'test_cospan_embedding.py';shutil.copyfile(REFERENCE/src.name,src)
    runpy.run_path(str(src))
    result=json.loads((tmp_path/'05_RESULTS/TREEMATCH_COSPAN_EMBEDDING_COMPAT_TESTS.json').read_text())
    assert result['summary']['tests_passed']==3
