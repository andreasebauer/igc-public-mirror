from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import product
from pathlib import Path
import json
import random

import pytest

from infinity_grid.adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from infinity_grid.exact_tree_relation_kernel import (
    ExactTreeRelationKernel, ExactTreeRelationKernelError, PreparedG4RelationTree,
    configure_relation_kernel, get_relation_kernel, tree_from_record,
)
from infinity_grid.g6_stage_executors import _basis, _join_exact_relation
from infinity_grid.canon import canonical_sha256
import infinity_grid.exact_tree_relation_kernel as module


@pytest.fixture(scope='module')
def basis(): return _basis()


@pytest.fixture(scope='module')
def kernel(): return ExactTreeRelationKernel()


def fixture_data():
    p=Path(__file__).parents[1]/'infinity_grid/resources/engineering/EXACT_TREE_KERNEL_E1_FIXTURE.json'
    d=json.loads(p.read_text())
    assert d['fixture_sha256']==canonical_sha256({k:v for k,v in d.items() if k!='fixture_sha256'})
    return d


@pytest.mark.parametrize('left,right',list(product(('D2_PATH','D2_BROOM','D4_PATH','D4_BROOM'),repeat=2)))
def test_all_496_frozen_basis_relations_equal_historical(left,right,basis,kernel):
    for op in G4AcceptedAdapter().operator_basis():
        old_trees, old, legal, attempted = _join_exact_relation(basis[left],basis[right],op)
        new=kernel.relation(basis[left],basis[right],op)
        assert new.canons==tuple(old)
        assert new.legal_owner_pairs==legal
        assert new.attempted_owner_pairs==attempted
        assert new.child_canon_constructions==new.rooted_owner_pair_candidates<=legal


@pytest.mark.parametrize('index', [0, 17, 35, 61, 83, 101, 133, 159, 181, 207, 231, 255])
def test_portable_higher_panel_exact_signature_and_returned_children(index,kernel):
    d=fixture_data(); state=tree_from_record(d['states'][index]['tree']); probe=tree_from_record(d['probe_tree'])
    ad=G4AcceptedAdapter()
    for op,pos in [((0,0),'LEFT'),((0,1),'LEFT'),((1,0),'RIGHT')]:
        a,b=(state,probe) if pos=='LEFT' else (probe,state)
        _,old,legal,attempted=_join_exact_relation(a,b,op)
        new=kernel.relation(a,b,op)
        assert new.canons==tuple(old)
        assert new.legal_owner_pairs==legal
        assert new.attempted_owner_pairs==attempted
        for child,canon in zip(new.children,new.canons):
            assert ad.public_read(child)['legal']
            assert ad.unrooted_canon(child)==canon


def relabel(t,seed):
    rng=random.Random(seed); perm=list(range(t.n));rng.shuffle(perm)
    labels=['']*t.n
    for v,k in enumerate(t.H_classes): labels[perm[v]]=k
    pairs=[]
    for i,((u,v),(a,b)) in enumerate(zip(t.edges,t.edge_operators)):
        if i%2: u,v,a,b=v,u,b,a
        pairs.append(((perm[u],perm[v]),(a,b)))
    rng.shuffle(pairs)
    return DecoratedG4Tree(t.n,tuple(p[0] for p in pairs),tuple(labels),tuple(p[1] for p in pairs))


@pytest.mark.parametrize('seed',range(6))
def test_asymmetric_endpoints_relabel_and_reorder_exact(seed,kernel):
    t=DecoratedG4Tree(5,((0,1),(1,2),(2,3),(1,4)),('A','B','C','D','A'),((0,1),(2,4),(4,2),(1,0)))
    p=DecoratedG4Tree(3,((0,1),(1,2)),('D','B','C'),((1,0),(0,1)))
    tr,pr=relabel(t,seed),relabel(p,seed+100)
    assert kernel.prepare(t).unrooted_canon==kernel.prepare(tr).unrooted_canon
    for op in [(0,1),(1,0),(2,4),(4,2),(0,0)]:
        new=kernel.relation(tr,pr,op)
        assert new.canons==kernel.relation(t,p,op).canons
        assert new.canons==tuple(_join_exact_relation(tr,pr,op)[1])


def test_exactly_one_graft_canon_per_candidate_no_child_validation(monkeypatch,basis,kernel):
    left,right=kernel.prepare(basis['D4_PATH']),kernel.prepare(basis['D4_BROOM'])
    calls=[]; original=module._grafted_canon
    def counted(*args):
        calls.append(1);return original(*args)
    def forbidden(*args,**kwargs): raise AssertionError('child preparation/public validation repeated')
    monkeypatch.setattr(module,'_grafted_canon',counted)
    monkeypatch.setattr(module,'prepare_endpoint_decorated_tree',forbidden)
    monkeypatch.setattr(G4AcceptedAdapter,'public_read',forbidden)
    monkeypatch.setattr(G4AcceptedAdapter,'_validate',forbidden)
    new=kernel.relation(left,right,(0,0))
    assert len(calls)==new.child_canon_constructions==new.rooted_owner_pair_candidates
    assert 0<len(calls)<new.legal_owner_pairs


def test_cache_hits_bounds_eviction_and_disabled_cache(basis):
    k=ExactTreeRelationKernel(max_cache_entries=2,max_cache_bytes=1024*1024)
    a=k.prepare(basis['D2_PATH']); assert k.prepare(basis['D2_PATH']) is a
    k.prepare(basis['D2_BROOM']); k.prepare(basis['D4_PATH'])
    m=k.metrics(); assert m['preparation_hits']==1 and m['preparation_evictions']==1
    assert m['prepared_cache_entries']==2 and m['prepared_cache_accounted_bytes']<=1024*1024
    zero=ExactTreeRelationKernel(max_cache_bytes=0)
    assert zero.prepare(basis['D2_PATH']) is not zero.prepare(basis['D2_PATH'])
    tiny=ExactTreeRelationKernel(max_cache_bytes=1)
    tiny.prepare(basis['D4_PATH'])
    assert tiny.metrics()['prepared_cache_entries']==0
    assert tiny.metrics()['preparation_uncached_oversize']==1
    assert zero.relation(basis['D4_PATH'],basis['D2_PATH'],(0,0)).canons==k.relation(basis['D4_PATH'],basis['D2_PATH'],(0,0)).canons


def test_cache_byte_bound_causes_eviction(basis):
    p=ExactTreeRelationKernel().prepare(basis['D4_PATH'])
    size=module._accounted_bytes(p)
    k=ExactTreeRelationKernel(max_cache_entries=128,max_cache_bytes=size+100)
    k.prepare(basis['D4_PATH']); k.prepare(basis['D2_PATH'])
    assert k.metrics()['prepared_cache_accounted_bytes']<=size+100
    assert k.metrics()['preparation_evictions']>=1


def test_scope_reset_is_lazy_and_discards_old_cache(monkeypatch,basis):
    configure_relation_kernel(scope_identity='SOURCE_Q_A')
    first=get_relation_kernel(); first.prepare(basis['D2_PATH'])
    configure_relation_kernel(scope_identity='SOURCE_Q_B')
    assert module._DEFAULT_KERNEL is None
    second=get_relation_kernel()
    assert second is not first and second.scope_identity=='SOURCE_Q_B'
    assert second.metrics()['prepared_cache_entries']==0


def test_immutable_prepared_and_different_authority_rejected(basis,kernel):
    p=kernel.prepare(basis['D2_PATH'])
    with pytest.raises(FrozenInstanceError): p.legal=False
    with pytest.raises(ExactTreeRelationKernelError): PreparedG4RelationTree()
    with pytest.raises(ExactTreeRelationKernelError): replace(p,legal=False)
    ad=G4AcceptedAdapter()
    ad._rows={k:dict(row) for k,row in ad._rows.items()}
    ad._rows['C']['caps7']=list(ad._rows['C']['caps7']); ad._rows['C']['caps7'][0]+=1
    other=ExactTreeRelationKernel(adapter=ad)
    with pytest.raises(ExactTreeRelationKernelError,match='authority mismatch'):
        other.relation(p,basis['D2_PATH'],(0,0))


@pytest.mark.parametrize('record',[
 {'n':1.2,'edges':[],'H_classes':['C'],'edge_operators':[]},
 {'n':True,'edges':[],'H_classes':['C'],'edge_operators':[]},
 {'n':1,'edges':[],'H_classes':[1],'edge_operators':[]},
 {'n':2,'edges':[[0,1]],'H_classes':['C','C'],'edge_operators':[[0,1.2]]},
])
def test_record_decoder_rejects_lossy_coercion(record):
    with pytest.raises(ExactTreeRelationKernelError): tree_from_record(record)


@pytest.mark.parametrize('tree',[
 DecoratedG4Tree(0,(),(),()),
 DecoratedG4Tree(2,(),('C',),()),
 DecoratedG4Tree(2,((0,2),),('C','C'),((0,0),)),
 DecoratedG4Tree(2,((0,0),),('C','C'),((0,0),)),
 DecoratedG4Tree(4,((0,1),(1,2),(2,0)),('C',)*4,((0,0),)*3),
 DecoratedG4Tree(3,((0,1),(0,1)),('C',)*3,((0,0),)*2),
 DecoratedG4Tree(1,(),('UNKNOWN',),()),
 DecoratedG4Tree(2,((0,1),),('C','C'),((8,0),)),
 DecoratedG4Tree(2,((0,1),),('C','C'),()),
])
def test_invalid_carrier_fails_before_observation(tree,kernel):
    with pytest.raises((ValueError,ExactTreeRelationKernelError)):
        kernel.prepare(tree)


def test_resource_exhaustion_guards_engineering_tiny_authority():
    ad=G4AcceptedAdapter(); ad._rows={k:dict(v) for k,v in ad._rows.items()}
    for row in ad._rows.values(): row['caps7']=[1]*7
    k=ExactTreeRelationKernel(adapter=ad)
    single=DecoratedG4Tree(1,(),('C',),())
    saturated=DecoratedG4Tree(2,((0,1),),('C','C'),((0,0),))
    illegal=DecoratedG4Tree(3,((0,1),(0,2)),('C',)*3,((0,0),)*2)
    assert k.prepare(saturated).legal
    assert not k.relation(saturated,single,(0,0)).canons
    assert not k.prepare(illegal).legal
    assert not k.relation(illegal,single,(0,0)).canons
    assert len(k.relation(single,single,(0,0)).canons)==1


def test_singleton_and_repeated_equal_branches_preserved(kernel):
    single=DecoratedG4Tree(1,(),('C',),())
    star=DecoratedG4Tree(6,tuple((0,i) for i in range(1,6)),('C',)*6,((0,0),)*5)
    new=kernel.relation(star,single,(0,0))
    assert new.canons==tuple(_join_exact_relation(star,single,(0,0))[1])
    assert new.rooted_owner_pair_candidates==2
    assert new.legal_owner_pairs==6


def test_registered_engineering_evaluators_pass_architecture_gate():
    from infinity_grid.v05_stage_architecture import require_controller_only_callable
    from infinity_grid.engine_kernel_acceptance import historical_reference_evaluator,exact_equivalence_evaluator,engineering_stage_handler
    from infinity_grid.g6_controller_evaluators import exact_one_step_relation_evaluator
    for fn in (historical_reference_evaluator,exact_equivalence_evaluator,engineering_stage_handler,exact_one_step_relation_evaluator):
        assert require_controller_only_callable(fn,role='E1 engineering')['status']=='PASS'


def test_extracted_reference_matches_unchanged_historical_function(basis):
    from infinity_grid.engine_kernel_acceptance import historical_reference_evaluator
    from dataclasses import asdict
    for a,b in [('D2_PATH','D4_BROOM'),('D4_PATH','D2_BROOM')]:
        for op in [(0,0),(0,1),(2,4)]:
            payload={'state_tree':asdict(basis[a]),'probe_tree':asdict(basis[b]),'operator':op,'position':'LEFT'}
            result=historical_reference_evaluator(payload)
            _,c,l,n=_join_exact_relation(basis[a],basis[b],op)
            assert result['signature']==tuple(c)
            assert result['metrics']['reference_legal_owner_pairs']==l
            assert result['metrics']['reference_attempted_owner_pairs']==n
