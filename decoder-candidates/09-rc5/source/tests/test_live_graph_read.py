"""Bounded live edge-read engineering fixtures; no scientific execution."""
from pathlib import Path
import json
import pytest
from infinity_grid.paths import IGPaths
from infinity_grid.graph_store import GraphStore, GraphValidationError, build_node, _record_filename
from infinity_grid.live_graph_read import LiveGraphChanged, MAX_RECORD_BYTES
from infinity_grid.graph_replay import ReplayPlanner, computation_closure, evidence_closure, minimum_preservation_set
from infinity_grid.jumpstart import JumpstartPlanner, build_profile


def seed(tmp_path):
    g = GraphStore(IGPaths(tmp_path/'ig').ensure())
    nodes = [g.put_node(build_node(node_type='CLAIM', semantic_identity={'claim_id':x,'version':1},
             semantic_role='ENGINEERING_FIXTURE', retention_class='PINNED'))['node_id'] for x in 'ABCD']
    return g, nodes


def edge(g,a,b,kind='REQUIRES',mandatory=True):
    return g.add_edge(edge_type=kind,source_node_id=a,target_node_id=b,mandatory=mandatory)


def tree(root):
    return {str(p.relative_to(root)):(p.read_bytes(),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def reject(*args,**kwargs):
    raise AssertionError('LEGACY_GLOBAL_SCAN')


def test_adjacency_matches_legacy_order_and_filters(tmp_path):
    g,(a,b,c,d)=seed(tmp_path)
    edge(g,a,b);edge(g,a,c,'SUPPORTS');edge(g,d,a);edge(g,a,a,'QUALIFIES')
    rows=g.list_edges()
    with g.live_read() as v:
        assert v.list_edges()==rows
        assert v.outgoing(a)==g.outgoing(a)
        assert v.incoming(a)==g.incoming(a)
        assert v.outgoing(a,{'SUPPORTS'})==g.outgoing(a,{'SUPPORTS'})
        assert v.incoming(a,set())==[]
        assert v.outgoing('absent')==[]


def test_repeated_adjacency_uses_index_without_source_reads(tmp_path,monkeypatch):
    g,(a,b,c,d)=seed(tmp_path);edge(g,a,b)
    with g.live_read() as v:
        monkeypatch.setattr(g,'list_edges',reject)
        original=Path.open
        def guarded(path,*args,**kwargs):
            if path.parent==g.edges_dir:raise AssertionError('EDGE_REREAD')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',guarded)
        for _ in range(20):assert len(v.outgoing(a))==len(v.incoming(b))==1
        plans=[v._con.execute('EXPLAIN QUERY PLAN SELECT kind,raw FROM edges WHERE '+col+'=? ORDER BY k',(a,)).fetchall() for col in ['source','target']]
        assert all(any('USING INDEX live_' in row[3] for row in plan) for plan in plans)


def test_reads_preserve_workspace_bytes_and_times(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);edge(g,a,b);before=tree(g.paths.root)
    computation_closure(g,[a]);minimum_preservation_set(g,[a]);evidence_closure(g,a);ReplayPlanner(g).plan(a)
    assert tree(g.paths.root)==before


def test_new_operation_sees_addition_and_removal(tmp_path):
    g,(a,b,c,d)=seed(tmp_path)
    assert computation_closure(g,[a])['nodes']==[a]
    e=edge(g,a,b)
    assert computation_closure(g,[a])['nodes']==sorted([a,b])
    (g.edges_dir/_record_filename(e['edge_id'])).unlink()
    assert computation_closure(g,[a])['nodes']==[a]


def test_concurrent_addition_refuses_result_and_cleans_temp(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);v=g.live_read()
    with pytest.raises(LiveGraphChanged):
        with v:
            temp=Path(v._temp.name);assert v.outgoing(a)==[];edge(g,a,b)
    assert not temp.exists() and v._con is None


def test_concurrent_removal_refuses_result(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);e=edge(g,a,b)
    with pytest.raises(LiveGraphChanged):
        with g.live_read() as v:
            (g.edges_dir/_record_filename(e['edge_id'])).unlink()
            assert len(v.outgoing(a))==1


def test_concurrent_same_length_rewrite_refuses_result(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);e=edge(g,a,b);p=g.edges_dir/_record_filename(e['edge_id'])
    with pytest.raises(LiveGraphChanged):
        with g.live_read():
            raw=p.read_bytes();p.write_bytes(raw.replace(b'REQUIRES',b'SUPPORTS'))


def test_planner_does_not_return_when_edges_change(tmp_path,monkeypatch):
    g,(a,b,c,d)=seed(tmp_path);edge(g,a,b)
    get=g.get_node;changed=False
    def mutate(nid):
        nonlocal changed
        n=get(nid)
        if not changed:
            changed=True
            (g.edges_dir/'unexpected.json').write_text('{}')
        return n
    monkeypatch.setattr(g,'get_node',mutate)
    with pytest.raises(LiveGraphChanged):computation_closure(g,[a])


def test_count_and_total_byte_budgets_refuse_whole_view(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);edge(g,a,b);edge(g,a,c)
    with pytest.raises(GraphValidationError,match='COUNT_BUDGET'):
        with g.live_read(max_edges=1):pass
    v=g.live_read(max_bytes=1)
    with pytest.raises(GraphValidationError,match='BYTE_BUDGET'):
        with v:pass
    assert v._temp is None and v._con is None


def test_oversize_record_refused_before_decode(tmp_path):
    g,_=seed(tmp_path);(g.edges_dir/'large.json').write_bytes(b' '*(MAX_RECORD_BYTES+1))
    with pytest.raises(GraphValidationError,match='RECORD_BUDGET'):
        with g.live_read():pass


def test_invalid_json_cleans_failed_view(tmp_path):
    g,_=seed(tmp_path);(g.edges_dir/'bad.json').write_text('{broken');v=g.live_read()
    with pytest.raises(ValueError):
        with v:pass
    assert v._temp is None and v._con is None


def test_valid_edge_wrong_filename_refused(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);e=edge(g,a,b)
    (g.edges_dir/_record_filename(e['edge_id'])).rename(g.edges_dir/'wrong.json')
    with pytest.raises(GraphValidationError,match='PATH_IDENTITY'):
        with g.live_read():pass


def test_symlink_record_refused(tmp_path):
    g,(a,b,c,d)=seed(tmp_path);e=edge(g,a,b)
    (g.edges_dir/'link.json').symlink_to(g.edges_dir/_record_filename(e['edge_id']))
    with pytest.raises(GraphValidationError,match='UNSAFE_LIVE_EDGE_RECORD'):
        with g.live_read():pass


def test_closed_view_and_exception_cleanup(tmp_path):
    g,_=seed(tmp_path);v=g.live_read()
    with pytest.raises(RuntimeError,match='original'):
        with v:
            temp=Path(v._temp.name);raise RuntimeError('original')
    assert not temp.exists()
    with pytest.raises(GraphValidationError,match='CLOSED'):v.list_edges()
    with pytest.raises(AttributeError):v.put_edge


def test_computation_mandatory_cycles_and_evidence_separation(tmp_path,monkeypatch):
    g,(a,b,c,d)=seed(tmp_path);ab=edge(g,a,b);ba=edge(g,b,a);edge(g,a,c,mandatory=False);edge(g,a,d,'SUPPORTS')
    monkeypatch.setattr(g,'list_edges',reject)
    out=computation_closure(g,[a])
    assert out['status']=='PASS' and out['nodes']==sorted([a,b])
    assert out['edges']==sorted([ab['edge_id'],ba['edge_id']])


def test_evidence_both_directions_selfloops_and_type_separation(tmp_path,monkeypatch):
    g,(a,b,c,d)=seed(tmp_path);ab=edge(g,a,b,'SUPPORTS');cb=edge(g,c,b,'FALSIFIES');bb=edge(g,b,b,'QUALIFIES');edge(g,b,d)
    monkeypatch.setattr(g,'list_edges',reject)
    out=evidence_closure(g,b)
    assert out['status']=='PASS' and out['nodes']==sorted([a,b,c])
    assert out['edges']==sorted([ab['edge_id'],cb['edge_id'],bb['edge_id']])


def test_minimum_preservation_pinned_mandatory_closure(tmp_path,monkeypatch):
    g,(a,b,c,d)=seed(tmp_path);edge(g,a,b);edge(g,a,c,mandatory=False)
    monkeypatch.setattr(g,'list_edges',reject)
    out=minimum_preservation_set(g,[a])
    assert out['status']=='PASS' and out['required_roots']==sorted([a,b])


def test_replay_local_and_forced_missing_recipe(tmp_path,monkeypatch):
    g,(a,b,c,d)=seed(tmp_path);monkeypatch.setattr(g,'list_edges',reject)
    p=ReplayPlanner(g);local=p.plan(a);missing=p.plan(a,force_rebuild=True)
    assert local['status']=='PASS' and local['steps']==[{'action':'USE_LOCAL','node_id':a}]
    assert missing['status']=='UNRESOLVED' and missing['unresolved']==[{'node_id':a,'reason':'MISSING_IRREDUCIBLE_ROOT'}]


def jump_fixture(tmp_path):
    g,(a,b,c,d)=seed(tmp_path)
    record=g.artifacts.put_bytes(b'engineering jumpstart fixture')
    root=g.register_artifact_node(record)['node_id']
    edge(g,a,root);edge(g,a,b,mandatory=False);edge(g,a,c,'PRODUCED_BY');edge(g,a,d,'SUPPORTS')
    p=JumpstartPlanner(g.paths)
    p.profiles.register(build_profile(profile_id='engineering',target_node_id=a,
        materialization={'mode':'MATERIAL_ONLY','expected_material_root_count':1},
        launch={'argv':['{python}','fixture.py']},smoke={'argv':['{python}','fixture.py']}))
    return g,a,root,p


def test_jumpstart_indexes_only_mandatory_material_dependencies(tmp_path,monkeypatch):
    g,a,root,p=jump_fixture(tmp_path);monkeypatch.setattr(p.graph,'list_edges',reject)
    before=tree(g.paths.root);out=p.plan(a)
    assert out['status']=='PASS' and out['mandatory_nodes']==sorted([a,root])
    assert [r['node_id'] for r in out['material_roots']]==[root]
    assert tree(g.paths.root)==before


def test_jumpstart_next_plan_observes_new_missing_leaf(tmp_path):
    g,a,root,p=jump_fixture(tmp_path);assert p.plan(a)['status']=='PASS'
    b=next(n['node_id'] for n in g.list_nodes() if n.get('semantic_identity',{}).get('claim_id')=='B')
    edge(g,a,b)
    out=p.plan(a)
    assert out['status']=='UNRESOLVED'
    assert any(r['reason']=='LEAF_HAS_NO_MATERIAL_CONTENT' for r in out['unresolved'])
