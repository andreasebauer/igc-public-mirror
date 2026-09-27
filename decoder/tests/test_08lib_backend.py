"""Frozen focused tests; execute only via registered Decoder VALIDATION."""
import sys
import pytest
from infinity_grid import regime_scanner as scanner
from infinity_grid import graph_library_backend as library

def test_exact_owner_maps_and_downstream_summary():
    color=(1,)*7
    cases=[(1,[color],[]),(6,[color]*6,[]),(2,[color]*2,[(0,1,6,6)]*5),
           (3,[color]*3,[(0,1,6,6),(0,1,6,6),(0,2,6,6),(1,2,6,6)]),
           (3,[color,(0,)*7,color],[(0,1,1,2),(1,2,2,1)]),
           (4,[color]*4,[(0,1,1,2),(1,2,2,1),(2,3,1,2),(0,3,2,1)])]
    for n,colors,edges in cases:
        before=(list(colors),list(edges))
        assert scanner._automorphisms(n,colors,edges,backend='igraph_bliss')==scanner._automorphisms_legacy(n,colors,edges)
        assert (colors,edges)==before
    class State:
        owner_caps=[(2,)*7]*3
        typed_edges=[(0,1,6,6),(0,1,6,6),(0,2,6,6),(1,2,6,6)]
        top_pairs=[(u,v) for u,v,a,b in typed_edges]
    pairs=[(a,b) for a in range(7) for b in range(7)]
    assert scanner._action_aggregate(State(),pairs,automorphism_backend='igraph_bliss')==scanner._action_aggregate(State(),pairs,automorphism_backend='legacy')
    assert scanner._action_aggregate(State(),pairs)==scanner._action_aggregate(State(),pairs,automorphism_backend='legacy')

def test_dispatch_and_no_error_fallback(monkeypatch):
    c=[(1,)*7]*2;e=[(0,1,0,1)]
    assert library.select_backend(2,c,e)=='igraph_bliss'
    assert library.select_backend(7,[(1,)*7]*7,[])=='legacy'
    assert scanner._automorphisms(7,[(1,)*7]*7,[])==scanner._automorphisms_legacy(7,[(1,)*7]*7,[])
    assert library.select_backend(2,c,[[0,1,0,1]])=='legacy'
    with pytest.raises(ValueError):scanner._automorphisms(2,c,[[0,1,0,1]],backend='igraph_bliss')
    with pytest.raises(ValueError):scanner._automorphisms(2,c,e,backend='typo')
    monkeypatch.setitem(sys.modules,'igraph',None)
    with pytest.raises(ModuleNotFoundError):scanner._automorphisms(2,c,e)
    assert scanner._automorphisms(2,c,e,backend='legacy')==scanner._automorphisms_legacy(2,c,e)

def test_version_contract(tmp_path):
    from infinity_grid import __version__
    from infinity_grid.change_sessions import source_version
    from infinity_grid.submission import SubmissionError
    # The current identity is tested centrally; this record keeps its historical pin.
    p=tmp_path/'infinity_grid';p.mkdir()
    for v in ['0.6.0','0.7.0.dev4','0.8.0.dev11+lib','0.8.0.dev18+lib','0.8.0.dev19+lib','0.8.0.dev20+lib','0.8.0.dev21+lib','0.8.0.dev22+lib','0.8.0.dev23+lib','0.8.0.dev24+lib','0.8.0.dev25+lib','0.8.0.dev26+lib','0.8.0.dev27+lib','0.8.0.dev28+lib','0.8.0.dev29+lib','0.8.0.dev30+lib','0.8.0.dev31+lib','0.8.0.dev32+lib','0.8.0.dev33+lib','0.8.0.dev34+lib','0.8.0.dev35+lib','0.8.0.dev36+lib','0.8.0.dev37+lib','0.8.0.dev38+lib','0.8.0.dev46+lib','0.8.0.dev47+lib','0.8.0.dev48+lib','0.8.0.dev49+lib','0.8.0.dev50+lib','0.8.0.dev51+lib','0.8.0.dev52+lib','0.8.0.dev53+lib','0.8.0.dev58+lib','0.8.0+lib']:
        (p/'_version.py').write_text('__version__ = '+repr(v)+'\n');assert source_version(tmp_path)==v
    for v in ['0.8lib','0.8.0','0.8.0+other','0.9.0','garbage']:
        (p/'_version.py').write_text('__version__ = '+repr(v)+'\n')
        with pytest.raises(SubmissionError):source_version(tmp_path)


def test_representative_qualification_cannot_mint_external_authority():
    from infinity_grid.representative_qualification import handler
    from infinity_grid.v05_execution_authority import ExecutionAuthorityError
    import pytest
    with pytest.raises(ExecutionAuthorityError):
        handler({'stage_id': 'ENG:REPRESENTATIVE_QUALIFICATION'}, None)
