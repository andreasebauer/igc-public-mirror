import os
from pathlib import Path
import sys
import zipfile

from infinity_grid.o7_c1_typed_baseline import ORIGINAL
from infinity_grid.frontier import _load_engine
from infinity_grid.o7_edge_canonicalization import CANONICALIZATION_VERSION,canonicalize_with_witness,verify_relabeling_witness
from infinity_grid.regime_scanner import O7State


def test_native_frontier_engine_and_scanner_use_joint_canonicalizer(tmp_path,monkeypatch):
    root=tmp_path/'native';root.mkdir()
    with zipfile.ZipFile(ORIGINAL) as z:z.extractall(root)
    monkeypatch.setenv('OSCOUT_DATA_ROOT',str(root))
    engine=_load_engine(root)
    assert engine.CANONICALIZATION_VERSION==CANONICALIZATION_VERSION
    twin=engine.load_parent_records()['O6_TOPOLOGY_TWIN_A']
    ctx=engine.LaneContext('ENGINEERING_BINDING',[twin,twin],[twin.pid]*2)
    state=O7State(engine,ctx,())
    left,right=state._first_endpoint(0,0),state._first_endpoint(0,1)
    raw=(engine.make_edge(0,left[1],0,1,right[1],0),)
    # A legacy hook would otherwise silently remain on the scanner path.
    engine.canonicalize_edges=lambda *a: (_ for _ in ()).throw(AssertionError('legacy hook used'))
    changed=state.add_top_relation(0,1,0,0)
    assert changed is not None
    expected=canonicalize_with_witness(ctx,raw)
    assert changed.edges==expected['edges']
    assert verify_relabeling_witness(ctx,raw,changed.edges,expected['witness'])
    other=O7State(engine,ctx,changed.edges,source_id='other occurrence',lane='other lane')
    assert other.construction_digest==changed.construction_digest


def test_original_helper_cannot_mutate_parent_process_globals(tmp_path,monkeypatch):
    import infinity_grid.o7_c1_typed_baseline as baseline
    parent=os.getpid();key='STEP1_ISOLATION_TEST';name='_step1_isolation_test_module'
    monkeypatch.delenv(key,raising=False)
    def mutate(*a,**k):
        os.environ[key]='child';sys.modules[name]='child'
        return {'pid':os.getpid()}
    monkeypatch.setattr(baseline,'_reconstruct_original_case',mutate)
    result=baseline.reconstruct_original_case(tmp_path)
    assert result['pid']!=parent
    assert key not in os.environ and name not in sys.modules
