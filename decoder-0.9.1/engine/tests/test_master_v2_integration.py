"""New integration checks; synthetic panels grant no campaign authority."""
from copy import deepcopy
import hashlib
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from infinity_grid import regime_scanner as rs, maturation_parallel as mp
from infinity_grid import materialized_discovery as md, theorem_registry as tr
from infinity_grid import fixed_grammar_transport as ft, exact_carrier_acceleration as acceleration
from infinity_grid import semantic_sentinel as ss

GENERIC='O_GENERIC_FIXED_GRAMMAR_FACTORISATION_V1'
ORG='O_DEPTH_ERASED_NORMALIZED_ORGANIZATION_V1'
SERVICE='O_DEPTH_ERASED_RELATION_ADD_SERVICE_V1'
READ='O_EXACT_CARRIER_UNBLINDING_READ_LANE_V1'
SCOPE=(GENERIC,ORG,SERVICE)

def panel(n=4,kind='connected',full=True):
    states=[]
    for i in range(3):
        caps=[tuple(1+(u+t+i)%3 if full or (u+t+i)%3 else 0 for t in range(7)) for u in range(n)]
        edges=[] if kind=='empty' else ([(0,v,0,0) for v in range(1,n)] if i==1 else [(u,u+1,0,0) for u in range(n-1)])
        states.append(SimpleNamespace(owner_caps=caps,typed_edges=edges,top_pairs=[(u,v) for u,v,a,b in edges],
            total_caps=tuple(map(sum,zip(*caps))),construction_digest=hashlib.sha256(f'{n}:{kind}:{full}:{i}'.encode()).hexdigest(),
            skin='shared' if i<2 else 'other',lane='SYNTHETIC',motif_id=f'probe-{i}',leaf_count=7+i,relation_count_total=len(edges)+i))
    return states

def analyzed(states,pairs):
    return {s.construction_digest:mp._selected_state_worker({'state':mp.state_wire(s),'bridge_pairs':pairs}) for s in states}

def test_full_parallel_reducer_matches_serial_v2_complete_packets():
    for n,kind,full,pairs in [(2,'connected',True,list(product(range(2),repeat=2))),
        (4,'connected',True,list(product(range(7),repeat=2))),
        (5,'connected',False,list(product(range(7),repeat=2)))]:
        states=panel(n,kind,full);parallel=mp.scan_level_from_analysis(states,8,pairs,'synthetic',analyzed(states,pairs))
        assert parallel==rs._scan_level(states,8,pairs,'synthetic')
        assert parallel[0]['normalized_signature']['schema_id']==rs.ORGANIZATIONAL_SCHEMA_V2
        assert parallel[0]['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2


def test_disconnected_probes_remain_valid_but_full_panels_refuse():
    states=panel(6,'empty',False);pairs=[(0,0),(0,1),(1,0)];rows=analyzed(states,pairs)
    assert all(row['action']['schema_id']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2 for row in rows.values())
    with pytest.raises(ValueError,match='CONNECTED_PANEL_REQUIRED'):
        mp.scan_level_from_analysis(states,8,pairs,'synthetic',rows)
    with pytest.raises(ValueError,match='CONNECTED_PANEL_REQUIRED'):
        rs._scan_level(states,8,pairs,'synthetic')

def test_parallel_full_support_and_cross_route_genealogy_agree():
    states=panel();pairs=list(product(range(7),repeat=2));rows=analyzed(states,pairs)
    serial,_=rs._scan_level(states,7,pairs,'synthetic');parallel,_=mp.scan_level_from_analysis(states,8,pairs,'synthetic',rows)
    assert parallel['branching']['type_pair_support_min']==28
    assert parallel['obstruction_relief']['all_bridge_types_supported_everywhere']
    assert all(value=='PERSISTS' for value in rs._classify_genealogy(serial,parallel).values())
    assert parallel['quotient_observer']['resource_future_equivalence_basis']=='DIAGNOSTICS_ONLY_NO_FUTURE_EQUIVALENCE_AUTHORITY'

def test_reducer_refuses_old_unknown_or_missing_worker_schemas_before_graph(monkeypatch):
    states=panel();pairs=[(0,0)];rows=analyzed(states,pairs)
    monkeypatch.setattr(rs,'_panel_overlap',lambda *a:pytest.fail('reduction began'))
    for field in ['schema_id','organizational_schema']:
        bad=deepcopy(rows);bad[states[0].construction_digest].pop(field)
        with pytest.raises(ValueError,match='SCHEMA_MISMATCH'):mp.scan_level_from_analysis(states,8,pairs,'synthetic',bad)
    bad=deepcopy(rows);bad[states[0].construction_digest]['action'].pop('schema_id')
    with pytest.raises(ValueError,match='SCHEMA_MISMATCH'):mp.scan_level_from_analysis(states,8,pairs,'synthetic',bad)

def test_reducer_binds_exact_state_bridge_and_unique_roots():
    states=panel();pairs=[(0,0)];rows=analyzed(states,pairs)
    with pytest.raises(ValueError,match='INPUT_BINDING'):mp.scan_level_from_analysis(states,8,[(1,1)],'synthetic',rows)
    mutated=deepcopy(states);mutated[0].owner_caps[0]=tuple(x+1 for x in mutated[0].owner_caps[0])
    with pytest.raises(ValueError,match='INPUT_BINDING'):mp.scan_level_from_analysis(mutated,8,pairs,'synthetic',rows)
    with pytest.raises(ValueError,match='UNIQUE_STATES'):mp.scan_level_from_analysis([states[0]]*2,8,pairs,'synthetic',rows)
    with pytest.raises(ValueError,match='UNIQUE_STATES'):mp.scan_level_from_analysis([],8,pairs,'synthetic',{})
    assert mp.scan_level_from_analysis(states,8,pairs,'synthetic',rows)==rs._scan_level(states,8,pairs,'synthetic')

def test_actual_legacy_calibration_refuses_before_session_or_authority(monkeypatch):
    class TrapLock:
        def acquire(self):pytest.fail('session lock acquired')
        def release(self):pytest.fail('session lock released')
    monkeypatch.setattr(md,'_DISCOVERY_SESSION_LOCK',TrapLock())
    monkeypatch.setattr(md,'_install_isolated_scanner_caches',lambda:pytest.fail('caches changed'))
    monkeypatch.setattr(md,'_get_process_o7_runtime',lambda *a:pytest.fail('authority loaded'))
    for honor in [True,False]:
        with pytest.raises(md.MaterializedDiscoveryError,match='CALIBRATION_REOPEN_REQUIRED'):
            md.MaterializedDiscoverySession(honor_earned_laws=honor)

def test_calibration_version_gate_applies_above_old_window():
    obj=md.MaterializedDiscoverySession.__new__(md.MaterializedDiscoverySession)
    obj._diagnostic_calibration={'diagnostic_schema':rs.ACTION_DIAGNOSTIC_SCHEMA_V2,'organizational_schema':rs.ORGANIZATIONAL_SCHEMA_V2}
    for level in [7,13,14,100]:
        for scan in [{},{'diagnostic_schema':rs.ACTION_DIAGNOSTIC_SCHEMA_V2,'normalized_signature':{'schema_id':'LEGACY'}}]:
            with pytest.raises(md.MaterializedDiscoveryError,match='CALIBRATION_SCHEMA_MISMATCH'):
                obj._calibration_for_level(level,scan)

def test_earned_record_flags_and_cached_laws_cannot_grant_authority():
    with pytest.raises(tr.StaleTheoremError):rs.load_earned_regime_laws()
    with pytest.raises(tr.StaleTheoremError):rs._current_inherited_law({SERVICE:{'status':'EARNED'}},SERVICE)
    assert rs._current_inherited_law({},SERVICE) is None

def test_streaming_cached_metadata_still_checks_all_consumed_certificates(monkeypatch):
    calls=[]
    def verify(theorem_id,**kwargs):
        calls.append(theorem_id)
        if theorem_id==SERVICE:raise tr.StaleTheoremError('stale service')
        return {'status':'PASS'}
    monkeypatch.setattr(tr,'verify_theorem',verify)
    monkeypatch.setattr(ft,'_streaming_recipe_baseline_record',lambda:pytest.fail('cached authority consumed'))
    for i in range(2):
        with pytest.raises(tr.StaleTheoremError):ft.streaming_recipe_baseline()
    assert calls==list(SCOPE)*2

def test_streaming_reset_and_registered_frontier_refuse_before_effects(tmp_path,monkeypatch):
    output=tmp_path/'output';output.mkdir();sentinel=output/'keep.txt';sentinel.write_text('keep')
    monkeypatch.setattr(ft,'load_streaming_regime_spec',lambda:pytest.fail('spec/authority loaded'))
    monkeypatch.setattr(ft,'load_streaming_o7_seed',lambda:pytest.fail('seed loaded'))
    with pytest.raises(tr.StaleTheoremError):ft.run_fixed_grammar_transport(output,through=100,reset=True)
    inputs=[{},{},{},{}];before=deepcopy(inputs)
    with pytest.raises(tr.StaleTheoremError):ft._advance_frontier(*inputs)
    assert inputs==before and sentinel.read_text()=='keep'

def test_streaming_v1_lane_record_refuses_even_after_certificate_gate(monkeypatch):
    monkeypatch.setattr(ft,'require_current_theorems',lambda ids:[])
    ft.streaming_recipe_baseline.cache_clear()
    with pytest.raises(RuntimeError,match='STREAMING_DIAGNOSTIC_CALIBRATION_REOPEN_REQUIRED'):
        ft.streaming_recipe_baseline()

def test_exact_transport_requires_generic_and_read_lane_certificates(monkeypatch):
    called=[]
    def verify(theorem_id,**kwargs):
        called.append(theorem_id)
        if theorem_id==READ:raise tr.StaleTheoremError('stale read lane')
        return {'status':'PASS'}
    monkeypatch.setattr(tr,'verify_theorem',verify)
    monkeypatch.setattr(acceleration,'load_exact_carrier_acceleration_spec',lambda:pytest.fail('authority spec consumed'))
    with pytest.raises(tr.StaleTheoremError):acceleration.theorem_transport_certificate(level=8,parent_certificate_sha256='a'*64,
        material_instance_sha256='b'*64,authority_audit={'status':'PASS'})
    assert called==[GENERIC,READ]

def test_consumer_scope_is_explicit_and_fails_on_partial_or_false_pass(monkeypatch):
    for ids in [[],(),[GENERIC,GENERIC],'GENERIC',[None]]:
        with pytest.raises(tr.TheoremPremiseError,match='SCOPE_REQUIRED'):tr.require_current_theorems(ids)
    monkeypatch.setattr(tr,'verify_theorem',lambda *a,**k:{'status':'STALE_REOPEN_REQUIRED'})
    with pytest.raises(tr.StaleTheoremError):tr.require_current_theorems([GENERIC])

def test_authority_resources_stay_frozen_and_only_frontier_guard_hash_changes():
    root=Path(rs.__file__).parent
    expected={'resources/decoder/O_NATIVE_ACTION_REGISTRY_V1.json': '6357d4500807442896bbaeccfa00f8e0a5b991a7c4b2a117042fb22e06d6f57b', 'resources/decoder/O_NATIVE_SEMANTICS_BASELINE_V1.json': '7c7f211e34d54d539b9d04edf426765bd6049814b415a0b91ef7ff4edca59adf', 'resources/decoder/THEOREM_PREMISE_REGISTRY_V1.json': '6d6d744fad706a06aa8f3f615a8641a73a05cdf5b857efeb09135fd1cdd94dd8', 'resources/decoder/O7_O13_TRUST_REPAIR_CALIBRATION_V1.json': '40fc16fc1de5ba2be906542b3fbf0ac896902e55db78928b5c926b89dfb51447'}
    for name,digest in expected.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    parent={'EXACT_CANONICAL_RESERVATION_SAMPLER_V1': '63ca8189fdfa1d8ec360e4f89fec623db67bec7254691f334f86381b0be68f46', 'EXACT_PROJECTED_RELATION_ADD_PROFILE_V1': '7a9520c4dedd24983cb8aa32ad70abb28f763cc567c320e21bcddc7fa579b208', 'FIXED_GRAMMAR_FRONTIER_ADVANCE_V1': '8c2267d31505d03e62fe7d92b388054d3223682894b393b9c1218ba1be33ed9c', 'RESOURCE_SCANNER_ACTION_DIAGNOSTIC_V2': '8cfb3b5e00b90d51a064f4ea90b3daff2e1e65ec717e5871e60c61229074f3d2', 'RESOURCE_SCANNER_ORGANIZATIONAL_SIGNATURE_V2': '1e92b751f34a15beeb9e778b092e2da4bfa685fbb69e2923faacd0eee84860aa', 'RESOURCE_SCANNER_RELATION_ADD_AGGREGATE_V1': '3ebc339bf8bd0ad0db6febbe707ab664c4a34fcf7b772849dc84d2360ac46044'}
    current=ss.native_semantics_snapshot();changes=[row['action_id'] for row in current['actions'] if row['callable_semantic_sha256']!=parent[row['action_id']]]
    assert changes==['FIXED_GRAMMAR_FRONTIER_ADVANCE_V1']
    assert ss.verify_native_semantics(raise_on_change=False)['status']=='REOPEN_REQUIRED'
    result=tr.verify_all_theorems(raise_on_stale=False)
    assert result['status']=='STALE_REOPEN_REQUIRED' and len(result['stale_theorems'])==4
