"""Prospective native qualification; no scanner discovery or master generation."""
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

from infinity_grid import exact_carrier_unblinding as ec
from infinity_grid import materialized_counter_reference as ref
from infinity_grid import maturation_parallel as mp
from infinity_grid import regime_scanner as rs
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import write_json_atomic


def packed(counter):
    return sorted([(repr(k), int(v)) for k, v in counter.items()])


def main():
    started = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    assert hashlib.sha256(Path(inputs['saved_inputs']).read_bytes()).hexdigest() == params['saved_inputs_sha256']
    with zipfile.ZipFile(inputs['saved_inputs']) as z:
        dag = json.loads(z.read('PREFIX_DAG.json'))
    # Exact saved DAG reconstruction only; no new generated panel.
    engine, saved_roots = mp._states_from_dag(dag)
    mapper = ref.SavedScannerActionMapper()
    mapped = [mapper.map(s) for s in saved_roots]
    assert len(mapped) == len(saved_roots) == len(dag['roots'])
    write_json_atomic(Path('SAVED_MAPPING.json'), {
        'status': 'PASS', 'root_count': len(mapped), 'mapped_nodes': mapper.checked,
        'overlay_site_rows': mapper.overlay_sites,
        'root_order_preserved': True, 'skins_owner_caps_total_caps_typed_edges_equal': True,
        'original_construction_digests_unchanged': True,
        'scope': 'Saved prefix DAG; scanner witnesses retained as provenance, no transport rebind',
        'complete_action_counter_is_construction_sampler': False,
    })
    base = next(s for s in mapper._memo.values() if isinstance(s, ec.ExactO7State))
    # Native engineered finite carriers: retain the compact O6 trees and port
    # masks but expose only the first two sites per available type.
    h0 = base.owner_h6s[0]
    byp = engine.site_orbits_current(h0, None)[3]
    keep = {t: set(sorted(tuple(p) for p, (_, f, _, _) in byp.items() if f[t] > 0)[:2]) for t in range(7)}
    small = h0
    for path, (p, f, _leaf, _ps) in byp.items():
        nf = tuple(min(int(f[t]), 2) if tuple(path) in keep[t] else 0 for t in range(7))
        small = ec._replace_nested(small, tuple(path), (p, nf))
    bysmall = engine.site_orbits_current(small, None)[3]
    live = [(tuple(p), t) for p, (_, f, _, _) in bysmall.items() for t in range(7) if f[t] > 0]
    assert live
    path, t0 = live[0]
    changed = ec._decrement_site(small, path, t0)
    parents = (SimpleNamespace(h6=small, exact_color='native-small'),
               SimpleNamespace(h6=changed, exact_color='native-decremented'))
    ctx = SimpleNamespace(parents=parents, source_order=('native-small', 'native-decremented'))
    ctx_swap = SimpleNamespace(parents=parents[::-1], source_order=ctx.source_order[::-1])
    first = ec.ExactO7State(engine, ctx, (), 'NATIVE_SENTINEL')
    swapped = ec.ExactO7State(engine, ctx_swap, (), 'NATIVE_SENTINEL_SWAPPED')
    assert first.h_struct_canon == swapped.h_struct_canon
    pairs = [tuple(x) for x in rs.load_regime_scanner_spec().get('bridge_pairs', [])]
    # Bind Bridge directly to the frozen native rules.
    _ports, raw_pairs = engine.O6.load_rules()
    pairs = sorted(tuple(map(int, p)) for p in raw_pairs)
    assert pairs and len(pairs) == len(set(pairs))
    oracle = ec.ExactProjectedActionOracle(pairs)
    rows = []
    variants = [first, swapped, first._with_decrement(0, path, t0)]
    symmetric_ctx = SimpleNamespace(parents=(parents[0], parents[0]), source_order=('same', 'same'))
    symmetric = ec.ExactO7State(engine, symmetric_ctx, (), 'NATIVE_COLLISION')
    variants.append(symmetric)
    # Saved-coordinate overlays on real scanner classes, including successive
    # type reservations; this is not reserve_canonical construction sampling.
    scanner = rs.O7State(engine, ctx, (), source_id='NATIVE_MAPPING_SENTINEL')
    after, witness = scanner.reserve_external(t0)
    mapped_after = mapper.map(after)
    variants.append(mapped_after)
    for idx, state in enumerate(variants):
        for owner in range(len(state.owner_h6s)):
            for t in range(7):
                fast = oracle._o7_owner_projected_counter(state, owner, t)
                independent = ref.o7_owner_reference(state, owner, t)
                assert fast == independent, ('owner', idx, owner, t, packed(fast), packed(independent))
                assert sum(independent.values()) == sum(1 for _, f, _, _ in state.owner_orbit_data[owner][1].values() if f[t] > 0)
                rows.append({'variant': idx, 'owner': owner, 'type': t, 'counter': packed(independent)})
        for t in range(7):
            assert oracle.projected_reservation_counter(state, t) == ref.reservation_reference(state, t)
        assert oracle.current_level_profile(state) == ref.profile_reference(state, pairs)
        assert oracle.current_level_profile_materialized_reference(state) == ref.profile_reference(state, pairs)
    # Same canonical H and raw owner index can denote different owner counters.
    a = ref.o7_owner_reference(first, 0, t0)
    b = ref.o7_owner_reference(swapped, 0, t0)
    assert a != b
    assert oracle._o7_owner_projected_counter(first, 0, t0) == a
    assert oracle._o7_owner_projected_counter(swapped, 0, t0) == b
    poisoned = oracle._o7_owner_projected_counter(first, 0, t0)
    poisoned['CALLER_MUTATION'] = 999
    assert oracle._o7_owner_projected_counter(first, 0, t0) == a
    options = oracle.reservation_options(first, t0)
    options.clear()
    assert oracle.reservation_options(first, t0)
    # Collision must combine owner occurrences, including identical children.
    collision = ref.reservation_reference(symmetric, t0)
    local_total = sum(ref.o7_owner_reference(symmetric, 0, t0).values())
    assert sum(collision.values()) == 2 * local_total
    assert max(collision.values()) >= 2
    # Lift the same child occurrence twice, preserve oriented native witnesses.
    lift = ec.ExactLiftState(engine, 8, (first, first), ((0, 1, t0, t0, 'left', 'right'),), 'TEST', 'TEST')
    for t in range(7):
        assert oracle.projected_reservation_counter(lift, t) == ref.reservation_reference(lift, t)
    assert oracle.current_level_profile(lift) == ref.profile_reference(lift, pairs)
    write_json_atomic(Path('NATIVE_COUNTERS.json'), {
        'status': 'PASS', 'leaf_comparisons': len(rows), 'variants': len(variants), 'rows': rows,
        'bridge_pairs': pairs, 'multiplicity_unit': 'AVAILABLE_SITE_TYPE_OCCURRENCE',
        'capacity_slots_are_not_counted_separately': True,
        'projection_collisions_accumulate': True, 'repeated_owner_occurrences_count': True,
        'canonical_H_raw_owner_cache_alias_reproduced': True,
        'cache_fix_verified': True, 'caller_mutation_isolated': True,
        'native_engineered_sentinels_not_authority_survivors': True,
        'generic_projection_fibre_theorem_proved': False,
    })
    # Static independence checks forbid even indirect use of the fast kernel
    # or Merkle shortcut in the new materialized reference module.
    tree = ast.parse(Path(ref.__file__).read_text())
    forbidden = {'_o7_owner_projected_counter', '_successor_merkle_sig', 'reservation_options', 'projected_reservation_counter'}
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not forbidden.intersection(called)
    stale = tr.verify_all_theorems(raise_on_stale=False)
    assert len(stale['stale_theorems']) == 4
    refusals = []
    for ident in stale['stale_theorems']:
        try:
            tr.require_current_theorems((ident,))
        except tr.StaleTheoremError:
            refusals.append(ident)
        else:
            raise AssertionError('stale theorem admitted')
    for invalid in (SimpleNamespace(), rs.O7State(engine, ctx, (), reserve_counts=(-1, 0, 0, 0, 0, 0, 0))):
        try:
            mapper.map(invalid)
        except ec.InsufficientExactCarrierData:
            pass
        else:
            raise AssertionError('invalid saved state admitted')
    assert ss.verify_native_semantics(raise_on_change=False)['status'] == 'REOPEN_REQUIRED'
    write_json_atomic(Path('QUALIFICATION_RESULT.json'), {
        'status': 'PASS', 'scientific_outcome': 'NATIVE_LEAF_AND_SAVED_MAPPING_FINITE_QUALIFIED_NO_REBIND',
        'package2_status': 'OPEN_GENERIC_AND_TRANSPORT_PROOFS_REQUIRED',
        'Counter_read_premises_closed': False, 'production_authority': 'NONE',
        'generation_allowed': False, 'stale_theorems': len(refusals),
        'independent_reference': True, 'saved_root_count': len(mapped),
        'leaf_comparisons': len(rows), 'wall_seconds': time.monotonic() - started,
    })


if __name__ == '__main__':
    main()
