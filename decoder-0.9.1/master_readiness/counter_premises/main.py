"""Source-bound premise adjudication; no discovery or generation execution."""
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import time

from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import write_json_atomic


def method(tree, cls, name):
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node):
    return sorted({n.func.attr for n in ast.walk(node) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)})


def main():
    started = time.monotonic()
    base = Path(ss.__file__).parent
    names = ['regime_scanner.py', 'maturation_parallel.py', 'exact_carrier_unblinding.py',
             'fixed_grammar_transport.py', 'materialized_discovery.py', 'qualified_discovery_inputs.py']
    trees = {name: ast.parse((base / name).read_text()) for name in names}
    pins = {name: hashlib.sha256((base / name).read_bytes()).hexdigest() for name in names}
    semantic = ss.verify_native_semantics(raise_on_change=False)
    registry = tr.verify_all_theorems(raise_on_stale=False)
    assert semantic['status'] == 'REOPEN_REQUIRED'
    assert len(registry['stale_theorems']) == 4
    refusals = []
    for theorem_id in registry['stale_theorems']:
        try:
            tr.require_current_theorems((theorem_id,))
        except tr.StaleTheoremError:
            refusals.append(theorem_id)
        else:
            raise AssertionError('stale certificate admitted')
    try:
        ss.current_action_contract()
    except ss.NativeSemanticsReopenRequired:
        refusals.append('registered_current_action_contract')
    else:
        raise AssertionError('changed semantics admitted')
    dispositions = []
    for row in registry['results']:
        assert [x['premise'] for x in row['impact']['changes']] == ['native_semantics']
        dispositions.append({'theorem_id': row['theorem_id'], 'registered_scope': row['registered_scope'],
                             'disposition': 'REOPEN_KEEP_FAIL_CLOSED', 'automatic_rebind_allowed': False,
                             'resource_pins_changed': False, 'impact': row['impact']})
    # Inspect the saved-state restoration path without running it again.
    restored = next(n for n in trees['maturation_parallel.py'].body if isinstance(n, ast.FunctionDef) and n.name == '_states_from_dag')
    assert {'O7State', 'LiftState'} <= set(calls(restored))
    classes = {}
    for cls in ['O7State', 'LiftState']:
        node = next(n for n in trees['regime_scanner.py'].body if isinstance(n, ast.ClassDef) and n.name == cls)
        methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        assert 'reserve_external' in methods
        assert 'reservation_options' not in methods and 'h_struct_canon' not in methods
        classes[cls] = {'complete_reservation_counter_interface': False, 'construction_sampler': 'reserve_external'}
    exact = trees['exact_carrier_unblinding.py']
    fast = method(exact, 'ExactProjectedActionOracle', 'current_level_profile')
    ref = method(exact, 'ExactProjectedActionOracle', 'current_level_profile_materialized_reference')
    assert '_o7_owner_projected_counter' in calls(fast) and '_o7_owner_projected_counter' in calls(ref)
    # The O7 branch specifically shares the fast leaf kernel: not an independent leaf test.
    ref_if = next(n for n in ref.body if isinstance(n, ast.If))
    assert '_o7_owner_projected_counter' in calls(ast.Module(body=ref_if.body, type_ignores=[]))
    snapshot = semantic['current_snapshot']
    diagnostics = [a for a in snapshot['actions'] if a['action_id'].startswith('RESOURCE_SCANNER_')]
    assert len(diagnostics) == 3
    assert all(a['successor_semantics'].startswith('NONE') for a in diagnostics)
    assert all(a['hidden_reads'] for a in diagnostics)
    changed = semantic['comparison']['changes']
    assert len(changed) == 4 and sum(r['change'] == 'NEW_ACTION' for r in changed) == 2
    audit = {'status': 'PASS', 'source_sha256': pins, 'saved_restoration_classes': classes,
             'production_sampler_is_complete_counter': False,
             'O7_materialized_reference_shares_fast_leaf_kernel': True,
             'V2_diagnostics_are_complete_successor_counters': False,
             'registered_diagnostic_contracts': diagnostics, 'changed_action_contracts': changed,
             'scope': 'Exact named source functions and registered contracts; no universal runtime equivalence claim'}
    write_json_atomic(Path('SOURCE_READ_AUDIT.json'), audit)
    write_json_atomic(Path('STALE_BINDING_DISPOSITION.json'), {'status': 'PASS', 'dispositions': dispositions,
                       'semantic_status': semantic['status'], 'semantic_evidence': semantic, 'theorem_evidence': registry})
    # Independent finite algebra control: enumerate physical copies, then compare
    # the grouped product. This tests Counter collisions, NOT the native leaf adapter.
    controls = []
    for weights in itertools.product(range(1, 4), repeat=4):
        left = [('x', weights[0]), ('y', weights[1])]
        right = [('u', weights[2]), ('v', weights[3])]
        for collapse in [False, True]:
            def project(a, b):
                return 'collision' if collapse else (a, b)
            expanded = Counter(project(a, b) for a, wa in left for b, wb in right for _ in range(wa) for _ in range(wb))
            grouped = Counter()
            for a, wa in left:
                for b, wb in right:
                    grouped[project(a, b)] += wa * wb
            assert expanded == grouped and sum(grouped.values()) == sum(w for _, w in left) * sum(w for _, w in right)
            controls.append({'weights': weights, 'projection_collapses': collapse, 'copies': sum(grouped.values())})
    witness = {'complete_counter': {'successor_A': 2, 'successor_B': 3}, 'one_choice_sampler': {'successor_A': 1}}
    assert Counter(witness['complete_counter']) != Counter(witness['one_choice_sampler'])
    assert len(set(witness['complete_counter'])) != sum(witness['complete_counter'].values())
    write_json_atomic(Path('COUNTER_ALGEBRA_CONTROLS.json'), {'status': 'PASS', 'finite_controls': len(controls),
                       'controls': controls, 'abstract_sampler_counterexample': witness,
                       'native_O7_leaf_equivalence_proved': False, 'universal_theorem_machine_checked': False})
    obligations = [
        {'id': 'P1', 'premise': 'Complete leaf pointed-choice Counter and declared multiplicity unit', 'status': 'OPEN',
         'next_action': 'Independent O7 materialized leaf enumeration; define whether choices count sites, slots or orbits; compare existing targeted sentinels'},
        {'id': 'P2', 'premise': 'Parent projection is extensional in the child projected bag', 'status': 'CONDITIONAL_ARGUMENT_ONLY',
         'next_action': 'Bind digest implementation and exact retained state domain to the generic lift proof'},
        {'id': 'P3', 'premise': 'Pair actions use distinct owner occurrences and independent choices', 'status': 'CONDITIONAL_ARGUMENT_ONLY',
         'next_action': 'Bind admitted Bridge orientation, labels, scope and collision multiplicity to native action basis'},
        {'id': 'P4', 'premise': 'Caches and reads preserve full Counter equivalence', 'status': 'OPEN',
         'next_action': 'Prove canonical identity is sufficient for owner-indexed caches and map actual saved scanner classes to exact action semantics'},
        {'id': 'P5', 'premise': 'V2 organizational/service transport has current scientific authority', 'status': 'OPEN',
         'next_action': 'Reprove changed read scope or explicitly remove transport consumption from the production route; do not reset baseline'}]
    write_json_atomic(Path('COUNTER_PROOF_OBLIGATIONS.json'), {'status': 'BLOCKED', 'obligations': obligations,
                       'finite_algebra_controls_do_not_discharge_native_premises': True})
    assert ss.verify_native_semantics(raise_on_change=False) == semantic
    assert tr.verify_all_theorems(raise_on_stale=False) == registry
    write_json_atomic(Path('REFUSAL_CHECKS.json'), {'status': 'PASS', 'check_count': len(refusals), 'checks': refusals})
    write_json_atomic(Path('QUALIFICATION_RESULT.json'), {'status': 'PASS',
        'scientific_outcome': 'SOURCE_BOUND_COUNTER_PREMISE_GAPS_CONFIRMED_NO_REBIND_AUTHORITY',
        'package2_status': 'BLOCKED_ON_NATIVE_PROOF', 'stale_theorems': 4, 'refusal_checks': 5,
        'finite_algebra_controls': 162, 'Counter_read_premises_closed': False,
        'production_authority': 'NONE', 'generation_allowed': False, 'scientific_jobs_replayed': 0,
        'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
