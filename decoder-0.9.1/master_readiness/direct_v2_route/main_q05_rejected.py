"""Prospective direct V2 route qualification; no transport or discovery replay."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

from infinity_grid import direct_v2_observation as direct
from infinity_grid import maturation_parallel as mp
from infinity_grid import regime_scanner as rs
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.scientific_architecture import ScientificProtocolRegistry
from infinity_grid.canon import canonical_sha256, write_json_atomic


def main():
    started = time.monotonic()
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    parameters = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role, sha in parameters['input_hashes'].items():
        assert hashlib.sha256(Path(inputs[role]).read_bytes()).hexdigest() == sha
    root = Path('saved_inputs')
    root.mkdir()
    with zipfile.ZipFile(inputs['saved_inputs']) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in names)
        z.extractall(root)
    binding = json.loads(Path(inputs['saved_binding']).read_text())
    contract = json.loads(Path(inputs['direct_contract']).read_text())
    direct.validate_contract(contract)
    baseline = ss.verify_native_semantics(raise_on_change=False)
    theorems = tr.verify_all_theorems(raise_on_stale=False)
    assert baseline['status'] == 'REOPEN_REQUIRED' and len(theorems['stale_theorems']) == 4
    session = direct.SavedDirectSession(root, binding, contract)
    # The new route qualification reconstructs only the saved prefix, never
    # R100 or any generated/discovered carrier cohort.
    dag = json.loads((root/'PREFIX_DAG.json').read_text())
    engine, states = mp._states_from_dag(dag)
    index = json.loads((root/'PREFIX_INDEX.json').read_text())['roots']
    assert len(states) == len(index) == 747
    _ports, pairs = engine.O6.load_rules()
    pairs = sorted(tuple(map(int, p)) for p in pairs)
    calls = set()
    def profile(frame, event, arg):
        if event == 'call':
            calls.add((str(frame.f_globals.get('__name__', '')), frame.f_code.co_name))
    assert sys.getprofile() is None
    rows = []
    exports = []
    sys.setprofile(profile)
    try:
        for domain, level in sorted(session.packets):
            saved = session.dispatch(domain, level)
            selected = [state for state, row in zip(states, index) if row['domain'] == domain and row['level'] == level]
            assert len(selected) == (24 if domain == 'selected' else 193)
            observed = direct.observe_materialized(selected, level, pairs, contract)
            packet = saved['observations']['V2_DIAGNOSTIC_PACKET']
            assert observed['observations']['V2_DIAGNOSTIC_PACKET'] == packet, (domain, level)
            for capability in direct.CAPABILITIES:
                assert saved['observations'][capability] == observed['observations'][capability]
            assert observed['theorem_transport'] == saved['theorem_transport'] == 'NOT_CONSUMED'
            assert saved['provenance']['producing_capture_id'] == binding['prefix_capture_id']
            assert saved['provenance']['producing_completion_sha256'] == binding['prefix_completion_sha256']
            rows.append({'domain': domain, 'level': level, 'states': len(selected),
                         'full_packet_equal': True, 'packet_sha256': saved['packet_sha256'],
                         'native_result_sha256': observed['science_sha256']})
            exports.append(observed)
    finally:
        sys.setprofile(None)
    forbidden = []
    for module, function in calls:
        if module == 'infinity_grid.fixed_grammar_transport' and function not in ('<module>',):
            forbidden.append((module, function))
        if module == 'infinity_grid.materialized_discovery' and function in (
                '_read_probe', '_resource_future_signature', 'theorem_transport_consistency', 'advance_to'):
            forbidden.append((module, function))
        if module == 'infinity_grid.regime_scanner' and function in (
                'reserve_external', '_build_next_panel', '_farthest_select', 'load_earned_regime_laws'):
            forbidden.append((module, function))
    assert not forbidden, forbidden
    assert ('infinity_grid.regime_scanner', '_scan_level') in calls
    assert ('infinity_grid.regime_scanner', '_action_aggregate_v2') in calls
    write_json_atomic(Path('DIRECT_ROUTE_COMPARISONS.json'), {'status': 'PASS', 'packets': len(rows),
        'rows': rows, 'full_normalized_packets_equal': True, 'source_identities_preserved': True})
    write_json_atomic(Path('DIRECT_ROUTE_EXPORTS.json'), {'route': direct.ROUTE, 'exports': exports})
    write_json_atomic(Path('DIRECT_EXECUTION_TRACE.json'), {'status': 'PASS',
        'method': 'CPython call-profile observation; no called function replaced',
        'profile_restored': sys.getprofile() is None, 'unique_calls': sorted(calls),
        'forbidden_calls': forbidden, 'native_scanner_executed': True,
        'transport_or_sampled_future_calls': 0})
    refusals = []
    def refuse(name, callback):
        try:
            callback()
        except direct.DirectObservationError as exc:
            refusals.append({'name': name, 'reason': str(exc)})
        else:
            raise AssertionError('Missing refusal: ' + name)
    for capability in ('TRANSITION_SYSTEM', 'ACTION_MULTIPLICITY', 'MATCHED_BACKBONE',
                       'INHERITED_NO_READ_THEOREM', 'THEOREM_GROWTH_BOUND'):
        refuse(capability, lambda c=capability: direct.compile_requests(contract, [c]))
    for domain, level in [('selected', 14), ('selected', 100), ('full_pool', 10), ('other', 13)]:
        refuse(str((domain, level)), lambda d=domain, l=level: session.dispatch(d, l))
    invalid = deepcopy(contract)
    invalid['production_authority'] = 'PRODUCTION'
    refuse('authority_escalation', lambda: direct.compile_requests(invalid, list(direct.CAPABILITIES)))
    invalid_source = deepcopy(contract)
    invalid_source['package_payload_sha256'] = '0' * 64
    invalid_source['science_sha256'] = canonical_sha256({k:v for k,v in invalid_source.items() if k != 'science_sha256'})
    refuse('resealed_wrong_source', lambda: direct.compile_requests(invalid_source, list(direct.CAPABILITIES)))
    invalid_read = deepcopy(contract)
    invalid_read['read_set'].remove('typed_edges')
    invalid_read['science_sha256'] = canonical_sha256({k:v for k,v in invalid_read.items() if k != 'science_sha256'})
    refuse('hidden_read_omitted', lambda: direct.compile_requests(invalid_read, list(direct.CAPABILITIES)))
    refuse('invalid_native_population', lambda: direct.observe_materialized([], 13, pairs, contract))
    # Result container mutation cannot poison a later consumer.
    result = session.dispatch('selected', 13)
    result['observations']['V2_BRANCHING_DIAGNOSTIC'].clear()
    assert session.dispatch('selected', 13)['observations']['V2_BRANCHING_DIAGNOSTIC']
    # Read-after-admission input tampering must fail, even before a returned copy.
    artifact = root/binding['artifacts']['materialized_audit']['file_name']
    old_bytes = artifact.read_bytes()
    artifact.write_bytes(old_bytes + b' ')
    try:
        session.dispatch('selected', 13)
    except Exception as exc:
        assert 'SAVED_INPUT_BYTES_CHANGED' in str(exc)
        refusals.append({'name': 'input_changed_after_admission', 'reason': str(exc)})
    else:
        raise AssertionError('modified input admitted')
    finally:
        artifact.write_bytes(old_bytes)
    # Compile existing downstream needs honestly. New diagnostics cannot silently
    # satisfy the old scientific capability vocabulary.
    downstream = []
    for test in ScientificProtocolRegistry().tests():
        required = test['required_capabilities']
        missing = sorted(set(required) - set(direct.CAPABILITIES))
        assert missing
        refuse('legacy_test:' + test['test_id'], lambda req=required: direct.compile_requests(contract, req))
        downstream.append({'test_id': test['test_id'], 'required_capabilities': required,
                           'missing': missing, 'status': 'NEEDS_EXPLICIT_PROTOCOL_ADAPTER'})
    session.close()
    refuse('closed_session', lambda: session.dispatch('selected', 13))
    # Existing routes must keep refusing the unchanged stale certificates.
    for ident in theorems['stale_theorems']:
        try:
            tr.require_current_theorems((ident,))
        except tr.StaleTheoremError:
            pass
        else:
            raise AssertionError('old stale consumer admitted')
    assert ss.verify_native_semantics(raise_on_change=False) == baseline
    assert tr.verify_all_theorems(raise_on_stale=False) == theorems
    write_json_atomic(Path('DOWNSTREAM_COMPATIBILITY.json'), {'status': 'EXPLICIT_ADAPTER_REQUIRED',
        'legacy_tests_checked': len(downstream), 'tests': downstream,
        'missing_requirements_silently_ignored': False, 'full_pipeline_qualified': False})
    write_json_atomic(Path('REFUSAL_CHECKS.json'), {'status': 'PASS', 'checks': refusals,
        'check_count': len(refusals), 'caller_mutation_isolated': True})
    write_json_atomic(Path('QUALIFICATION_RESULT.json'), {'status': 'PASS',
        'scientific_outcome': 'DIRECT_V2_DIAGNOSTIC_ROUTE_FINITE_QUALIFIED_PROTOCOL_ADAPTER_REQUIRED',
        'packets_compared': len(rows), 'native_diagnostic_dispatch': True,
        'transport_or_sampled_future_calls': 0, 'stale_theorems': 4,
        'legacy_tests_requiring_adapter': len(downstream), 'refusal_checks': len(refusals),
        'production_authority': 'NONE', 'generation_allowed': False,
        'package2_status': 'OPEN_DOWNSTREAM_PROTOCOL_INTEGRATION',
        'wall_seconds': time.monotonic() - started})


if __name__ == '__main__':
    main()
