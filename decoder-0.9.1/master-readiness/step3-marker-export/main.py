"""Execute only through the preserved Decoder SCRIPT capture."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel, tree_to_record, tree_from_record
from infinity_grid.g6_fresh_marker_kernel import (
    marker_observer_state, write_marker_states, decode_marker_observer_state, MarkerKernelError)
from marker_full_export import assemble_marker_export, exact_json, ExportContractError
from independent_marker_reference import freeze, canon, marker_reference


def main():
    inputs = json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    parameters = json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())

    def load(name):
        raw = Path(inputs[name]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == parameters['input_hashes'][name]
        return json.loads(raw)

    owner = load('accepted_r6_owner_export')
    owner_result = load('accepted_r6_qualification_result')
    classes = load('current_g4_class_authority')['candidate_interface']['rows']
    support = load('frozen_g6_support')['frozen_support']
    caps = {r['term_key']: r['caps7'] for r in classes}
    keys = {r['term_key']: r['candidate_key_sha256'] for r in classes}
    assert caps == support['H_class_caps7']
    assert owner['authority']['classes'] == [[k, keys[k], caps[k]] for k in sorted(keys)]
    assert owner_result['status'] == 'PASS' and owner_result['retained_SET_children'] == 18
    assert owner_result['full_export_sha256'] == parameters['input_hashes']['accepted_r6_owner_export']
    assert owner['formation_count'] == 35 and owner['child_SET_count'] == 18
    left, right = owner['parents']['left'], owner['parents']['right']
    assert left == support['S0_basis_records']['D2_PATH']['tree']
    assert right == support['S0_basis_records']['D4_BROOM']['tree']
    assert owner['operator'] == [0, 0] and owner['semantic_multiplicity'] == 'SET'
    kernel = ExactTreeRelationKernel(scope_identity=parameters['task_ref'])
    def full_decode(q):
        parent, stats = decode_marker_observer_state(tuple(q), kernel=kernel)
        identity = kernel.prepare(parent).unrooted_canon
        return {'status': 'PASS', 'candidate_count': 1, 'candidates': (identity,),
                'tree': tree_to_record(parent), 'unrooted_canon': identity, 'stats': dict(stats)}
    qleft = marker_observer_state(tree_from_record(left), kernel=kernel)
    qright = marker_observer_state(tree_from_record(right), kernel=kernel)
    native_diagnostics = {}
    native_states = write_marker_states(qleft, qright, (0, 0), kernel=kernel, diagnostics=native_diagnostics)
    write = {'states': native_states, 'diagnostics': native_diagnostics}
    diagnostics = write['diagnostics']
    decoded = [full_decode(q) for _record, q, _can in diagnostics['raw_child_rows']]
    binding = {k: parameters[k] for k in ['recipe_ref', 'source_ref', 'runtime_ref', 'input_root_ref', 'task_ref']}
    kwargs = dict(binding=binding, left_tree=left, right_tree=right, operator=[0, 0],
                  left_marker_state=qleft, right_marker_state=qright, write_result=write,
                  decoded_children=decoded)
    export = assemble_marker_export(**kwargs)
    assert len(export['children']) == diagnostics['raw_child_count'] == 18
    owner_ids = {canon(r['full_tree'], keys) for r in owner['child_SET']}
    raw_ids = {canon(r['raw_tree'], keys) for r in export['children']}
    assert owner_ids == raw_ids and len(raw_ids) == 18
    assert tuple(diagnostics['raw_child_canons']) == tuple(freeze(r['native_identity']) for r in export['children'])
    assert diagnostics['left_decoded_unrooted_canon'] == canon(left, keys)
    assert diagnostics['right_decoded_unrooted_canon'] == canon(right, keys)
    reference_rows = []; formation_count = 0; recovery_count = 0
    records = [('left', left, qleft, full_decode(qleft)),
               ('right', right, qright, full_decode(qright))]
    records += [('child:' + str(r['ordinal']), r['raw_tree'], r['marker_state'], r['decoder'])
                for r in export['children']]
    assert len(records) == 20
    for name, record, q, dec in records:
        native_q = freeze(q)
        reference_q, formations, recoveries = marker_reference(record, keys, caps)
        assert native_q == reference_q
        expected = canon(record, keys)
        assert dec['status'] == 'PASS' and dec['candidate_count'] == 1
        assert freeze(dec['unrooted_canon']) == expected == canon(dec['tree'], keys)
        assert freeze(dec['candidates']) == (expected,)
        cold = ExactTreeRelationKernel(scope_identity=parameters['task_ref'] + ':' + name)
        cold_parent, cold_stats = decode_marker_observer_state(native_q, kernel=cold)
        cold_record = tree_to_record(cold_parent)
        assert canon(cold_record, keys) == expected
        assert not cold_stats.get('cache_seeded_from_exact_marker_observation', False)
        assert cold_stats['all_children_same_parent'] is True
        assert cold_stats['marker_child_count'] == len(reference_q)
        formation_count += len(formations); recovery_count += len(recoveries)
        reference_rows.append({'occurrence': name, 'full_source_tree': record,
            'independent_identity': expected, 'independent_marker_state': reference_q,
            'marker_formation_occurrences': formations, 'independent_recovered_parents': recoveries,
            'cold_decoder_tree': cold_record, 'cold_decoder_stats': cold_stats,
            'native_kernel_decoder': dec})
    assert formation_count == 228
    expected_states = tuple(sorted({freeze(r['marker_state']) for r in export['children']}, key=repr))
    assert freeze(export['write_result']['states']) == expected_states
    assert len(expected_states) == 18
    # Export assembly fails closed on incomplete and corrupted service records.
    refusal_names = []
    for label in ['missing_diagnostics', 'bool_count', 'float_count', 'lost_raw_child',
                  'decoder_identity', 'lost_projection', 'empty_source', 'bool_vertex', 'float_operator']:
        bad = deepcopy(kwargs)
        d = bad['write_result']['diagnostics']
        if label == 'missing_diagnostics': del d['raw_child_rows']
        elif label == 'bool_count': d['raw_child_count'] = True
        elif label == 'float_count': d['raw_child_count'] = 18.0
        elif label == 'lost_raw_child': d['raw_child_rows'] = d['raw_child_rows'][:-1]
        elif label == 'decoder_identity': bad['decoded_children'][0]['unrooted_canon'] = ()
        elif label == 'lost_projection': bad['write_result']['states'] = ()
        elif label == 'empty_source': bad['binding']['source_ref'] = ''
        elif label == 'bool_vertex': bad['left_tree']['n'] = True
        elif label == 'float_operator': bad['operator'][0] = 0.0
        try: assemble_marker_export(**bad)
        except ExportContractError: refusal_names.append(label)
        else: raise AssertionError('export corruption admitted: ' + label)
    decode_refusals = []
    for label, q in [('empty', ()), ('missing_D', (canon(left, keys),)),
                     ('parent_disagreement', tuple(qleft) + tuple(qright))]:
        try: decode_marker_observer_state(q, kernel=ExactTreeRelationKernel(scope_identity='negative:' + label))
        except MarkerKernelError: decode_refusals.append(label)
        else: raise AssertionError('invalid marker decode admitted: ' + label)
    assert len(refusal_names) == 9 and len(decode_refusals) == 3
    huge = (1 << 200) + 7
    assert json.loads(json.dumps(exact_json({'counter': huge})))['counter'] == huge
    correspondence = {'schema_id': 'IG_MASTER_MARKER_INDEPENDENT_CORRESPONDENCE_V1',
        'binding': binding, 'accepted_owner_export_sha256': parameters['input_hashes']['accepted_r6_owner_export'],
        'authority': {'classes': owner['authority']['classes']}, 'records': reference_rows,
        'export_refusals': refusal_names, 'cold_decode_refusals': decode_refusals}
    for name, value in [('FULL_MARKER_EXPORT.json', export), ('INDEPENDENT_CORRESPONDENCE.json', correspondence)]:
        Path(name).write_text(json.dumps(exact_json(value), sort_keys=True, indent=2) + '\n')
    result = {'schema_id': 'IG_MASTER_STEP3_MARKER_EXPORT_QUALIFICATION_RESULT_V1',
        'status': 'PASS', 'scientific_outcome': 'FULL_MARKER_KERNEL_EXPORT_CORRESPONDENCE_EARNED_FOR_DECLARED_SCOPE',
        'accepted_owner_export_sha256': parameters['input_hashes']['accepted_r6_owner_export'],
        'parent_records': 2, 'retained_SET_children': 18, 'projected_marker_states': 18,
        'independent_marker_records': 20, 'marker_formation_occurrences': formation_count,
        'independent_marker_deletions': recovery_count, 'cold_decodes': 20,
        'export_refusals': len(refusal_names), 'cold_decode_refusals': len(decode_refusals),
        'arbitrary_precision_roundtrip_bits': huge.bit_length(),
        'full_export_sha256': hashlib.sha256(Path('FULL_MARKER_EXPORT.json').read_bytes()).hexdigest(),
        'correspondence_sha256': hashlib.sha256(Path('INDEPENDENT_CORRESPONDENCE.json').read_bytes()).hexdigest(),
        'pipeline_qualified': False, 'restricted_KernelView_stage_qualified': False,
        'G3_DAG_qualified': False, 'official_canary': False,
        'full_generation_allowed': False}
    Path('QUALIFICATION_RESULT.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
