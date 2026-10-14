"""Lossless assembler for an already evaluated G6 marker service bundle.

This module imports no Decoder, dispatches no work, and grants no scientific
acceptance. Runtime registration and independent verification are Step3 work.
"""
from copy import deepcopy
import json


class ExportContractError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ExportContractError(message)


def exact_json(value):
    """Reject floats, booleans as integers, opaque objects, and lossy coercion."""
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) in (list, tuple):
        return [exact_json(x) for x in value]
    if type(value) is dict:
        require(all(type(k) is str for k in value), 'non-string dictionary key')
        return {k: exact_json(v) for k, v in value.items()}
    raise ExportContractError('non-exact JSON value: ' + type(value).__name__)


def tree(record):
    require(type(record) is dict, 'missing full tree record')
    require(set(record) == {'n', 'edges', 'H_classes', 'edge_operators'}, 'tree fields')
    n = record['n']
    require(type(n) is int and n > 0, 'tree vertex count')
    require(len(record['H_classes']) == n and all(type(x) is str for x in record['H_classes']), 'H-class occurrence coverage')
    require(len(record['edges']) == len(record['edge_operators']) == n - 1, 'tree edge/operator coverage')
    for e, op in zip(record['edges'], record['edge_operators']):
        require(len(e) == len(op) == 2, 'edge/operator arity')
        require(all(type(x) is int for x in (*e, *op)), 'exact integer endpoints')
        require(e[0] != e[1] and all(0 <= x < n for x in e), 'vertex range')
        require(all(0 <= x < 7 for x in op), 'endpoint type range')
    # This is structural record validation, not a graph canon/resource verifier.
    return exact_json(record)


def assemble_marker_export(*, binding, left_tree, right_tree, operator,
                           left_marker_state, right_marker_state,
                           write_result, decoded_children):
    required = {'recipe_ref', 'source_ref', 'runtime_ref', 'input_root_ref', 'task_ref'}
    require(type(binding) is dict and set(binding) == required, 'exact binding fields')
    require(all(type(v) is str and v for v in binding.values()), 'empty binding')
    require(len(operator) == 2 and all(type(x) is int and 0 <= x < 7 for x in operator), 'operator')
    require(type(write_result) is dict and set(write_result) == {'states', 'diagnostics'}, 'full MARKER_WRITE result required')
    d = write_result['diagnostics']
    fields = {'left_decoded_unrooted_canon', 'right_decoded_unrooted_canon',
              'left_decode_stats', 'right_decode_stats', 'raw_child_count',
              'raw_child_canons', 'raw_child_rows', 'projected_marker_states',
              'fused_raw_projection'}
    require(type(d) is dict and fields <= set(d), 'missing marker diagnostics')
    count = d['raw_child_count']
    rows = d['raw_child_rows']
    require(type(count) is int and count >= 0, 'raw child count')
    require(d['fused_raw_projection'] is True, 'full fused projection required')
    require(len(rows) == len(d['raw_child_canons']) == len(decoded_children) == count, 'child correspondence coverage')
    require(exact_json(write_result['states']) == exact_json(d['projected_marker_states']), 'write/projection mismatch')
    children = []
    for i, (row, dec) in enumerate(zip(rows, decoded_children)):
        require(len(row) == 3, 'child record/state/identity triple required')
        record, state, identity = row
        require(exact_json(identity) == exact_json(d['raw_child_canons'][i]), 'canon correspondence mismatch')
        require(type(dec) is dict and {'status', 'candidate_count', 'candidates', 'tree', 'unrooted_canon', 'stats'} <= set(dec), 'full decoder record required')
        require(dec['status'] == 'PASS' and type(dec['candidate_count']) is int and dec['candidate_count'] == 1, 'decoder unsuccessful')
        require(len(dec['candidates']) == 1 and exact_json(dec['candidates'][0]) == exact_json(identity) == exact_json(dec['unrooted_canon']), 'decoder identity mismatch')
        # Different complete labeled representatives may have equal native
        # identity. Retain both; an independent verifier checks their equality.
        tree(dec['tree'])
        children.append({'ordinal': i, 'raw_tree': tree(record), 'marker_state': exact_json(state),
                         'native_identity': exact_json(identity), 'decoder': exact_json(dec)})
    out = {'schema_id': 'IG_MASTER_G6_MARKER_FULL_EXPORT_CANDIDATE_V1',
           'binding': exact_json(binding), 'parents': {'left': tree(left_tree), 'right': tree(right_tree)},
           'operator': list(operator), 'parent_marker_states': {'left': exact_json(left_marker_state), 'right': exact_json(right_marker_state)},
           'write_result': exact_json(write_result), 'children': children,
           'scientific_acceptance': False, 'owner_formation_completeness': 'SEPARATE_G6_OCCURRENCE_EXPORT_REQUIRED'}
    # Verify an exact JSON round trip, including arbitrary-precision integers.
    out = deepcopy(out)
    require(json.loads(json.dumps(out, allow_nan=False)) == out, 'JSON round trip')
    return out
