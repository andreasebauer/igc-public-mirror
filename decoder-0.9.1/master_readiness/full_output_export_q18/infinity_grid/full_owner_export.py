"""Candidate complete G5/G6 owner export, with injected Decoder functions.

No default kernel, private pool, generation dispatch or durable I/O is created.
Calling export_owner_occurrences constructs raw children and is scientific
work: registration, exact input binding and Step3 qualification are required.
"""
from .full_marker_export import exact_json, tree, require


def export_owner_occurrences(*, kernel, graft_pair_raw, tree_to_record,
                             left, right, operator, binding):
    require(type(binding) is dict and set(binding) == {
        'recipe_ref', 'source_ref', 'runtime_ref', 'input_root_ref', 'task_ref',
        'left_parent_occurrence_ref', 'right_parent_occurrence_ref'}, 'owner export binding')
    require(all(type(v) is str and v for v in binding.values()), 'empty binding')
    require(type(operator) in (tuple, list) and len(operator) == 2
            and all(type(x) is int and 0 <= x < 7 for x in operator), 'operator')
    op = tuple(operator)
    require(op in kernel.authority.operators, 'operator outside authority')
    lp, rp = kernel.prepare(left), kernel.prepare(right)
    left_record, right_record = tree(tree_to_record(left)), tree(tree_to_record(right))
    require(lp.legal is True and rp.legal is True, 'illegal input carrier')
    # FULL owners, never rooted_owner_representatives. Owner coordinates belong
    # to the exact labeled parent occurrence, not to an anonymous public class.
    lo, ro = tuple(lp.legal_owners(op[0])), tuple(rp.legal_owners(op[1]))
    for owners, prepared, record, endpoint_type in [(lo, lp, left_record, op[0]), (ro, rp, right_record, op[1])]:
        require(len(owners) == len(set(owners)), 'duplicate legal owner')
        require(all(type(v) is int and 0 <= v < record['n'] for v in owners), 'owner vertex range')
        require(len(prepared.local_remaining) == record['n'], 'local resource coverage')
        for caps in prepared.local_remaining:
            require(len(caps) == 7 and all(type(x) is int and x >= 0 for x in caps), 'exact local resource row')
        # Check that the provider returned every eligible vertex, not a
        # representative list with the right aggregate count.
        expected = tuple(i for i, caps in enumerate(prepared.local_remaining) if caps[endpoint_type] > 0)
        require(owners == expected, 'incomplete or reordered owner list')
    retained = kernel.relation(left, right, op)
    require(len(retained.children) == len(retained.canons), 'retained child correspondence')
    require(len(set(retained.canons)) == len(retained.canons), 'duplicate exact SET identity')
    retained_index = {can: i for i, can in enumerate(retained.canons)}
    children = []
    for can, child in zip(retained.canons, retained.children):
        require(kernel.prepare(child).unrooted_canon == can, 'retained child identity mismatch')
        children.append({'native_identity': exact_json(can), 'full_tree': tree(tree_to_record(child))})
    rows, represented = [], set()
    for lu in lo:
        for rv in ro:
            child = graft_pair_raw(left, right, lu, rv, op)
            record = tree(tree_to_record(child))
            # Validate ordered component slots and the exact added cross edge,
            # before consulting the native identity. No raw witness is discarded.
            off = left_record['n']
            expected_edges = left_record['edges'] + [[u+off, v+off] for u, v in right_record['edges']] + [[lu, rv+off]]
            require(record == {'n': off+right_record['n'], 'edges': expected_edges,
                               'H_classes': left_record['H_classes']+right_record['H_classes'],
                               'edge_operators': left_record['edge_operators']+right_record['edge_operators']+[list(op)]}, 'raw graft slot/wiring mismatch')
            can = kernel.prepare(child).unrooted_canon
            require(can in retained_index, 'raw owner outcome absent from retained SET')
            represented.add(can)
            rows.append({'ordinal': len(rows), 'left_owner': lu, 'right_owner': rv,
                         'operator': list(op), 'left_owner_remaining': exact_json(lp.local_remaining[lu]),
                         'right_owner_remaining': exact_json(rp.local_remaining[rv]),
                         'left_component_map': list(range(off)),
                         'right_component_map': list(range(off, off+right_record['n'])),
                         'full_raw_child': record, 'native_identity': exact_json(can),
                         'retained_child_ordinal': retained_index[can]})
    require(represented == set(retained.canons), 'retained SET contains unrepresented outcome')
    require(type(retained.legal_owner_pairs) is int and retained.legal_owner_pairs == len(rows), 'legal owner count mismatch')
    out = {'schema_id': 'IG_MASTER_COMPLETE_OWNER_RELATION_CANDIDATE_V1',
           'binding': exact_json(binding), 'authority': {'adapter_identity': exact_json(kernel.authority.adapter_identity),
            'operators': exact_json(kernel.authority.operators), 'classes': exact_json(kernel.authority.classes)},
           'parents': {'left': left_record, 'right': right_record}, 'operator': list(op),
           'eligible_left_owners': list(lo), 'eligible_right_owners': list(ro),
           'child_SET': children, 'formation_occurrences': rows,
           'formation_count': len(rows), 'child_SET_count': len(children),
           'semantic_multiplicity': 'SET', 'formation_order': 'LEFT_OWNER_MAJOR_RIGHT_OWNER_MINOR',
           'scientific_acceptance': False}
    return exact_json(out)
