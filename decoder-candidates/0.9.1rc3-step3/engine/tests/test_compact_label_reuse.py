"""Exactness and boundary checks for reuse of finite authority-label values."""
from dataclasses import replace
import pytest
import infinity_grid.exact_tree_relation_kernel as mod
from infinity_grid.g5_capabilities import _g5_s1_carriers


def test_compact_label_reuse_matches_generic_with_asymmetric_and_missing_labels(monkeypatch):
    kernel=mod.ExactTreeRelationKernel(); tree=_g5_s1_carriers()['D4_PATH']
    op=next(op for op in sorted(kernel.authority.operators) if op[0]!=op[1])
    tree=replace(tree,edge_operators=tuple(op if i%2==0 else (op[1],op[0]) for i in range(len(tree.edges))))
    expected=mod._compact_prepare_tree(tree,mod._ExactStructuralInterner(),kernel._frozen_keys,kernel._caps)
    table={op:mod.freeze_label(op),(op[1],op[0]):mod.freeze_label((op[1],op[0]))}
    original=mod.freeze_label;calls=[]
    def observe(value):
        calls.append(value);return original(value)
    monkeypatch.setattr(mod,'freeze_label',observe)
    actual=mod._compact_prepare_tree(tree,mod._ExactStructuralInterner(),kernel._frozen_keys,kernel._caps,frozen_edge_labels=table)
    assert actual==expected and not calls
    # Missing/reverse labels use the same full type-tagged representation.
    for partial in ({op:table[op]},{}):
        calls.clear()
        got=mod._compact_prepare_tree(tree,mod._ExactStructuralInterner(),kernel._frozen_keys,kernel._caps,frozen_edge_labels=partial)
        assert got==expected and calls
    assert table[op]!=table[(op[1],op[0])]


@pytest.mark.parametrize('bad',[True,1.0])
def test_compact_label_lookup_cannot_admit_bool_or_float_operator_aliases(bad):
    kernel=mod.ExactTreeRelationKernel();tree=_g5_s1_carriers()['D4_PATH']
    tree=replace(tree,edge_operators=((bad,0),)+tree.edge_operators[1:])
    for table in (None,kernel._endpoint_labels):
        with pytest.raises(mod.ExactTreeRelationKernelError):
            mod._compact_prepare_tree(tree,mod._ExactStructuralInterner(),kernel._frozen_keys,kernel._caps,frozen_edge_labels=table)


def test_family_label_reuse_keeps_authority_table_bounded_and_scopes_exact():
    carriers=_g5_s1_carriers();left=carriers['D4_PATH'];rights=(carriers['D2_PATH'],carriers['D2_BROOM'])
    kernel=mod.ExactTreeRelationKernel(scope_identity='LABEL_REUSE:A');table=dict(kernel._endpoint_labels)
    ops=tuple(reversed(kernel.authority.operators))
    expected=tuple(p for right in rights for p in mod.ExactTreeRelationKernel(scope_identity='LABEL_REUSE:REFERENCE').relation_profile_batch(left,right,ops))
    # Scientific counts and owner enumeration must agree; work counters differ
    # intentionally between accepted batch and compact family algorithms.
    def semantics(profiles):
        return tuple((p.exact_outcome_count,p.attempted_owner_pairs,p.legal_owner_pairs,
                      p.rooted_owner_pair_candidates) for p in profiles)
    for _ in range(2):
        assert semantics(kernel.relation_profile_family(left,rights,ops))==semantics(expected)
        assert kernel._endpoint_labels==table and len(table)==len(kernel.authority.operators)
    assert semantics(mod.ExactTreeRelationKernel(scope_identity='LABEL_REUSE:B').relation_profile_family(left,rights,ops))==semantics(expected)
    with pytest.raises(mod.ExactTreeRelationKernelError):
        kernel.relation_profile_family(left,rights,((999,999),))
    assert kernel._endpoint_labels==table
