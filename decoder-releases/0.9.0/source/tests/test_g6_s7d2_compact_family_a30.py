from __future__ import annotations

import json
from pathlib import Path


_OUTER_INDICES = (0, 17, 30, 37, 62, 93)


def _fixture():
    path = Path(__file__).with_name("fixtures") / "g6_s7d2_a30_certified_carriers.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _basis_context():
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel, tree_from_record
    from infinity_grid.g6_s7_evaluators import _s7_factor_swap_representative_contexts
    from infinity_grid.v05_kernel_service_providers import _basis_records

    records = _basis_records()
    refs = tuple(sorted(records))
    seed_trees = tuple(tree_from_record(records[ref]) for ref in refs)
    operators = tuple(ExactTreeRelationKernel().authority.operators)
    contexts = _s7_factor_swap_representative_contexts(refs, operators)
    assert len(contexts) == 124
    assert all(position == "LEFT" for _ref, _operator, position in contexts)
    return records, refs, seed_trees, operators, contexts


def _kernel(scope: str):
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel

    return ExactTreeRelationKernel(
        scope_identity=scope,
        max_cache_entries=4096,
        max_cache_bytes=512 * 1024 * 1024,
        max_relation_cache_entries=0,
        max_relation_cache_bytes=0,
    )


def _profile_counts(profiles):
    return tuple(int(profile.exact_outcome_count) for profile in profiles)


def _accepted_batch_counts(kernel, child, seed_trees, operators):
    return tuple(
        count
        for seed in seed_trees
        for count in _profile_counts(kernel.relation_profile_batch(child, seed, operators))
    )


def test_a30_compact_family_matches_accepted_exact_batch_on_wide_certified_a6_carriers():
    """72 full-parent coordinates retain the frozen structural P124 signature.

    The fixture is a byte-for-byte subset of the certified A6 reuse input: two
    exact members from six distinct collision classes across both S1 and higher
    panels.  The accepted A27 batch path obtains each outcome count from exact
    center-rooted canonical graft equality.  A30 must match all 124 coordinates
    for every exact outer child, then match the complete child-profile multiset.
    """
    from infinity_grid.exact_tree_relation_kernel import tree_from_record
    from infinity_grid.g6_s7_evaluators import _multiset_signature

    fixture = _fixture()
    assert fixture["source_reuse_sha256"] == "fe6ad6352b8f0ed1358a861900a43bb11fea6143c736f89c73750d07aad6d757"
    selected = fixture["selected"]
    assert len(selected) == 12
    assert {row["source_panel"] for row in selected} == {"s1", "higher"}
    _records, refs, seed_trees, operators, contexts = _basis_context()
    assert any(a != b for a, b in operators)  # asymmetric edge labels are included

    compared_parent_coordinates = 0
    compared_children = 0
    ambiguous_candidates = 0
    for parent_index, row in enumerate(selected):
        parent = tree_from_record(row["state"])
        for outer_index in _OUTER_INDICES:
            ref, outer_operator, position = contexts[outer_index]
            assert position == "LEFT"
            seed = seed_trees[refs.index(ref)]
            outer_kernel = _kernel(f"A30:WIDE:OUTER:{parent_index}:{outer_index}")
            outer = outer_kernel.relation(parent, seed, outer_operator)
            compact_kernel = _kernel(f"A30:WIDE:COMPACT:{parent_index}:{outer_index}")
            accepted_kernel = _kernel(f"A30:WIDE:ACCEPTED:{parent_index}:{outer_index}")
            compact_blocks = []
            accepted_blocks = []
            for child_index, child in enumerate(outer.children):
                compact = _profile_counts(
                    compact_kernel.relation_profile_family(child, seed_trees, operators)
                )
                accepted = _accepted_batch_counts(
                    accepted_kernel, child, seed_trees, operators
                )
                assert len(compact) == 124
                assert compact == accepted, (
                    row["state_token"], outer_index, child_index
                )
                compact_blocks.append(("S7_CHILD_D1_BRANCH_COUNT_PREFIX", compact))
                accepted_blocks.append(("S7_CHILD_D1_BRANCH_COUNT_PREFIX", accepted))
            assert _multiset_signature(compact_blocks) == _multiset_signature(accepted_blocks)
            compared_parent_coordinates += 1
            compared_children += len(outer.children)
            ambiguous_candidates += int(
                compact_kernel.metrics()["relation_profile_family_ambiguous_candidates"]
            )

    assert compared_parent_coordinates == 12 * len(_OUTER_INDICES)
    assert compared_children > compared_parent_coordinates
    # Ensures the test reached the conservative complete-signature branch, not
    # only the straightforward unique-new-bridge certificate.
    assert ambiguous_candidates > 0


def test_a30_family_preserves_order_and_falls_back_exactly_outside_s7d2_shape():
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
    from infinity_grid.g5_capabilities import _g5_s1_carriers

    carriers = _g5_s1_carriers()
    operators = tuple(ExactTreeRelationKernel().authority.operators)

    # Equal sizes may exchange whole input components; A30 must retain A27.
    equal_kernel = _kernel("A30:FALLBACK:EQUAL")
    equal_family = equal_kernel.relation_profile_family(
        carriers["D2_PATH"], (carriers["D2_BROOM"],), operators
    )
    equal_expected = _kernel("A30:FALLBACK:EQUAL:REFERENCE").relation_profile_batch(
        carriers["D2_PATH"], carriers["D2_BROOM"], operators
    )
    assert equal_family == equal_expected
    assert equal_kernel.metrics()["relation_profile_family_general_fallbacks"] == 1

    # Reversed sizes are outside the common-large-left S7D2 contract.
    reverse_kernel = _kernel("A30:FALLBACK:REVERSED")
    reverse_family = reverse_kernel.relation_profile_family(
        carriers["D2_PATH"], (carriers["D4_PATH"],), tuple(reversed(operators))
    )
    reverse_expected = _kernel("A30:FALLBACK:REVERSED:REFERENCE").relation_profile_batch(
        carriers["D2_PATH"], carriers["D4_PATH"], tuple(reversed(operators))
    )
    assert reverse_family == reverse_expected
    assert reverse_kernel.metrics()["relation_profile_family_general_fallbacks"] == 1

    # Empty vector boundaries are exact and side-effect free.
    empty_kernel = _kernel("A30:EMPTY")
    assert empty_kernel.relation_profile_family(carriers["D4_PATH"], (), operators) == ()
    assert empty_kernel.relation_profile_family(
        carriers["D4_PATH"], (carriers["D2_PATH"],), ()
    ) == ()
    assert empty_kernel.metrics().get("relation_profile_family_calls", 0) == 0


def test_a30_public_family_provider_is_right_major_and_exact():
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import (
        _basis_records,
        _exact_relation_profile_batch_service,
        _exact_relation_profile_family_service,
        _exact_relation_service,
    )

    basis = _basis_records()
    refs = tuple(sorted(basis))
    configure_relation_kernel(
        scope_identity="A30:PROVIDER:CHILD",
        max_relation_cache_entries=0,
        max_relation_cache_bytes=0,
    )
    operators = tuple(get_relation_kernel().authority.operators)
    child = None
    for operator in operators:
        relation = _exact_relation_service(basis["D4_PATH"], basis["D2_BROOM"], operator)
        if relation["children"]:
            child = relation["children"][0]
            break
    assert child is not None

    ordered_refs = tuple(reversed(refs))
    ordered_ops = tuple(reversed(operators))
    configure_relation_kernel(
        scope_identity="A30:PROVIDER:FAMILY",
        max_relation_cache_entries=0,
        max_relation_cache_bytes=0,
    )
    got = _exact_relation_profile_family_service(
        child, tuple(basis[ref] for ref in ordered_refs), ordered_ops
    )
    configure_relation_kernel(
        scope_identity="A30:PROVIDER:REFERENCE",
        max_relation_cache_entries=0,
        max_relation_cache_bytes=0,
    )
    expected = tuple(
        count
        for ref in ordered_refs
        for count in _exact_relation_profile_batch_service(
            child, basis[ref], ordered_ops
        )["exact_outcome_counts"]
    )
    assert tuple(got["exact_outcome_counts"]) == expected
    assert len(expected) == 124


def test_a30_validation_runtime_retains_pytest_prefixed_benchmark_payload():
    from infinity_grid.v05_validation_runtime import _benchmark_payloads

    payload = {"benchmark": "A30", "speedup": 12.5}
    output = ".IG_BENCHMARK_JSON:" + json.dumps(payload, sort_keys=True) + "\n1 passed"
    assert _benchmark_payloads(output) == [payload]
