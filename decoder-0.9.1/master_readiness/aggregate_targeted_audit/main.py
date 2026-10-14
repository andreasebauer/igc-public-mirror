"""Prospectively captured aggregate diagnostic audit; never run directly."""
from collections import Counter
from itertools import combinations, combinations_with_replacement, product
import hashlib
import json
import os
from pathlib import Path
import time

from infinity_grid import regime_scanner as scanner
from infinity_grid import graph_library_backend as backend
from infinity_grid.semantic_sentinel import (
    native_semantics_snapshot, verify_native_semantics,
    NativeSemanticsReopenRequired, current_action_contract,
)
from infinity_grid.theorem_registry import verify_all_theorems, StaleTheoremError
from oracle import summary, owner_maps, digest, normalized_edge


def write(name, value):
    Path(name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


class ProbeState:
    def __init__(self, caps, edges):
        self._caps = [tuple(row) for row in caps]
        self._edges = list(edges)
        self._top = [(u, v) for u, v, a, b in edges]
        self.reads = set()

    @property
    def owner_caps(self):
        self.reads.add("owner_caps")
        return self._caps

    @property
    def typed_edges(self):
        self.reads.add("typed_edges")
        return self._edges

    @property
    def top_pairs(self):
        self.reads.add("top_pairs")
        return self._top

    def __getattr__(self, name):
        raise AssertionError("UNDECLARED_PROBE_FIELD:" + name)

    def value(self):
        return (tuple(self._caps), tuple(self._edges), tuple(self._top))


def main():
    started = time.monotonic()
    inputs = json.loads(Path(os.environ["DECODER_INPUTS_PATH"]).read_text())
    params = json.loads(Path(os.environ["DECODER_PARAMETERS_PATH"]).read_text())
    for name, expected in params["input_hashes"].items():
        assert hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest() == expected, name
    protocol = json.loads(Path(inputs["audit_protocol"]).read_text())
    parent_snapshot = json.loads(Path(inputs["parent_semantic_snapshot"]).read_text())
    parent_status = json.loads(Path(inputs["parent_semantic_baseline_status"]).read_text())
    current_snapshot = native_semantics_snapshot()
    assert current_snapshot == parent_snapshot
    baseline_status = verify_native_semantics(raise_on_change=False)
    assert baseline_status == parent_status
    assert baseline_status["status"] == "REOPEN_REQUIRED"
    expected_change = [{"action_id": "RESOURCE_SCANNER_RELATION_ADD_AGGREGATE_V1",
                        "change": "ACTION_CONTRACT_OR_IMPLEMENTATION_CHANGED",
                        "fields": ["callable_semantic_sha256"]}]
    assert baseline_status["comparison"]["changes"] == expected_change
    refusals = []
    for name, call in [("native_semantics", verify_native_semantics),
                       ("action_contract", current_action_contract)]:
        try:
            call()
        except NativeSemanticsReopenRequired:
            refusals.append(name)
        else:
            raise AssertionError("REOPEN_GUARD_DID_NOT_REFUSE:" + name)
    theorems = verify_all_theorems(raise_on_stale=False)
    assert theorems["status"] == "STALE_REOPEN_REQUIRED"
    assert len(theorems["stale_theorems"]) == protocol["expected_theorems_stale"]
    try:
        verify_all_theorems()
    except StaleTheoremError:
        refusals.append("theorem_registry")
    else:
        raise AssertionError("THEOREM_GUARD_DID_NOT_REFUSE")
    write("SEMANTIC_BASELINE_STATUS.json", baseline_status)
    write("THEOREM_PREMISE_STATUS.json", theorems)
    rows = []
    mode_counts = Counter()
    topology_reads = set()

    def check(case_id, caps, edges, pairs, modes):
        caps = tuple(tuple(row) for row in caps)
        edges = tuple(tuple(edge) for edge in edges)
        pairs = tuple(tuple(pair) for pair in pairs)
        expected = summary(caps, edges, pairs)
        maps = list(owner_maps(caps, edges))
        actual_hashes = {}
        for mode in modes:
            state = ProbeState(caps, edges)
            before = state.value()
            kwargs = {} if mode == "default" else {"automorphism_backend": mode}
            actual = scanner._action_aggregate(state, list(pairs), **kwargs)
            chosen = None if mode == "default" else mode
            colors = [tuple(int(x > 0) for x in row) for row in caps]
            actual_maps = scanner._automorphisms(len(caps), colors, list(edges), backend=chosen)
            if actual != expected or actual_maps != maps or state.value() != before:
                write("FIRST_FAILURE.json", {"case_id": case_id, "mode": mode,
                    "caps": caps, "edges": edges, "pairs": pairs, "expected": expected,
                    "actual": actual, "expected_maps": maps, "actual_maps": actual_maps,
                    "input_unchanged": state.value() == before})
                raise AssertionError("AGGREGATE_OR_GROUP_MISMATCH:" + case_id + ":" + mode)
            topology_reads.update(state.reads)
            actual_hashes[mode] = digest(actual)
            mode_counts[mode] += 1
        row = {"case_id": case_id, "input_sha256": digest([caps, edges, pairs]),
               "expected": expected, "owner_maps_sha256": digest(maps),
               "actual_sha256_by_mode": actual_hashes, "input_unchanged": True}
        rows.append(row)
        return expected

    # Complete finite family, fixed before execution: n=1..3, free support on
    # types 0/1, all 0..2 unordered typed edge occurrences (including duplicates).
    exhaustive = 0
    for n in range(1, 4):
        slots = [(u, v, a, b) for u, v in combinations(range(n), 2)
                 for a, b in product(range(2), repeat=2)]
        for color_id, colors in enumerate(product(range(4), repeat=n)):
            caps = [(color & 1, (color >> 1) & 1, 0, 0, 0, 0, 0) for color in colors]
            for m in range(3):
                for edge_id, edges in enumerate(combinations_with_replacement(slots, m)):
                    check(f"exhaustive:{n}:{color_id}:{m}:{edge_id}", caps, edges,
                          list(product(range(2), repeat=2)), ["legacy", "igraph_bliss", "default"])
                    exhaustive += 1
    assert exhaustive == protocol["expected_exhaustive_cases"]
    ext = json.loads(Path(inputs["extended_cases"]).read_text())
    extended = {}
    for row in ext["cases"]:
        extended[row["case_id"]] = check(row["case_id"], row["caps"], row["edges"],
                                        row["bridge_pairs"], row["modes"])
    assert len(extended) == protocol["expected_extended_cases"]

    # Relabel complete inputs, including endpoint type orientation, independently.
    relabel_count = 0
    order_counterexamples = []
    for row in ext["relabel_cases"]:
        p = row["permutation"]
        caps = [None] * len(p)
        for old, new in enumerate(p):
            caps[new] = row["caps"][old]
        edges = sorted(normalized_edge(p[u], p[v], a, b) for u, v, a, b in row["edges"])
        got = check(row["case_id"], caps, edges, row["bridge_pairs"],
                    ["legacy", "igraph_bliss", "default"])
        original = extended[row["original_case_id"]]
        core_fields = ["legal_action_labels", "total_action_copies", "type_pair_support",
                       "owner_pair_support", "action_orbits", "automorphism_size", "owner_orbits"]
        assert all(got[k] == original[k] for k in core_fields)
        changed = sorted(k for k in got if got[k] != original[k])
        if changed:
            order_counterexamples.append({"case_id": row["case_id"],
                "original_case_id": row["original_case_id"], "permutation": p,
                "changed_fields": changed, "original": original, "relabelled": got})
        relabel_count += 1
    assert relabel_count == protocol["expected_relabel_cases"]
    assert order_counterexamples, "ORDER_SENSITIVITY_PREDICTED_BY_STATIC_KEY_ANALYSIS_NOT_OBSERVED"
    write("RELABEL_SCOPE_CONTROL.json", {"schema_id": "IG_AGGREGATE_RELABEL_SCOPE_CONTROL_V1",
        "status": "PASS", "comparison_cases": relabel_count,
        "core_counts_and_group_invariants_preserved": True,
        "complete_diagnostic_relabelling_invariant": False,
        "counterexample_count": len(order_counterexamples), "counterexamples": order_counterexamples,
        "interpretation": "Ordered (a,b) diagnostic keying reads owner order; this is not a transition-label or q-action counterexample.",
        "role_scope_issue_requires_separate_resolution": True})

    outside_refusals = 0
    for row in ext["cases"]:
        if "igraph_bliss" in row["modes"]:
            continue
        state = ProbeState(row["caps"], [tuple(e) for e in row["edges"]])
        try:
            scanner._action_aggregate(state, [tuple(p) for p in row["bridge_pairs"]],
                                      automorphism_backend="igraph_bliss")
        except ValueError as exc:
            assert str(exc) == "Outside qualified igraph automorphism domain"
            outside_refusals += 1
        else:
            raise AssertionError("OUTSIDE_DOMAIN_WAS_ACCEPTED")
    assert outside_refusals == protocol["expected_outside_domain_refusals"]
    path = extended["control:path4"]
    star = extended["control:star4"]
    legal = ["legal_action_labels", "total_action_copies", "type_pair_support", "owner_pair_support"]
    assert all(path[k] == star[k] for k in legal)
    differing = sorted(k for k in path if path[k] != star[k])
    assert differing
    controls = {"schema_id": "IG_AGGREGATE_TOPOLOGY_SCOPE_CONTROL_V1", "status": "PASS",
        "legality_counts_equal_for_matched_caps": True, "diagnostic_topology_sensitive": True,
        "different_fields": differing, "path4": path, "star4": star,
        "observed_state_reads": sorted(topology_reads),
        "existing_registry_hidden_topology_read": False,
        "existing_registry_scientific_role": "RELATION_ADD_SEMANTICS",
        "diagnostic_is_complete_transition_counter": False,
        "role_scope_issue_requires_separate_resolution": True}
    write("TOPOLOGY_SCOPE_CONTROL.json", controls)
    write("CASE_RESULTS.json", {"schema_id": "IG_AGGREGATE_INDEPENDENT_CASE_RESULTS_V1",
                                "status": "PASS", "case_count": len(rows), "cases": rows})
    total_comparisons = sum(mode_counts.values())
    assert total_comparisons == protocol["expected_summary_comparisons"]
    result = {"schema_id": "IG_AGGREGATE_TARGETED_AUDIT_RESULT_V1", "status": "PASS",
        "scientific_outcome": protocol["scientific_outcome"],
        "exhaustive_cases": exhaustive, "extended_cases": len(extended),
        "relabel_cases": relabel_count, "summary_comparisons": total_comparisons,
        "exact_owner_group_comparisons": total_comparisons,
        "modes": dict(mode_counts), "outside_domain_refusals": outside_refusals,
        "guard_refusals": refusals, "theorems_stale": theorems["stale_theorems"],
        "all_parent_action_hashes_preserved": True,
        "semantic_baseline_status": "REOPEN_REQUIRED", "theorem_status": "STALE_REOPEN_REQUIRED",
        "baseline_updated": False, "guard_weakened": False,
        "role_scope_issue_requires_separate_resolution": True,
        "complete_diagnostic_relabelling_invariant": False,
        "diagnostic_order_counterexamples": len(order_counterexamples),
        "actual_current_R100_cohort_qualified": False, "full_generation_allowed": False,
        "historical_jobs_replayed": False, "case_results_sha256": digest(rows),
        "backend_usage_observed": backend.usage(), "wall_seconds": time.monotonic() - started,
        "scope": "Finite synthetic diagnostic inputs only; no campaign population or universal theorem adoption"}
    write("QUALIFICATION_RESULT.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
