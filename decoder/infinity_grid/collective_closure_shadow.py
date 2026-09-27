from __future__ import annotations

"""Non-promoting collective closure shadow observers.

The K4 descriptor in this module is deliberately outside the current O-regime
promotion/maturation signatures.  It exists to predict a previously discovered
H1->H2 branch-observer refinement on a frozen n=7 census without changing the
scientific authority of the existing scanner.
"""

import itertools
import json
from collections import Counter, defaultdict
from importlib.resources import files
from typing import Any, Iterable, Mapping, Sequence

from .canon import canonical_sha256

K4_DESCRIPTOR_ID = "COLLECTIVE_K4_CLOSURE_COUNT_V1"
SPEC_RESOURCE = "resources/decoder/O_REGIME_K4_SHADOW_OBSERVER_SPEC_v1.json"
FIXTURE_RESOURCE = "resources/decoder/O_REGIME_K4_SHADOW_N7_VALIDATION_FIXTURE_v1.json"


class CollectiveClosureShadowError(RuntimeError):
    pass


def _load_json_resource(path: str) -> dict[str, Any]:
    p = files("infinity_grid").joinpath(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_k4_shadow_spec() -> dict[str, Any]:
    spec = _load_json_resource(SPEC_RESOURCE)
    expected = spec.get("spec_sha256")
    observed = canonical_sha256({k: v for k, v in spec.items() if k != "spec_sha256"})
    if expected != observed:
        raise CollectiveClosureShadowError(f"K4 shadow spec hash mismatch: {expected} != {observed}")
    firewall = spec.get("firewall", {})
    forbidden_true = [
        "promotion_authorized",
        "research_frontier_mutation_authorized",
        "included_in_regime_normalized_signature",
        "included_in_maturation_stabilization_signature",
        "included_in_plateau_gate",
        "included_in_structural_shock_gate",
        "included_in_lift_gate",
        "included_in_resource_read_semantics",
    ]
    if any(firewall.get(k) is not False for k in forbidden_true):
        raise CollectiveClosureShadowError("K4 shadow firewall is not fail-closed")
    return spec


def load_k4_shadow_fixture() -> dict[str, Any]:
    fixture = _load_json_resource(FIXTURE_RESOURCE)
    expected = fixture.get("fixture_sha256")
    observed = canonical_sha256({k: v for k, v in fixture.items() if k != "fixture_sha256"})
    if expected != observed:
        raise CollectiveClosureShadowError(f"K4 shadow fixture hash mismatch: {expected} != {observed}")
    return fixture


def _simple_edge_support(n: int, pairs: Iterable[Sequence[int]]) -> set[tuple[int, int]]:
    n = int(n)
    if n < 0:
        raise CollectiveClosureShadowError("owner count must be nonnegative")
    support: set[tuple[int, int]] = set()
    for e in pairs:
        if len(e) != 2:
            raise CollectiveClosureShadowError(f"edge must have arity 2: {e!r}")
        u, v = int(e[0]), int(e[1])
        if not (0 <= u < n and 0 <= v < n):
            raise CollectiveClosureShadowError(f"edge outside owner range: {(u, v)!r} for n={n}")
        if u == v:
            # The collective K4 descriptor is defined on simple top-level owner support.
            # Self-relations do not contribute to any four-owner clique.
            continue
        support.add((u, v) if u < v else (v, u))
    return support


def k4_closure_count(n: int, pairs: Iterable[Sequence[int]]) -> int:
    """Count complete four-owner closures in simple top-level relation support."""
    n = int(n)
    support = _simple_edge_support(n, pairs)
    count = 0
    for quad in itertools.combinations(range(n), 4):
        if all(((u, v) if u < v else (v, u)) in support for u, v in itertools.combinations(quad, 2)):
            count += 1
    return count


def k4_shadow_observation(n: int, pairs: Iterable[Sequence[int]]) -> dict[str, Any]:
    spec = load_k4_shadow_spec()
    count = k4_closure_count(n, pairs)
    payload = {
        "schema_id": "IG_O_REGIME_K4_SHADOW_OBSERVATION_V1",
        "descriptor_id": K4_DESCRIPTOR_ID,
        "owner_count": int(n),
        "k4_closure_count": int(count),
        "status": "SHADOW_NONPROMOTING",
        "promotion_authorized": False,
        "research_frontier_mutated": False,
        "spec_sha256": spec["spec_sha256"],
    }
    payload["science_sha256"] = canonical_sha256(payload)
    return payload


def _partition_count(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> int:
    return len({tuple(row[f] for f in fields) for row in rows})


def _h1_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["future_h1_sha256"])].append(row)
    return groups


def validate_k4_shadow_prediction(fixture: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the frozen n=7 K4 shadow prediction gate.

    This is intentionally a shadow/integration validation on the frozen census
    where the descriptor was discovered.  It does not authorize promotion.
    """
    spec = load_k4_shadow_spec()
    
    if fixture is None:
        fixture_obj = dict(load_k4_shadow_fixture())
    else:
        fixture_obj = dict(fixture)
        expected_fixture_sha = fixture_obj.get("fixture_sha256")
        observed_fixture_sha = canonical_sha256({k: v for k, v in fixture_obj.items() if k != "fixture_sha256"})
        if not isinstance(expected_fixture_sha, str) or expected_fixture_sha != observed_fixture_sha:
            return {
                "schema_id":"IG_O_REGIME_K4_SHADOW_VALIDATION_RESULT_V1",
                "status":"FAIL",
                "classification":"UNVERIFIED_CUSTOM_FIXTURE_REJECTED",
                "promotion_authorized":False,
                "research_frontier_mutated":False,
                "errors":["custom K4 fixture is not self-hash verified"],
            }
        # The validation gate is frozen to the audited census authority; custom
        # fixtures are observations, not substitutes for the official gate.
        official = load_k4_shadow_fixture()
        if expected_fixture_sha != official.get("fixture_sha256"):
            return {
                "schema_id":"IG_O_REGIME_K4_SHADOW_VALIDATION_RESULT_V1",
                "status":"FAIL",
                "classification":"CUSTOM_FIXTURE_NOT_FROZEN_AUTHORITY",
                "promotion_authorized":False,
                "research_frontier_mutated":False,
                "errors":["custom K4 fixture is not the frozen validation authority"],
            }
    source_rows = fixture_obj.get("rows")
    if not isinstance(source_rows, list) or not source_rows:
        raise CollectiveClosureShadowError("K4 shadow fixture has no rows")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        n = int(source["n"])
        k4 = k4_closure_count(n, source["edges"])
        rows.append({
            "canonical_id": str(source["canonical_id"]),
            "future_h1_sha256": str(source["future_h1_sha256"]),
            "future_h2_sha256": str(source["future_h2_sha256"]),
            "k4_closure_count": int(k4),
        })

    h1_groups = _h1_groups(rows)
    h1_sizes = Counter(len(v) for v in h1_groups.values())
    doubletons = [v for v in h1_groups.values() if len(v) == 2]
    split_doubletons = sum(len({x["k4_closure_count"] for x in group}) == 2 for group in doubletons)

    # H1+K4 is a predictor of H2 iff no H1+K4 class contains more than one H2 label.
    h1_k4_groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        h1_k4_groups[(row["future_h1_sha256"], row["k4_closure_count"])].add(row["future_h2_sha256"])
    conflicts = sum(len(v) > 1 for v in h1_k4_groups.values())

    # Exhaustive relabel invariance: for n=7, evaluating all 5040 permutations on
    # all 853 graphs would be needlessly expensive for an integration gate.  The
    # descriptor is mathematically permutation-invariant by subset counting, but
    # we still execute a deterministic adversarial relabel suite over every one of
    # the six collision fibers plus fixed generic controls.
    relabel_checks = 0
    relabel_failures = 0
    control_rows = [group[0] for group in doubletons] + rows[:6]
    fixture_by_id = {str(x["canonical_id"]): x for x in source_rows}
    perms = [
        (6, 5, 4, 3, 2, 1, 0),
        (1, 2, 3, 4, 5, 6, 0),
        (2, 0, 6, 4, 1, 5, 3),
    ]
    for row in control_rows:
        source = fixture_by_id[row["canonical_id"]]
        n = int(source["n"])
        baseline = int(row["k4_closure_count"])
        for perm in perms:
            if len(perm) != n:
                continue
            relabeled = [[perm[int(u)], perm[int(v)]] for u, v in source["edges"]]
            relabel_checks += 1
            if k4_closure_count(n, relabeled) != baseline:
                relabel_failures += 1

    gate = spec["prediction_gate"]
    metrics = {
        "rows": len(rows),
        "h1_class_count": _partition_count(rows, ["future_h1_sha256"]),
        "h2_class_count": _partition_count(rows, ["future_h2_sha256"]),
        "h1_singletons": int(h1_sizes.get(1, 0)),
        "h1_doubletons": len(doubletons),
        "h1_plus_k4_class_count": _partition_count(rows, ["future_h1_sha256", "k4_closure_count"]),
        "h1_plus_k4_h2_conflict_count": int(conflicts),
        "h1_doubletons_split_by_k4": int(split_doubletons),
        "relabel_checks": int(relabel_checks),
        "relabel_failures": int(relabel_failures),
    }
    checks = {
        "complete_n7_rows": metrics["rows"] == 853,
        "h1_class_count": metrics["h1_class_count"] == int(gate["required_h1_classes"]),
        "h2_class_count": metrics["h2_class_count"] == int(gate["required_h2_classes"]),
        "h1_doubletons": metrics["h1_doubletons"] == int(gate["required_h1_doubletons"]),
        "h1_plus_k4_class_count": metrics["h1_plus_k4_class_count"] == int(gate["required_h1_plus_k4_classes"]),
        "h1_plus_k4_h2_conflicts": metrics["h1_plus_k4_h2_conflict_count"] == int(gate["required_h1_plus_k4_h2_conflicts"]),
        "all_h1_doubletons_split_by_k4": metrics["h1_doubletons_split_by_k4"] == metrics["h1_doubletons"] and bool(gate["required_all_h1_doubletons_split_by_k4"]),
        "relabel_invariance": metrics["relabel_failures"] == 0 and metrics["relabel_checks"] > 0 and bool(gate["required_relabel_invariance"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result = {
        "schema_id": "IG_O_REGIME_K4_SHADOW_VALIDATION_RESULT_V1",
        "status": status,
        "classification": "K4_SHADOW_PREDICTS_ALL_FROZEN_N7_H1_TO_H2_SPLITS" if status == "PASS" else "K4_SHADOW_PREDICTION_GATE_FAILED",
        "descriptor_id": K4_DESCRIPTOR_ID,
        "spec_sha256": spec["spec_sha256"],
        "fixture_sha256": fixture_obj["fixture_sha256"],
        "metrics": metrics,
        "checks": checks,
        "firewall": dict(spec["firewall"]),
        "promotion_authorized": False,
        "research_frontier_mutated": False,
        "nonclaim": "Frozen-census shadow prediction/integration pass only; independent holdout generalization is still required before any earned descriptor promotion.",
        "next": spec["next_if_pass"] if status == "PASS" else "REPAIR_OR_REJECT_SHADOW_DESCRIPTOR",
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
