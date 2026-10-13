from __future__ import annotations

"""G5:R2 non-promoting predictive hidden action-read audit.

This stage starts from the R1-certified finite endpoint-typed H-class-labelled
hidden tree fiber.  It asks whether a local nonbacktracking cavity message can
replace the exact rooted decorated-tree canon for the frozen one-step exact
hidden-child observer.  It also transfers the path/broom fixed-depth lower
bound to the graduated G5 hidden carrier and executes two fresh bounded panels.
"""
from collections import defaultdict
from importlib.resources import files
from itertools import combinations, product
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .decorated_tree_messages import graft_fixed_leaf
from .typed_tree_messages import prepare_endpoint_decorated_tree, endpoint_unrooted_tree_canon
from .uplift_g3_r1 import _generate_tree_shapes


class G5R2Error(RuntimeError):
    pass

_SPEC = "G5_R2_PREDICTIVE_HIDDEN_ACTION_READ_SPEC_V1.json"


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def r2_spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_R2_PREDICTIVE_HIDDEN_ACTION_READ_SPEC_V1":
        raise G5R2Error("bad G5:R2 spec schema")
    if canonical_sha256({k: v for k, v in obj.items() if k != "science_sha256"}) != obj.get("science_sha256"):
        raise G5R2Error("G5:R2 spec hash mismatch")
    return obj


def verify_authority(r1_primary: Mapping[str, Any], r1_verification: Mapping[str, Any], r1_closeout: Mapping[str, Any]) -> dict[str, Any]:
    s = r2_spec(); a = s["authority"]; failures: list[str] = []
    if r1_primary.get("schema_id") != "IG_G5_R1_FIBER_GRAFT_RESULT_V1" or r1_primary.get("status") != "PASS":
        failures.append("R1_PRIMARY_SCHEMA_OR_STATUS")
    if str(r1_primary.get("classification")) != str(a["required_r1_classification"]):
        failures.append("R1_CLASSIFICATION")
    if str(r1_primary.get("science_sha256")) != str(a["g5_r1_primary_science_sha256"]):
        failures.append("R1_PRIMARY_IDENTITY")
    if r1_primary.get("g5_graduation_preserved") is not True or r1_primary.get("topology_promoted") is not False:
        failures.append("R1_FIREWALL")
    if r1_primary.get("exact_reference_hidden_read") != "ENDPOINT_TYPED_DECORATED_TREE_CANON":
        failures.append("R1_EXACT_REFERENCE_READ")
    if r1_verification.get("status") != "PASS" or str(r1_verification.get("verification_sha256")) != str(a["g5_r1_independent_verification_sha256"]):
        failures.append("R1_VERIFICATION")
    if r1_closeout.get("status") != "CERTIFIED_PASS" or str(r1_closeout.get("closeout_sha256")) != str(a["g5_r1_closeout_sha256"]):
        failures.append("R1_CLOSEOUT")
    out = {
        "schema_id": "IG_G5_R2_AUTHORITY_CHECK_V1",
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "r1_primary_science_sha256": r1_primary.get("science_sha256"),
        "r1_verification_sha256": r1_verification.get("verification_sha256"),
        "r1_closeout_sha256": r1_closeout.get("closeout_sha256"),
        "public_descriptor": a["required_public_descriptor"],
        "g5_graduation_preserved": True,
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G5R2Error("authority mismatch: " + ",".join(failures))
    return out


def _q_wire(ad: G4AcceptedAdapter, tree: DecoratedG4Tree) -> tuple[Any, ...]:
    q = ad.public_read(tree)
    if not q.get("legal"):
        raise G5R2Error("fresh-panel parent unexpectedly illegal")
    return (
        bool(q.get("legal")), str(q.get("descriptor")), tuple(int(x) for x in q.get("caps7", [])),
        tuple(sorted((str(k), int(v)) for k, v in dict(q.get("H_class_bag", {})).items())),
    )


def _prepared(ad: G4AcceptedAdapter, tree: DecoratedG4Tree):
    # Build once per exact decorated parent; exact tuple equality remains the
    # scientific equality operation.  Hashes are only output handles.
    keys = [ad._vertex_key(k) for k in tree.H_classes]  # internal registered adapter identity
    return prepare_endpoint_decorated_tree(tree.n, tree.edges, keys, tree.edge_operators), keys


def _graft_child_canon(ad: G4AcceptedAdapter, tree: DecoratedG4Tree, root: int, new_H: str, op: Sequence[int]) -> tuple[Any, ...] | None:
    rel = ad.graft_relation(tree, int(root), new_H_class=str(new_H), operator=tuple(map(int, op)))
    if not rel:
        return None
    if len(rel) != 1:
        raise G5R2Error("explicit-owner leaf graft must be single-valued")
    return ad.unrooted_canon(rel[0])


def global_fixed_depth_lower_bound(max_support_depth: int = 12) -> dict[str, Any]:
    s = r2_spec(); ad = G4AcceptedAdapter(); rows = []
    ckey = ad._vertex_key("C")
    ops = set(ad.operator_basis())
    if (0, 0) not in ops:
        raise G5R2Error("operator (0,0) absent")
    # Public resource realizability: C has enormous positive type-0 capacity in
    # the frozen accepted adapter; a max-degree-3 tree uses at most three type-0
    # reservations per vertex.
    c_caps = tuple(int(x) for x in ad._rows["C"]["caps7"])
    if c_caps[0] < 3:
        raise G5R2Error("C type-0 capacity < 3")
    for depth in range(int(max_support_depth) + 1):
        n = depth + 3
        path_edges = [(i, i + 1) for i in range(n - 1)]
        broom_edges = [(i, i + 1) for i in range(depth)] + [(depth, depth + 1), (depth, depth + 2)]
        colors = [ckey] * n; grades = [(0, 0)] * (n - 1)
        pp = prepare_endpoint_decorated_tree(n, path_edges, colors, grades)
        pb = prepare_endpoint_decorated_tree(n, broom_edges, colors, grades)
        mp = pp.rooted_truncated(0, depth); mb = pb.rooted_truncated(0, depth)
        if mp != mb:
            raise G5R2Error(f"path/broom local messages split at d={depth}")
        cp = endpoint_unrooted_tree_canon(n + 1, path_edges + [(0, n)], colors + [ckey], grades + [(0, 0)])
        cb = endpoint_unrooted_tree_canon(n + 1, broom_edges + [(0, n)], colors + [ckey], grades + [(0, 0)])
        if cp == cb:
            raise G5R2Error(f"path/broom children failed to separate at d={depth}")
        # Verify actual public Q equality through the accepted adapter for the parent pair.
        tp = DecoratedG4Tree(n, tuple(path_edges), tuple("C" for _ in range(n)), tuple((0, 0) for _ in range(n - 1)))
        tb = DecoratedG4Tree(n, tuple(broom_edges), tuple("C" for _ in range(n)), tuple((0, 0) for _ in range(n - 1)))
        qp, qb = _q_wire(ad, tp), _q_wire(ad, tb)
        if qp != qb:
            raise G5R2Error(f"path/broom public Q mismatch at d={depth}")
        rows.append({
            "depth": depth, "hidden_vertex_count": n,
            "same_public_Q": True, "same_depth_d_action_read": True,
            "distinct_exact_child_canons": True,
            "public_Q_sha256": canonical_sha256(qp),
            "truncated_action_read_sha256": canonical_sha256(mp),
            "path_child_canon_sha256": canonical_sha256(cp),
            "broom_child_canon_sha256": canonical_sha256(cb),
        })
    out = {
        "schema_id": "IG_G5_R2_FIXED_DEPTH_LOWER_BOUND_V1", "status": "PASS",
        "statement": "For every fixed finite d, a C-only path/broom pair in the R1-certified G5 hidden tree carrier has equal public Q and equal depth-d endpoint-typed cavity action read, while the same fixed C/(0,0) root graft yields distinct exact hidden child canons.",
        "proof_scope": "C_ONLY_MAX_DEGREE_3_OPERATOR_(0,0)_RESOURCE_REALIZABLE_G5_SUBCARRIER_AND_FROZEN_EXACT_CHILD_OBSERVER",
        "resource_realizability": {"C_type0_capacity": c_caps[0], "required_max_degree": 3, "sequential_leaf_attachment_proof": True},
        "support_rows": rows,
        "machine_support_depth_0_through": int(max_support_depth),
        "global_fixed_finite_local_depth_sufficient": False,
        "promotion": False,
    }
    out["science_sha256"] = canonical_sha256(out); return out


def _panel_from_parents(*, name: str, n: int, parents, candidate_depths: Sequence[int], expected_raw: int, new_H: str = "C", new_op=(0,0)) -> dict[str, Any]:
    ad = G4AcceptedAdapter(); exact: dict[tuple[Any, ...], dict[str, Any]] = {}; raw = 0; parent_count = 0
    first_conflict: dict[str, dict[str, Any]] = {}
    for tree in parents:
        q = _q_wire(ad, tree); prep, _keys = _prepared(ad, tree); parent_count += 1
        rooted = prep.all_rooted_canons()
        messages_by_root = tuple(tuple(prep.rooted_truncated(root, int(d)) for d in candidate_depths) for root in range(n))
        for root in range(n):
            raw += 1
            child = _graft_child_canon(ad, tree, root, new_H, new_op)
            if child is None:
                continue
            key = (q, rooted[root], str(new_H), tuple(map(int,new_op)))
            rec = {"Q": q, "rooted_parent_canon": rooted[root], "child_canon": child, "messages": messages_by_root[root]}
            old = exact.get(key)
            if old is None:
                exact[key] = rec
            elif old["child_canon"] != child or old["messages"] != rec["messages"]:
                raise G5R2Error(f"{name}: exact rooted dedup invariance failed")
    if raw != int(expected_raw):
        raise G5R2Error(f"{name}: raw count mismatch {raw} != {expected_raw}")
    # Intern exact child tuples; dict equality is full structural tuple equality.
    child_ids: dict[tuple[Any, ...], int] = {}
    for rec in exact.values():
        ch = rec["child_canon"]
        if ch not in child_ids: child_ids[ch] = len(child_ids) + 1
        rec["child_id"] = child_ids[ch]
    depth_rows=[]; first_predictive=None
    for j, depth in enumerate(candidate_depths):
        groups: dict[tuple[Any, ...], set[int]] = defaultdict(set)
        examples: dict[tuple[Any, ...], list[dict[str,Any]]] = defaultdict(list)
        for rec in exact.values():
            k=(rec["Q"], rec["messages"][j], str(new_H), tuple(map(int,new_op)))
            groups[k].add(rec["child_id"])
            if len(examples[k])<3: examples[k].append(rec)
        bad=[(k,v) for k,v in groups.items() if len(v)>1]
        depth_key = str(int(depth))
        if bad and depth_key not in first_conflict:
            k,v=bad[0]; ex=examples[k]
            first_conflict[depth_key]={
                "depth":int(depth),"distinct_child_count":len(v),
                "Q_sha256":canonical_sha256(k[0]),"action_read_sha256":canonical_sha256(k[1]),
                "example_rooted_parent_canons_sha256":[canonical_sha256(x["rooted_parent_canon"]) for x in ex],
                "example_child_canons_sha256":[canonical_sha256(x["child_canon"]) for x in ex],
            }
        pred=not bad
        if pred and first_predictive is None: first_predictive=int(depth)
        depth_rows.append({
            "depth":int(depth),"action_read_class_count":len(groups),"nonpredictive_class_count":len(bad),
            "max_distinct_child_canons_in_one_QA_class":max((len(v) for _,v in bad),default=1),
            "predictive":pred,"class_count_ratio_to_exact_rooted_action_cases":len(groups)/len(exact) if exact else 1.0,
        })
    if first_predictive is None:
        raise G5R2Error(f"{name}: no predictive registered depth")
    row = next(x for x in depth_rows if x["depth"]==first_predictive)
    nontrivial = bool(row["class_count_ratio_to_exact_rooted_action_cases"] < 1.0)
    out={
        "schema_id":"IG_G5_R2_FRESH_ACTION_READ_PANEL_V1","status":"PASS","name":name,"n":n,
        "decorated_parent_instances":parent_count,"raw_action_instances":raw,"exact_rooted_action_cases_after_dedup":len(exact),
        "exact_child_canon_count":len(child_ids),"candidate_depths":[int(x) for x in candidate_depths],"depth_rows":depth_rows,
        "first_predictive_depth":first_predictive,"nontrivial_reduction_candidate_survived":nontrivial,
        "first_conflict_by_nonpredictive_depth":first_conflict,"structural_equality_used":True,"digest_only_equality_used":False,
    }
    out["science_sha256"]=canonical_sha256(out); return out


def fresh_n8_color_panel() -> dict[str, Any]:
    s=r2_spec()["fresh_panel_A"]; n=int(s["n"]); shapes=_generate_tree_shapes(n)[n]
    if len(shapes)!=23: raise G5R2Error("n8 tree count mismatch")
    def gen():
        for edges in shapes.values():
            for fiber in s["public_Q_fibers"]:
                dd=int(fiber["D_count"])
                for dpos in combinations(range(n),dd):
                    colors=["C"]*n
                    for j in dpos: colors[j]="D"
                    yield DecoratedG4Tree(n,tuple(edges),tuple(colors),tuple((0,0) for _ in range(n-1)))
    o=_panel_from_parents(name=s["name"],n=n,parents=gen(),candidate_depths=s["candidate_depths"],expected_raw=s["raw_action_instances_expected"],new_H=s["new_H_class"],new_op=tuple(s["new_edge_operator"]))
    o["scope"]={"underlying_shape_count":23,"public_Q_fibers":s["public_Q_fibers"],"existing_edge_operator":s["existing_edge_operator"],"freshness":s["freshness"]}
    o["science_sha256"]=canonical_sha256({k:v for k,v in o.items() if k!="science_sha256"});return o


def endpoint_typed_n5_panel() -> dict[str, Any]:
    s=r2_spec()["fresh_panel_B"]; n=int(s["n"]); shapes=_generate_tree_shapes(n)[n]
    if len(shapes)!=3: raise G5R2Error("n5 tree count mismatch")
    pal=[tuple(map(int,x)) for x in s["existing_edge_operator_palette"]]
    def gen():
        for edges in shapes.values():
            for colors in product(tuple(s["vertex_H_palette"]),repeat=n):
                for grades in product(pal,repeat=n-1):
                    yield DecoratedG4Tree(n,tuple(edges),tuple(colors),tuple(grades))
    o=_panel_from_parents(name=s["name"],n=n,parents=gen(),candidate_depths=s["candidate_depths"],expected_raw=s["raw_action_instances_expected"],new_H=s["new_H_class"],new_op=tuple(s["new_edge_operator"]))
    o["scope"]={"underlying_shape_count":3,"vertex_H_palette":s["vertex_H_palette"],"edge_operator_palette":s["existing_edge_operator_palette"],"decoration_rule":s["decoration_rule"]}
    o["science_sha256"]=canonical_sha256({k:v for k,v in o.items() if k!="science_sha256"});return o


def all_operator_endpoint_sentinel() -> dict[str, Any]:
    ad=G4AcceptedAdapter(); ops=list(ad.operator_basis()); failures=[]; rows=[]
    if len(ops)!=31: failures.append("OPERATOR_BASIS_SIZE")
    for op in ops:
        a,b=op
        t1=DecoratedG4Tree(2,((0,1),),("C","D"),(op,))
        rev=(b,a)
        if rev not in set(ops): failures.append(f"REVERSED_OPERATOR_NOT_ADMITTED:{op}"); continue
        t2=DecoratedG4Tree(2,((1,0),),("C","D"),(rev,))
        c1=ad.unrooted_canon(t1);c2=ad.unrooted_canon(t2)
        if c1!=c2: failures.append(f"STORAGE_REVERSAL_CANON:{op}")
        q1,q2=_q_wire(ad,t1),_q_wire(ad,t2)
        if q1!=q2: failures.append(f"STORAGE_REVERSAL_PUBLIC:{op}")
        for root in (0,1):
            if ad.rooted_canon(t1,root)!=ad.rooted_canon(t2,root): failures.append(f"ROOTED_STORAGE_REVERSAL:{op}:{root}")
            ch1=_graft_child_canon(ad,t1,root,"C",(0,0));ch2=_graft_child_canon(ad,t2,root,"C",(0,0))
            if ch1!=ch2: failures.append(f"GRAFT_STORAGE_REVERSAL:{op}:{root}")
        mismatch_distinguished=None
        if a!=b:
            tm=DecoratedG4Tree(2,((0,1),),("C","D"),((b,a),))
            mismatch_distinguished=(ad.unrooted_canon(t1)!=ad.unrooted_canon(tm))
            if not mismatch_distinguished: failures.append(f"ASYMMETRIC_ENDPOINT_ASSIGNMENT_COLLAPSED:{op}")
        rows.append({"operator":[a,b],"storage_reversal_exact_equal":c1==c2,"public_equal":q1==q2,"asymmetric_endpoint_assignment_distinguished":mismatch_distinguished})
    out={"schema_id":"IG_G5_R2_ALL_31_OPERATOR_ENDPOINT_SENTINEL_V1","status":"PASS" if not failures else "FAIL","operator_count":len(ops),"rows":rows,"failures":failures,"endpoint_direction_semantics_preserved":not failures}
    out["science_sha256"]=canonical_sha256(out)
    if failures: raise G5R2Error("all-operator sentinel failed: "+",".join(failures[:5]))
    return out


def run_g5_r2_predictive_hidden_read(*, engine: Any, r1_primary: Mapping[str,Any], r1_verification: Mapping[str,Any], r1_closeout: Mapping[str,Any]) -> dict[str, Any]:
    s=r2_spec(); auth=verify_authority(r1_primary,r1_verification,r1_closeout)
    lower=global_fixed_depth_lower_bound(max(s["global_lower_bound_target"]["machine_support_depths"]))
    panelA=fresh_n8_color_panel(); panelB=endpoint_typed_n5_panel(); sentinel=all_operator_endpoint_sentinel()
    # Outcome is determined solely by preregistered predicates.  A compression
    # candidate is not promoted even if one survived a bounded panel.
    panel_status={
        "fresh_n8_first_predictive_depth":panelA["first_predictive_depth"],
        "fresh_n8_nontrivial_reduction":panelA["nontrivial_reduction_candidate_survived"],
        "endpoint_n5_first_predictive_depth":panelB["first_predictive_depth"],
        "endpoint_n5_nontrivial_reduction":panelB["nontrivial_reduction_candidate_survived"],
    }
    any_reduction=bool(panelA["nontrivial_reduction_candidate_survived"] or panelB["nontrivial_reduction_candidate_survived"])
    if any_reduction:
        classification="G5_R2_GLOBAL_NO_FIXED_LOCAL_DEPTH_LOWER_BOUND_EARNED_BOUNDED_LOCAL_COMPRESSION_CANDIDATE_SURVIVED_NO_PROMOTION_R3_DESIGN_AUTHORIZED"
    else:
        classification="G5_R2_GLOBAL_NO_FIXED_LOCAL_DEPTH_LOWER_BOUND_EARNED_FRESH_N8_AND_ENDPOINT_TYPED_N5_REQUIRE_COMPLETE_ROOTED_INFORMATION_NO_NONTRIVIAL_LOCAL_COMPRESSION_EARNED_R3_DESIGN_AUTHORIZED"
    out={
        "schema_id":"IG_G5_R2_PREDICTIVE_HIDDEN_ACTION_READ_RESULT_V1","status":"PASS","stage_ref":"G5:R2","classification":classification,
        "promotion":False,"g5_graduation_preserved":True,"g6_started":False,"authority":auth,"frozen_observer_system":s["frozen_observer_system"],
        "global_fixed_depth_lower_bound":lower,"fresh_n8_color_panel":panelA,"endpoint_typed_n5_panel":panelB,"all_operator_endpoint_sentinel":sentinel,
        "panel_status":panel_status,"compact_hidden_read_promoted":False,"topology_promoted":False,"public_descriptor_changed":False,
        "exact_reference_hidden_read":"ENDPOINT_TYPED_DECORATED_TREE_CANON","next_authorized_stage":"G5:R3_DESIGN_AFTER_HUMAN_REVIEW",
        "cost_control":s["cost_control"],"nonclaims":s["nonclaims"]
    }
    out["science_sha256"]=canonical_sha256(out);return out


def stable_payload(result: Mapping[str,Any])->dict[str,Any]:
    return {k:v for k,v in result.items() if k not in {"source_sha256","source_version","registry_sha256","execution_metadata"}}


def compare_cold_replay(primary: Mapping[str,Any], cold: Mapping[str,Any])->dict[str,Any]:
    p=canonical_sha256(stable_payload(primary));c=canonical_sha256(stable_payload(cold));fail=[]
    if primary.get("science_sha256")!=cold.get("science_sha256"): fail.append("SCIENCE_SHA")
    if primary.get("classification")!=cold.get("classification"): fail.append("CLASSIFICATION")
    if p!=c: fail.append("STABLE_PAYLOAD")
    out={"schema_id":"IG_G5_R2_COLD_REPLAY_COMPARISON_V1","status":"PASS" if not fail else "FAIL","certification":"CERTIFIED_PASS" if not fail else "CERTIFICATION_FAILED","failures":fail,"primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"stable_scientific_payload_exact_equal":p==c,"stable_scientific_payload_sha256":p}
    out["comparison_sha256"]=canonical_sha256(out);return out


def certified_closeout(primary: Mapping[str,Any], cold: Mapping[str,Any], comparison: Mapping[str,Any], independent: Mapping[str,Any])->dict[str,Any]:
    ok=primary.get("status")=="PASS" and cold.get("status")=="PASS" and comparison.get("certification")=="CERTIFIED_PASS" and independent.get("status")=="PASS"
    out={"schema_id":"IG_G5_R2_CERTIFIED_CLOSEOUT_V1","status":"CERTIFIED_PASS" if ok else "CERTIFICATION_FAILED","experiment_id":"G5:R2.PREDICTIVE_HIDDEN_ACTION_READ","classification":primary.get("classification") if ok else "G5_R2_REVIEW_REQUIRED_NO_PROMOTION","science_sha256":primary.get("science_sha256"),"stable_science_payload_sha256":comparison.get("stable_scientific_payload_sha256"),"comparison_sha256":comparison.get("comparison_sha256"),"independent_verification_sha256":independent.get("verification_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),"promotion":False,"g5_graduation_preserved":True,"compact_hidden_read_promoted":False,"topology_promoted":False,"g6_started":False,"next_authorized_stage":"G5:R3_DESIGN_AFTER_HUMAN_REVIEW" if ok else None,"failures":comparison.get("failures",[])}
    out["closeout_sha256"]=canonical_sha256(out);return out
