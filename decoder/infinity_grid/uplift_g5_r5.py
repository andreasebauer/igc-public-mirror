from __future__ import annotations

"""G5:R5 marker-free one-step exact-hidden separation audit.

R3 earned exact hidden future congruence; R4 earned scoped necessity using a
fresh marker.  R5 removes that marker trick and asks a bounded, actual-carrier
question: within fixed public Q fibers, does the complete certified one-step
leaf-graft action signature over H in {C,D} and all 31 directed operators
already separate exact hidden parents?

Both scientific outcomes are legitimate.  Injectivity earns only a bounded
marker-free separation result.  A collision earns an exact insufficiency
witness.  Neither outcome promotes hidden topology or changes public G5.
"""

from collections import defaultdict
from importlib.resources import files
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .uplift_g3_r1 import _generate_tree_shapes


class G5R5Error(RuntimeError):
    pass

_SPEC = "G5_R5_MARKER_FREE_ONE_STEP_SEPARATION_SPEC_V1.json"


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def r5_spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_R5_MARKER_FREE_ONE_STEP_SEPARATION_SPEC_V1":
        raise G5R5Error("bad G5:R5 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G5R5Error("G5:R5 spec hash mismatch")
    return obj


def verify_authority(r4_primary: Mapping[str, Any], r4_verification: Mapping[str, Any], r4_closeout: Mapping[str, Any]) -> dict[str, Any]:
    s = r5_spec(); a = s["authority"]; failures: list[str] = []
    if r4_primary.get("schema_id") != "IG_G5_R4_MARKED_LEAF_HIDDEN_MINIMALITY_RESULT_V1" or r4_primary.get("status") != "PASS":
        failures.append("R4_PRIMARY_SCHEMA_OR_STATUS")
    if str(r4_primary.get("science_sha256")) != str(a["g5_r4_primary_science_sha256"]): failures.append("R4_PRIMARY_IDENTITY")
    if r4_primary.get("scoped_hidden_minimality_earned") is not True: failures.append("R4_MINIMALITY_AUTHORITY")
    if r4_primary.get("promotion") is not False or r4_primary.get("g5_graduation_preserved") is not True: failures.append("R4_FIREWALL")
    if r4_verification.get("status") != "PASS" or str(r4_verification.get("verification_sha256")) != str(a["g5_r4_independent_verification_sha256"]): failures.append("R4_VERIFICATION")
    if r4_closeout.get("status") != "CERTIFIED_PASS" or str(r4_closeout.get("closeout_sha256")) != str(a["g5_r4_closeout_sha256"]): failures.append("R4_CLOSEOUT")
    out = {"schema_id":"IG_G5_R5_AUTHORITY_CHECK_V1","status":"PASS" if not failures else "FAIL","failures":failures,"r4_primary_science_sha256":r4_primary.get("science_sha256"),"r4_verification_sha256":r4_verification.get("verification_sha256"),"r4_closeout_sha256":r4_closeout.get("closeout_sha256")}
    out["science_sha256"] = canonical_sha256(out)
    if failures: raise G5R5Error("R4 authority mismatch: " + ",".join(failures))
    return out


def _q_key(pub: Mapping[str, Any]) -> tuple[Any, ...]:
    if not pub.get("legal") or pub.get("descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        raise G5R5Error("illegal or wrong public parent in Q key")
    return (tuple(map(int, pub["caps7"])), tuple(sorted((str(k), int(v)) for k,v in pub["H_class_bag"].items())))


def _relation_child_canons(ad: G4AcceptedAdapter, tree: DecoratedG4Tree, new_H: str, op: Sequence[int]) -> tuple[tuple[Any, ...], ...]:
    out: set[tuple[Any, ...]] = set()
    for root in range(tree.n):
        for child in ad.graft_relation(tree, root, new_H_class=str(new_H), operator=tuple(map(int,op))):
            out.add(ad.unrooted_canon(child))
    return tuple(sorted(out))


def _full_one_step_signature(ad: G4AcceptedAdapter, tree: DecoratedG4Tree) -> tuple[Any, ...]:
    rows = []
    for new_H in ("C", "D"):
        for op in ad.operator_basis():
            rows.append((new_H, tuple(op), _relation_child_canons(ad, tree, new_H, op)))
    return tuple(rows)


def _tree_record(tree: DecoratedG4Tree) -> dict[str, Any]:
    return {"n":int(tree.n),"edges":[list(map(int,e)) for e in tree.edges],"H_classes":list(tree.H_classes),"edge_operators":[list(map(int,x)) for x in tree.edge_operators]}


def _audit_parent_set(exact_parents: Mapping[tuple[Any,...], DecoratedG4Tree], *, panel_id: str, raw_parent_count: int, rejected_illegal_count: int) -> dict[str, Any]:
    ad = G4AcceptedAdapter(); failures: list[str] = []; by_q: dict[tuple[Any,...], list[tuple[tuple[Any,...],DecoratedG4Tree]]] = defaultdict(list)
    for pcan, tree in exact_parents.items():
        pub = ad.public_read(tree)
        if not pub.get("legal"): failures.append("ILLEGAL_PARENT_AFTER_FILTER"); continue
        by_q[_q_key(pub)].append((pcan,tree))

    q_rows=[]; first_collision=None; total_signature_rows=0; total_empty_payload_entries=0; max_relation_children=0
    for q in sorted(by_q):
        owners: dict[tuple[Any,...], tuple[tuple[Any,...],DecoratedG4Tree]] = {}
        collision_count=0
        for pcan,tree in sorted(by_q[q], key=lambda x:x[0]):
            sig=_full_one_step_signature(ad,tree); total_signature_rows += 1
            empty=sum(1 for _,_,rel in sig if not rel); total_empty_payload_entries += empty
            max_relation_children=max(max_relation_children,max((len(rel) for _,_,rel in sig), default=0))
            old=owners.get(sig)
            if old is None: owners[sig]=(pcan,tree)
            elif old[0] != pcan:
                collision_count += 1
                if first_collision is None:
                    first_collision={
                      "Q":{"caps7":list(q[0]),"H_class_bag":[list(x) for x in q[1]]},
                      "parent_a":_tree_record(old[1]),"parent_b":_tree_record(tree),
                      "parent_a_exact_canon":old[0],"parent_b_exact_canon":pcan,
                      "common_full_one_step_signature":sig,
                    }
        q_rows.append({"Q_caps7":list(q[0]),"Q_H_class_bag":[list(x) for x in q[1]],"exact_parent_count":len(by_q[q]),"distinct_one_step_signature_count":len(owners),"collision_count":collision_count})
    injective = first_collision is None
    out={
      "schema_id":"IG_G5_R5_MARKER_FREE_PANEL_RESULT_V1","panel_id":panel_id,"status":"PASS" if not failures else "FAIL",
      "raw_parent_count":int(raw_parent_count),"rejected_illegal_parent_count":int(rejected_illegal_count),"exact_parent_canon_count":len(exact_parents),"public_Q_fiber_count":len(by_q),
      "q_rows":q_rows,"full_action_payload_count":62,"evaluated_exact_parent_signatures":total_signature_rows,"empty_payload_relation_entry_count":total_empty_payload_entries,"max_exact_child_canons_in_one_payload_relation":max_relation_children,
      "within_Q_full_one_step_signature_injective":injective,"collision_witness":first_collision,
      "structural_equality_used":True,"digest_only_equality_used":False,"failures":failures,
    }
    out["science_sha256"]=canonical_sha256(out)
    if failures: raise G5R5Error(panel_id+" failed: "+",".join(failures[:10]))
    return out


def fresh_primary_panel() -> dict[str, Any]:
    s=r5_spec()["fresh_primary_panel"]; ad=G4AcceptedAdapter(); shapes=_generate_tree_shapes(int(s["n_max"])); raw=0; illegal=0; exact: dict[tuple[Any,...],DecoratedG4Tree]={}; rows=[]
    for n in range(int(s["n_min"]),int(s["n_max"])+1):
        nr=ni=0; before=len(exact)
        for edges in shapes[n].values():
            for colors in product(tuple(s["vertex_H_palette"]),repeat=n):
                raw+=1;nr+=1
                t=DecoratedG4Tree(n,tuple(edges),tuple(colors),tuple((0,0) for _ in range(n-1)))
                pub=ad.public_read(t)
                if not pub.get("legal"): illegal+=1;ni+=1;continue
                exact.setdefault(ad.unrooted_canon(t),t)
        rows.append({"n":n,"unlabelled_shape_count":len(shapes[n]),"raw_decorated_parent_count":nr,"rejected_illegal_count":ni,"new_exact_parent_canons":len(exact)-before})
    out=_audit_parent_set(exact,panel_id="FRESH_N4_N7_BINARY_H_FIXED_00_EDGES",raw_parent_count=raw,rejected_illegal_count=illegal)
    out["generation_rows"]=rows; out["science_sha256"]=canonical_sha256({k:v for k,v in out.items() if k!="science_sha256"}); return out


def endpoint_typed_sentinel_panel() -> dict[str, Any]:
    s=r5_spec()["endpoint_typed_sentinel_panel"]; ad=G4AcceptedAdapter(); ops=tuple(ad.operator_basis()); shapes=_generate_tree_shapes(3); raw=0;illegal=0;exact:dict[tuple[Any,...],DecoratedG4Tree]={}
    if len(ops)!=31: raise G5R5Error("operator basis size != 31")
    for edges in shapes[3].values():
        for colors in product(tuple(s["vertex_H_palette"]),repeat=3):
            for edge_ops in product(ops,repeat=2):
                raw+=1;t=DecoratedG4Tree(3,tuple(edges),tuple(colors),tuple(edge_ops));pub=ad.public_read(t)
                if not pub.get("legal"): illegal+=1;continue
                exact.setdefault(ad.unrooted_canon(t),t)
    out=_audit_parent_set(exact,panel_id="N3_FULL_ENDPOINT_TYPED_PARENT_SENTINEL",raw_parent_count=raw,rejected_illegal_count=illegal)
    out["all_31_parent_edge_operators_covered"]=True;out["all_62_action_payloads_covered"]=True;out["science_sha256"]=canonical_sha256({k:v for k,v in out.items() if k!="science_sha256"});return out


def run_g5_r5_marker_free_one_step_separation(*, engine: Any, r4_primary: Mapping[str,Any], r4_verification: Mapping[str,Any], r4_closeout: Mapping[str,Any]) -> dict[str,Any]:
    s=r5_spec();auth=verify_authority(r4_primary,r4_verification,r4_closeout);primary=fresh_primary_panel();sentinel=endpoint_typed_sentinel_panel()
    injective=bool(primary["within_Q_full_one_step_signature_injective"] and sentinel["within_Q_full_one_step_signature_injective"])
    classification=s["pass_classifications"]["injective" if injective else "collision"]
    witness=primary.get("collision_witness") or sentinel.get("collision_witness")
    out={
      "schema_id":"IG_G5_R5_MARKER_FREE_ONE_STEP_SEPARATION_RESULT_V1","status":"PASS","stage_ref":"G5:R5","classification":classification,"promotion":False,"g5_graduation_preserved":True,"g6_started":False,
      "authority":auth,"fresh_primary_panel":primary,"endpoint_typed_sentinel_panel":sentinel,
      "marker_free_one_step_separation_earned_on_frozen_scope":injective,"marker_free_one_step_collision_earned_on_frozen_scope":not injective,"first_collision_witness":witness,
      "global_marker_free_minimality_claim":False,"all_finite_continuation_separation_claim":False,"exact_hidden_canon_promoted_to_public":False,"topology_promoted":False,"public_descriptor_changed":False,
      "next_authorized_stage":s["next_authorized_stage"],"cost_control":s["cost_control"],"nonclaims":s["nonclaims"]
    }
    out["science_sha256"]=canonical_sha256(out);return out


def stable_payload(result: Mapping[str,Any])->dict[str,Any]:
    return {k:v for k,v in result.items() if k not in {"source_sha256","source_version","registry_sha256","execution_metadata"}}


def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    ps=canonical_sha256(stable_payload(primary));cs=canonical_sha256(stable_payload(cold));fail=[]
    if primary.get("science_sha256")!=cold.get("science_sha256"):fail.append("SCIENCE_SHA")
    if primary.get("classification")!=cold.get("classification"):fail.append("CLASSIFICATION")
    if ps!=cs:fail.append("STABLE_PAYLOAD")
    out={"schema_id":"IG_G5_R5_COLD_REPLAY_COMPARISON_V1","status":"PASS" if not fail else "FAIL","certification":"CERTIFIED_PASS" if not fail else "CERTIFICATION_FAILED","failures":fail,"primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"stable_scientific_payload_exact_equal":ps==cs,"stable_scientific_payload_sha256":ps};out["comparison_sha256"]=canonical_sha256(out);return out


def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],comparison:Mapping[str,Any],independent:Mapping[str,Any])->dict[str,Any]:
    ok=primary.get("status")=="PASS" and cold.get("status")=="PASS" and comparison.get("certification")=="CERTIFIED_PASS" and independent.get("status")=="PASS"
    out={"schema_id":"IG_G5_R5_CERTIFIED_CLOSEOUT_V1","status":"CERTIFIED_PASS" if ok else "CERTIFICATION_FAILED","experiment_id":"G5:R5.MARKER_FREE_ONE_STEP_SEPARATION","classification":primary.get("classification") if ok else "G5_R5_REVIEW_REQUIRED_NO_PROMOTION","science_sha256":primary.get("science_sha256"),"comparison_sha256":comparison.get("comparison_sha256"),"independent_verification_sha256":independent.get("verification_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),"promotion":False,"g5_graduation_preserved":True,"marker_free_one_step_separation_earned_on_frozen_scope":bool(primary.get("marker_free_one_step_separation_earned_on_frozen_scope")) if ok else False,"marker_free_one_step_collision_earned_on_frozen_scope":bool(primary.get("marker_free_one_step_collision_earned_on_frozen_scope")) if ok else False,"global_marker_free_minimality_claim":False,"topology_promoted":False,"g6_started":False,"next_authorized_stage":"G5:R6_DESIGN_AFTER_HUMAN_REVIEW" if ok else None,"failures":comparison.get("failures",[])};out["closeout_sha256"]=canonical_sha256(out);return out
