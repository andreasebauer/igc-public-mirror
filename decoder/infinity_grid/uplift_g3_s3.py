from __future__ import annotations

"""Registered G3:S3 higher-order compatibility / irreducibility audit.

The S3 observer is deliberately finite and non-promoting.  It asks whether connected
simple three-unit P3/K3 compatibility facts contain any residual not reconstructible from
(1) the certified S2 pair-observer quotient and (2) inherited CAPS7 on each G2 unit.

Because P3/K3 have maximum vertex degree two, the only possible new local compatibility
information is the branch-sensitive effect of consuming two incident endpoint reservations
on the same whole G2 unit.  S3 therefore certifies the complete ordered 7x7 two-reservation
continuation table for every exact representative in the frozen S0 collision class.  If that
table is representative-independent and the S2 evidence binding replays exactly, the declared
triple observer factors through CAPS7 + S2 without reading hidden G1-unit incidence topology.
"""
from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip, json

from .canon import canonical_sha256
from .g2_relation import reserve_external_relation
from .uplift_g3_s0 import phase0_spec
from .uplift_g3_s2 import load_s1_signature_index

class G3S3Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s3_spec() -> dict[str, Any]:
    obj=json.loads(_resource("G3_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id")!="IG_G3_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_SPEC_V1":
        raise G3S3Error("bad G3:S3 spec schema")
    payload={k:v for k,v in obj.items() if k!="science_sha256"}
    if canonical_sha256(payload)!=obj.get("science_sha256"):
        raise G3S3Error("G3:S3 spec hash mismatch")
    if obj.get("phase0_spec_sha256")!=phase0_spec().get("science_sha256"):
        raise G3S3Error("G3:S3/Phase0 binding mismatch")
    return obj


def _caps7(st: Any) -> list[int]:
    caps=[int(x) for x in st.total_caps]
    if len(caps)!=7 or any(x<0 for x in caps):
        raise G3S3Error("malformed CAPS7 in G3:S3")
    return caps


def _caps7_hash(caps: Sequence[int]) -> str:
    obj={"schema_id":"IG_G2_CAPS7_STATE_V1","coordinates":[int(x) for x in caps]}
    return canonical_sha256(obj)


def ordered_two_reservation_signature(state: Any, first_type: int, second_type: int) -> dict[str, Any]:
    """Exact branch-sensitive CAPS7-only local continuation for two ordered reservations."""
    a,b=int(first_type),int(second_type)
    if not (0<=a<7 and 0<=b<7): raise G3S3Error("endpoint type outside seven-type alphabet")
    firsts=reserve_external_relation(state,a)
    hist=Counter(); payload_by_hash={}; total_second=0
    for first in firsts:
        seconds=reserve_external_relation(first,b)
        total_second += len(seconds)
        bp={
          "first_caps7":_caps7(first),
          "second_relation_cardinality":len(seconds),
          "second_successor_caps7_set":[list(x) for x in sorted({tuple(_caps7(s)) for s in seconds})],
        }
        h=canonical_sha256(bp); payload_by_hash.setdefault(h,bp); hist[h]+=1
    first_sig={
      "endpoint_type":a,
      "relation_cardinality":len(firsts),
      "successor_caps7_set":[list(x) for x in sorted({tuple(_caps7(s)) for s in firsts})],
    }
    out={
      "schema_id":"IG_G3_S3_ORDERED_TWO_RESERVATION_SIGNATURE_V1",
      "first_reserved_type":a,"second_reserved_type":b,
      "first_step":first_sig,
      "first_branch_profile_histogram":[{"branch_profile_sha256":h,"multiplicity":int(hist[h]),"profile":payload_by_hash[h]} for h in sorted(hist)],
      "final_relation_output_cardinality":int(total_second),
      "joint_legal":bool(total_second>0),
      "direct_topology_read":False,"construction_identity_recorded":False,
    }
    out["science_sha256"]=canonical_sha256(out)
    return out


def verify_s3_authority(*, s0_result: Mapping[str,Any], s1_result: Mapping[str,Any], s2_result: Mapping[str,Any]) -> dict[str,Any]:
    a=s3_spec()["authority"]; failures=[]
    checks=[
      (s0_result.get("status")==a["g3_s0_status"],"S0_STATUS"),(s0_result.get("classification")==a["g3_s0_classification"],"S0_CLASS"),(s0_result.get("science_sha256")==a["g3_s0_science_sha256"],"S0_SHA"),
      (s1_result.get("status")==a["g3_s1_status"],"S1_STATUS"),(s1_result.get("classification")==a["g3_s1_classification"],"S1_CLASS"),(s1_result.get("outcome")==a["g3_s1_outcome"],"S1_OUTCOME"),(s1_result.get("science_sha256")==a["g3_s1_science_sha256"],"S1_SHA"),
      (s2_result.get("status")==a["g3_s2_status"],"S2_STATUS"),(s2_result.get("classification")==a["g3_s2_classification"],"S2_CLASS"),(s2_result.get("science_sha256")==a["g3_s2_science_sha256"],"S2_SHA"),
      (bool(s2_result.get("g3_s3_unlocked")) is bool(a["required_g3_s3_unlocked"]),"S3_UNLOCK"),(bool(s2_result.get("topology_promoted")) is bool(a["required_topology_promoted"]),"TOPOLOGY_PROMOTION"),(bool(s2_result.get("g3_graduated")) is bool(a["required_g3_graduated"]),"G3_GRADUATION"),
      (int(s2_result.get("quotient_audit",{}).get("public_pair_context_key_count",-1))==int(a["required_s2_public_pair_context_keys"]),"S2_PUBLIC_KEYS"),
      (int(s2_result.get("quotient_audit",{}).get("representatives_per_public_context_key_expected",-1))==int(a["required_s2_representatives_per_key"]),"S2_REPS"),
      (int(s2_result.get("quotient_audit",{}).get("representative_conflict_count",-1))==int(a["required_s2_conflicts"]),"S2_CONFLICTS"),
    ]
    failures.extend(name for ok,name in checks if not ok)
    if failures: raise G3S3Error("G3:S3 authority verification failed: "+",".join(failures))
    out={"schema_id":"IG_G3_S3_AUTHORITY_V1","status":"PASS","g3_s0_science_sha256":s0_result["science_sha256"],"g3_s1_science_sha256":s1_result["science_sha256"],"g3_s2_science_sha256":s2_result["science_sha256"],"g3_phase0_science_sha256":phase0_spec()["science_sha256"],"g3_s3_spec_sha256":s3_spec()["science_sha256"]}
    out["science_sha256"]=canonical_sha256(out); return out


def verify_s2_signature_binding(*, s0_result:Mapping[str,Any], s2_result:Mapping[str,Any], signature_index:Mapping[str,Any]) -> dict[str,Any]:
    """Replay S2 observer hashes exactly against the complete certified S1 index."""
    challenge=list(s0_result.get("challenge_corpus",{}).get("records",[]))
    if len(challenge)!=6: raise G3S3Error("S3 expects six S0 challenge carriers")
    caps_by_ref={str(r["g2_carrier_ref"]):str(r["public_interface"]["science_sha256"]) for r in challenge}
    rows=list(signature_index.get("rows",[]))
    if len(rows)!=2232: raise G3S3Error("S3 requires exact 2232-row S1 signature index")
    grouped=defaultdict(set); counts=Counter()
    for row in rows:
        tr,cr=str(row.get("target_ref")),str(row.get("context_ref"))
        if tr not in caps_by_ref or cr not in caps_by_ref: raise G3S3Error("S1 signature row outside S0 corpus")
        op=tuple(map(int,row.get("operator",[]))); ori=str(row.get("orientation"))
        sig=row.get("operational_signature",{}); sh=str(sig.get("science_sha256"))
        if canonical_sha256({k:v for k,v in sig.items() if k!="science_sha256"})!=sh: raise G3S3Error("S1 signature hash mismatch")
        key=(caps_by_ref[tr],caps_by_ref[cr],op,ori); grouped[key].add(sh); counts[key]+=1
    qrows=list(s2_result.get("quotient_table",[])); qmap={}
    for q in qrows:
        key=(str(q["target_caps7_sha256"]),str(q["context_caps7_sha256"]),tuple(map(int,q["operator"])),str(q["orientation"]))
        if key in qmap: raise G3S3Error("duplicate S2 quotient key")
        qmap[key]=str(q["observer_signature_sha256"])
    failures=[]
    for key in sorted(grouped,key=str):
        if counts[key]!=36 or len(grouped[key])!=1 or qmap.get(key)!=next(iter(grouped[key])):
            failures.append({"key":[key[0],key[1],list(key[2]),key[3]],"representatives":counts[key],"s1_values":sorted(grouped[key]),"s2_value":qmap.get(key)})
    if len(grouped)!=62 or set(grouped)!=set(qmap): failures.append({"failure":"KEY_SET","s1":len(grouped),"s2":len(qmap)})
    out={"schema_id":"IG_G3_S3_S2_SIGNATURE_BINDING_V1","status":"PASS" if not failures else "FAIL","s1_exact_row_count":len(rows),"public_pair_context_key_count":len(grouped),"representatives_per_key":36,"conflict_count":len(failures),"conflict_examples":failures[:16]}
    out["science_sha256"]=canonical_sha256(out)
    if failures: raise G3S3Error("G3:S3 S2/S1 signature binding failed")
    return out


def certify_caps7_two_reservation_sufficiency(*, s0_result:Mapping[str,Any], task_rows:Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    challenge=list(s0_result.get("challenge_corpus",{}).get("records",[]))
    caps_by_ref={str(r["g2_carrier_ref"]):str(r["public_interface"]["science_sha256"]) for r in challenge}
    topo_by_ref={str(r["g2_carrier_ref"]):str(r["hidden_challenge_diagnostic"]["topology_canon"]) for r in challenge}
    expected=len(caps_by_ref)*49
    if len(task_rows)!=expected: raise G3S3Error(f"expected {expected} local continuation rows, got {len(task_rows)}")
    seen=set(); grouped=defaultdict(list); full_vectors=defaultdict(dict)
    for row in task_rows:
        ref=str(row["carrier_ref"]); a=int(row["first_type"]); b=int(row["second_type"])
        if ref not in caps_by_ref or not (0<=a<7 and 0<=b<7): raise G3S3Error("bad local continuation row")
        key=(ref,a,b)
        if key in seen: raise G3S3Error("duplicate local continuation row")
        seen.add(key)
        sig=row["signature"]; sh=str(sig.get("science_sha256"))
        if canonical_sha256({k:v for k,v in sig.items() if k!="science_sha256"})!=sh: raise G3S3Error("local continuation signature hash mismatch")
        grouped[(caps_by_ref[ref],a,b)].append((ref,sh)); full_vectors[ref][(a,b)]=sh
    conflicts=[]
    for (caps,a,b), vals in sorted(grouped.items(),key=str):
        hs=sorted({h for _,h in vals})
        if len(vals)!=len(caps_by_ref) or len(hs)!=1:
            conflicts.append({"caps7_sha256":caps,"first_type":a,"second_type":b,"representative_count":len(vals),"signature_count":len(hs),"signature_sha256s":hs,"carrier_refs":[r for r,_ in vals]})
    vectors=[]
    for ref in sorted(caps_by_ref):
        vec=[full_vectors[ref][(a,b)] for a in range(7) for b in range(7)]
        vectors.append({"carrier_ref":ref,"topology_label":topo_by_ref[ref],"continuation_vector_sha256":canonical_sha256(vec)})
    vector_hashes=sorted({x["continuation_vector_sha256"] for x in vectors})
    if len(vector_hashes)!=1: conflicts.append({"failure":"FULL_VECTOR_CONFLICT","vector_hashes":vector_hashes})
    out={
      "schema_id":"IG_G3_S3_CAPS7_TWO_RESERVATION_SUFFICIENCY_V1","status":"PASS" if not conflicts else "FAIL",
      "exact_carrier_count":len(caps_by_ref),"caps7_class_count":len(set(caps_by_ref.values())),"hidden_topology_class_count":len(set(topo_by_ref.values())),
      "ordered_endpoint_pair_count":49,"task_row_count":len(task_rows),"continuation_conflict_count":len(conflicts),"conflict_examples":conflicts[:16],
      "all_equal_caps7_representatives_share_one_complete_ordered_two_reservation_table":not conflicts,
      "representative_vectors":vectors,
      "topology_used_only_after_public_sufficiency_test":True,
    }
    out["science_sha256"]=canonical_sha256(out); return out


def triple_reconstruction_coverage(*, pair_key_count:int, exact_unit_representatives:int, local_sufficiency_pass:bool, pair_binding_pass:bool) -> dict[str,Any]:
    e=int(pair_key_count); n=int(exact_unit_representatives)
    p3=e**2; k3=e**3; reps=n**3
    passed=bool(local_sufficiency_pass and pair_binding_pass and e==62 and n==6)
    out={
      "schema_id":"IG_G3_S3_TRIPLE_RECONSTRUCTION_COVERAGE_V1","status":"PASS" if passed else "FAIL",
      "edge_public_context_count":e,"exact_unit_representatives":n,"independent_exact_triples_per_public_context":reps,
      "public_context_counts":{"P3_CONNECTED_PATH":p3,"K3_TRIANGLE":k3,"TOTAL":p3+k3},
      "exact_representative_assignment_counts":{"P3_CONNECTED_PATH":p3*reps,"K3_TRIANGLE":k3*reps,"TOTAL":(p3+k3)*reps},
      "max_vertex_degree":2,
      "factorization_argument":"Each edge is fixed by the certified S2 pair observer. Each vertex has degree at most two, and its complete ordered two-reservation branch-sensitive CAPS7 continuation is fixed by its CAPS7 class. Therefore the declared joint P3/K3 public observer is a deterministic function of CAPS7 vertex labels plus S2 edge values.",
      "enumeration_note":"Counts are exact combinatorial coverage of the frozen public edge basis and independent exact unit instances; the scientific reduction is by local factorization, not extrapolation from sampled triples.",
    }
    out["science_sha256"]=canonical_sha256(out); return out


def finalize_s3_result(*, s0_result:Mapping[str,Any], s1_result:Mapping[str,Any], s2_result:Mapping[str,Any], signature_index:Mapping[str,Any], task_rows:Sequence[Mapping[str,Any]], reproduction:Mapping[str,Any]) -> dict[str,Any]:
    spec=s3_spec(); auth=verify_s3_authority(s0_result=s0_result,s1_result=s1_result,s2_result=s2_result)
    binding=verify_s2_signature_binding(s0_result=s0_result,s2_result=s2_result,signature_index=signature_index)
    local=certify_caps7_two_reservation_sufficiency(s0_result=s0_result,task_rows=task_rows)
    cov=triple_reconstruction_coverage(pair_key_count=binding["public_pair_context_key_count"],exact_unit_representatives=local["exact_carrier_count"],local_sufficiency_pass=local["status"]=="PASS",pair_binding_pass=binding["status"]=="PASS")
    passed=local["status"]=="PASS" and binding["status"]=="PASS" and cov["status"]=="PASS"
    result={
      "schema_id":"IG_G3_S3_HIGHER_ORDER_COMPATIBILITY_IRREDUCIBILITY_RESULT_V1","status":"PASS" if passed else "REVIEW_REQUIRED","stage_ref":"G3:S3",
      "classification":"G3_NO_IRREDUCIBLE_TRIPLE_RESIDUAL_ON_REGISTERED_P3_K3_CAPS7_S2_SCOPE_S4_UNLOCKED" if passed else "G3_HIGHER_ORDER_RESIDUAL_OR_BINDING_FAILURE_S4_LOCKED",
      "authority":auth,"challenge_reproduction":dict(reproduction),"s2_signature_binding":binding,"caps7_two_reservation_sufficiency":local,"triple_reconstruction":cov,
      "irreducible_residual_count":0 if passed else None,"minimal_added_read_at_tested_triple_observer":"NONE" if passed else None,
      "topology_promoted":False,"g3_s4_unlocked":passed,"g3_graduated":False,"next_authorized_stage":"G3:S4" if passed else None,
      "scope":"EXACT_FROZEN_G3_S0_SIX_UNIT_COLLISION_DOMAIN__COMPLETE_62_EDGE_PAIR_OBSERVER_BASIS__CONNECTED_SIMPLE_P3_K3_TRIPLE_OBSERVER",
      "nonclaims":list(spec["nonclaims"]),
    }
    result["science_sha256"]=canonical_sha256(result); return result
