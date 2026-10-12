from __future__ import annotations

"""Registered G4:S4 one-step relation-valued composition-closure audit.

This stage is deliberately bounded to the two certified G3 construction terms frozen by
G4:S0 and the 31 directed G4 bridge operators.  It reuses complete G4 pair relations as
inputs to one further binary composition with a third certified G3 term.  No pair branch,
hidden topology, shell profile, owner witness, ancestry, or construction digest is a public
selector.  PASS unlocks S5 only; it never graduates G4 or promotes Tier 1.
"""

from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g4_term_state import (
    G3TermState, G4PairTermState, G4TripleTermState,
    compose_g4_pair_relation, compose_g4_recursive_triple_relation,
)
from .uplift_g4_s0 import phase0_spec
from .uplift_g4_s1 import load_s0_term_states, pair_context_kernel_measurement, expand_pair_context_kernel_signature
from .uplift_g4_s2 import load_s1_signature_index


class G4S4Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s4_spec() -> dict[str, Any]:
    obj = json.loads(_resource("G4_S4_COMPOSITION_CLOSURE_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G4_S4_COMPOSITION_CLOSURE_SPEC_V1":
        raise G4S4Error("bad G4:S4 spec schema")
    if canonical_sha256({k:v for k,v in obj.items() if k != "science_sha256"}) != obj.get("science_sha256"):
        raise G4S4Error("G4:S4 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G4S4Error("G4:S4/Phase0 binding mismatch")
    return obj


def _caps7(st: G3TermState | G4PairTermState | G4TripleTermState) -> list[int]:
    out = [int(x) for x in st.total_caps]
    if len(out) != 7 or any(x < 0 for x in out):
        raise G4S4Error("malformed CAPS7")
    return out


def _tier1_from_s3(s3: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = s3.get("candidate_interface_index", {}).get("rows", [])
    out = {str(r["term_ref"]): {"candidate_key_sha256": str(r["candidate_key_sha256"]), "caps7": [int(x) for x in r["caps7"]], "tier1_value": dict(r["tier1_value"])} for r in rows}
    if len(out) != 2:
        raise G4S4Error("S3 candidate index malformed")
    return out


def verify_s4_authority(*, s0_result: Mapping[str,Any], s1_result: Mapping[str,Any], s2_result: Mapping[str,Any], s3_result: Mapping[str,Any], s3_replay: Mapping[str,Any], s3_closeout: Mapping[str,Any]) -> dict[str,Any]:
    a=s4_spec()["authority"]; failures=[]
    checks=[
        (s3_result.get("status")==a["g4_s3_status"],"S3_STATUS"),
        (s3_result.get("classification")==a["g4_s3_classification"],"S3_CLASS"),
        (s3_result.get("science_sha256")==a["g4_s3_science_sha256"],"S3_SCIENCE_SHA"),
        (s3_result.get("source_sha256")==a["g4_s3_source_sha256"],"S3_SOURCE_SHA"),
        (s3_result.get("registry_sha256")==a["g4_s3_registry_sha256"],"S3_REGISTRY_SHA"),
        (bool(s3_result.get("g4_s4_unlocked")) is bool(a["required_g4_s4_unlocked"]),"S4_UNLOCK"),
        (bool(s3_result.get("g4_graduated")) is bool(a["required_g4_graduated"]),"G4_GRADUATION"),
        (bool(s3_result.get("public_descriptor_promoted")) is bool(a["required_public_descriptor_promoted"]),"PUBLIC_PROMOTION"),
        (bool(s3_result.get("topology_promoted")) is bool(a["required_topology_promoted"]),"TOPOLOGY_PROMOTION"),
        (bool(s3_result.get("shell_profile_promoted")) is bool(a["required_shell_profile_promoted"]),"SHELL_PROMOTION"),
        (s3_result.get("candidate_interface_index",{}).get("candidate")==a["required_candidate"],"S3_CANDIDATE"),
        (int(s3_result.get("tier1_two_reservation_sufficiency",{}).get("continuation_conflict_count",-1))==int(a["required_s3_local_conflicts"]),"S3_LOCAL_CONFLICTS"),
        (int(s3_result.get("triple_reconstruction",{}).get("public_context_counts",{}).get("P3_CONNECTED_PATH",-1))==int(a["required_s3_p3_context_count"]),"S3_P3_COUNT"),
        (int(s3_result.get("triple_reconstruction",{}).get("public_context_counts",{}).get("K3_TRIANGLE",-1))==int(a["required_s3_k3_context_count"]),"S3_K3_COUNT"),
        (s3_replay.get("schema_id")==a["g4_s3_replay_schema_id"],"S3_REPLAY_SCHEMA"),
        (s3_replay.get("comparison_sha256")==a["g4_s3_replay_comparison_sha256"],"S3_REPLAY_SHA"),
        (s3_replay.get("certification")==a["required_g4_s3_replay_certification"],"S3_REPLAY_CERT"),
        (s3_closeout.get("closeout_sha256")==a["g4_s3_closeout_sha256"],"S3_CLOSEOUT_SHA"),
        (s3_closeout.get("status")=="CERTIFIED_PASS" and s3_closeout.get("next_authorized_stage")=="G4:S4","S3_CLOSEOUT_AUTH"),
        (s0_result.get("science_sha256")==s3_result.get("authority",{}).get("g4_s0_science_sha256"),"S0_CHAIN"),
        (s1_result.get("science_sha256")==s3_result.get("authority",{}).get("g4_s1_science_sha256"),"S1_CHAIN"),
        (s2_result.get("science_sha256")==s3_result.get("authority",{}).get("g4_s2_science_sha256"),"S2_CHAIN"),
    ]
    failures.extend(n for ok,n in checks if not ok)
    if failures:
        raise G4S4Error("G4:S4 authority verification failed: "+",".join(failures))
    out={"schema_id":"IG_G4_S4_AUTHORITY_VERIFICATION_V1","status":"PASS","g4_s0_science_sha256":s0_result["science_sha256"],"g4_s1_science_sha256":s1_result["science_sha256"],"g4_s2_science_sha256":s2_result["science_sha256"],"g4_s3_science_sha256":s3_result["science_sha256"],"g4_s3_replay_comparison_sha256":s3_replay["comparison_sha256"],"g4_s3_closeout_sha256":s3_closeout["closeout_sha256"],"g4_phase0_spec_sha256":phase0_spec()["science_sha256"],"g4_s4_spec_sha256":s4_spec()["science_sha256"],"lower_layer_rematerialization":False,"public_descriptor_promoted":False,"g4_graduated":False}
    out["science_sha256"]=canonical_sha256(out); return out


def ordered_three_reservation_signature(state:G3TermState, a:int,b:int,c:int)->dict[str,Any]:
    a,b,c=map(int,(a,b,c));
    if not all(0<=x<7 for x in (a,b,c)): raise G4S4Error("endpoint type outside alphabet")
    firsts=state.reserve_external_relation(a); fh=Counter(); fp={}; total2=total3=0
    for x in firsts:
        seconds=x.reserve_external_relation(b); total2+=len(seconds); sh=Counter(); sp={}
        for y in seconds:
            thirds=y.reserve_external_relation(c); total3+=len(thirds)
            p={"second_caps7":_caps7(y),"third_relation_cardinality":len(thirds),"third_successor_caps7_set":[list(z) for z in sorted({tuple(_caps7(q)) for q in thirds})]}
            h=canonical_sha256(p); sp.setdefault(h,p); sh[h]+=1
        p={"first_caps7":_caps7(x),"second_relation_cardinality":len(seconds),"second_successor_caps7_set":[list(z) for z in sorted({tuple(_caps7(q)) for q in seconds})],"second_branch_profile_histogram":[{"branch_profile_sha256":h,"multiplicity":int(sh[h]),"profile":sp[h]} for h in sorted(sh)]}
        h=canonical_sha256(p); fp.setdefault(h,p); fh[h]+=1
    out={"schema_id":"IG_G4_S4_ORDERED_THREE_RESERVATION_SIGNATURE_V1","first_reserved_type":a,"second_reserved_type":b,"third_reserved_type":c,"first_relation_cardinality":len(firsts),"second_relation_output_cardinality":total2,"final_relation_output_cardinality":total3,"joint_legal":bool(total3),"first_branch_profile_histogram":[{"branch_profile_sha256":h,"multiplicity":int(fh[h]),"profile":fp[h]} for h in sorted(fh)],"direct_topology_read":False,"direct_shell_profile_read":False,"construction_identity_recorded":False}
    out["science_sha256"]=canonical_sha256(out); return out


def certify_three_reservation(*, s3_result:Mapping[str,Any], task_rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    cand=_tier1_from_s3(s3_result); expected=2*343
    if len(task_rows)!=expected: raise G4S4Error(f"expected {expected} local-three rows")
    grouped=defaultdict(list); seen=set()
    for r in task_rows:
        ref=str(r["term_ref"]); key=(ref,int(r["a"]),int(r["b"]),int(r["c"]));
        if ref not in cand or key in seen: raise G4S4Error("bad/duplicate local-three row")
        seen.add(key); sig=r["signature"]; sh=str(sig["science_sha256"])
        if canonical_sha256({k:v for k,v in sig.items() if k!="science_sha256"})!=sh: raise G4S4Error("local signature hash mismatch")
        grouped[(cand[ref]["candidate_key_sha256"],key[1],key[2],key[3])].append(sh)
    conflicts=[{"candidate_key_sha256":k[0],"reservation_types":list(k[1:]),"signature_hashes":sorted(set(v))} for k,v in sorted(grouped.items(),key=str) if len(set(v))>1]
    out={"schema_id":"IG_G4_S4_TIER1_THREE_RESERVATION_SUFFICIENCY_V1","status":"PASS" if not conflicts else "FAIL","candidate":{"tier":1,"name":"SCALAR_TREE_SUMMARIES"},"task_row_count":len(task_rows),"ordered_endpoint_triple_count":343,"candidate_class_count":2,"conflict_count":len(conflicts),"conflict_examples":conflicts[:16],"bounded_singleton_class_note":"Each frozen Tier-1 class contains one exact S0 term; this is a bounded closure premise, not global Tier-1 sufficiency.","public_descriptor_promoted":False,"topology_promoted":False,"shell_profile_promoted":False}
    out["science_sha256"]=canonical_sha256(out); return out


def s1_pair_signature_index(rows:Mapping[str,Any])->dict[tuple[str,str,int,int],str]:
    out={}
    for r in rows.get("rows",[]):
        if str(r["orientation"])!="TARGET_LEFT_CONTEXT_RIGHT": continue
        k=(str(r["target_ref"]),str(r["context_ref"]),int(r["operator"][0]),int(r["operator"][1]))
        out[k]=str(r["operational_signature"]["science_sha256"])
    if len(out)!=124: raise G4S4Error("S1 pair signature direct-orientation index must contain 124 rows")
    return out


def build_pair_basis(*, states:Mapping[str,G3TermState], bridge_pairs:Sequence[Sequence[int]], s1_index:Mapping[str,Any])->tuple[dict[str,Any],dict[int,tuple[G4PairTermState,...]],list[dict[str,Any]]]:
    refs=sorted(states); expected=s1_pair_signature_index(s1_index); rels={}; rows=[]; failures=[]; idx=0
    for lr in refs:
        for rr in refs:
            for a,b in sorted(tuple(map(int,x)) for x in bridge_pairs):
                rel=compose_g4_pair_relation(states[lr],states[rr],a,b)
                km=pair_context_kernel_measurement(left=states[lr],right=states[rr],operator=[a,b])
                sig=expand_pair_context_kernel_signature(km,orientation="TARGET_LEFT_CONTEXT_RIGHT")
                exp=expected[(lr,rr,a,b)]
                ok=(sig["science_sha256"]==exp and bool(rel))
                if not ok: failures.append({"index":idx,"left_ref":lr,"right_ref":rr,"operator":[a,b],"expected":exp,"observed":sig["science_sha256"],"relation_cardinality":len(rel)})
                rels[idx]=rel; rows.append({"pair_key_index":idx,"left_ref":lr,"right_ref":rr,"operator":[a,b],"s1_signature_sha256":exp,"observed_signature_sha256":sig["science_sha256"],"pair_relation_cardinality":len(rel),"public_output_caps7_set":[list(x) for x in sorted({tuple(_caps7(st)) for st in rel})]}); idx+=1
    basis={"schema_id":"IG_G4_S4_COMPLETE_PAIR_CANDIDATE_BASIS_V1","status":"PASS" if not failures else "FAIL","pair_candidate_count":len(rows),"ordered_base_pair_assignment_count":4,"directed_operator_count":len(bridge_pairs),"complete_relation_used":True,"exact_branch_selected":False,"construction_identity_public_selector":False,"failure_count":len(failures),"failure_examples":failures[:16],"rows":rows}
    basis["science_sha256"]=canonical_sha256(basis); return basis,rels,rows


def recursive_signature(*, pair_relation:Sequence[G4PairTermState], third:G3TermState, operator:Sequence[int])->tuple[dict[str,Any],tuple[G4TripleTermState,...]]:
    a,b=map(int,operator); outs=compose_g4_recursive_triple_relation(pair_relation,third,a,b)
    pair_caps=_caps7(pair_relation[0]) if pair_relation else [0]*7; expected=[pair_caps[t]+third.total_caps[t]-(1 if t==a else 0)-(1 if t==b else 0) for t in range(7)]
    mismatch=sum(1 for st in outs if _caps7(st)!=expected)
    out={"schema_id":"IG_G4_S4_RECURSIVE_P3_SIGNATURE_V1","operator":[a,b],"pair_relation_cardinality":len(pair_relation),"recursive_relation_cardinality":len(outs),"relation_nonempty":bool(outs),"expected_output_caps7":expected,"expected_caps7_nonnegative":all(x>=0 for x in expected),"public_output_caps7_set":[list(x) for x in sorted({tuple(_caps7(st)) for st in outs})],"caps7_write_mismatch_count":mismatch,"complete_pair_relation_used":True,"exact_branch_selected":False,"direct_topology_read":False,"construction_identity_recorded":False}
    out["science_sha256"]=canonical_sha256(out); return out,outs


def reserve_capability(outputs:Sequence[G4TripleTermState])->dict[str,Any]:
    failures=[]; hist={t:Counter() for t in range(7)}; checks=0
    for st in outputs:
        caps=_caps7(st)
        for t in range(7):
            if caps[t]<=0: continue
            rel=st.reserve_external_relation(t); checks+=1; hist[t][len(rel)]+=1; exp=list(caps); exp[t]-=1
            if not rel: failures.append({"endpoint_type":t,"failure":"EMPTY"})
            elif any(_caps7(x)!=exp for x in rel): failures.append({"endpoint_type":t,"failure":"CAPS7_MISMATCH"})
    out={"schema_id":"IG_G4_S4_RECURSIVE_RESERVE_CAPABILITY_V1","status":"PASS" if not failures else "FAIL","recursive_output_count":len(outputs),"reserve_checks":checks,"failure_count":len(failures),"failure_examples":failures[:16],"relation_cardinality_histograms":[{"endpoint_type":t,"histogram":[{"relation_cardinality":k,"output_count":v} for k,v in sorted(hist[t].items())]} for t in range(7)],"direct_topology_read":False,"construction_identity_recorded":False}
    out["science_sha256"]=canonical_sha256(out); return out


def finalize_s4_result(*, s0_result:Mapping[str,Any],s1_result:Mapping[str,Any],s2_result:Mapping[str,Any],s3_result:Mapping[str,Any],s3_replay:Mapping[str,Any],s3_closeout:Mapping[str,Any],reproduction:Mapping[str,Any],local_rows:Sequence[Mapping[str,Any]],pair_basis:Mapping[str,Any],recursive_rows:Sequence[Mapping[str,Any]],reserve_rows:Sequence[Mapping[str,Any]])->dict[str,Any]:
    auth=verify_s4_authority(s0_result=s0_result,s1_result=s1_result,s2_result=s2_result,s3_result=s3_result,s3_replay=s3_replay,s3_closeout=s3_closeout)
    local=certify_three_reservation(s3_result=s3_result,task_rows=local_rows)
    failures=[]; cards=Counter(); seen=set()
    for r in recursive_rows:
        k=(int(r["pair_key_index"]),str(r["third_ref"]),int(r["operator"][0]),int(r["operator"][1]))
        if k in seen: raise G4S4Error("duplicate recursive P3 row")
        seen.add(k); sig=r["signature"]
        if canonical_sha256({x:y for x,y in sig.items() if x!="science_sha256"})!=sig["science_sha256"]: raise G4S4Error("recursive signature hash mismatch")
        cards[int(sig["recursive_relation_cardinality"])]+=1
        if not sig["relation_nonempty"] or not sig["expected_caps7_nonnegative"] or int(sig["caps7_write_mismatch_count"])!=0: failures.append({"key":list(k),"signature":sig})
    reserve_fail=[r for r in reserve_rows if r["reserve_capability"]["status"]!="PASS"]
    passed=auth["status"]=="PASS" and local["status"]=="PASS" and pair_basis.get("status")=="PASS" and len(recursive_rows)==7688 and not failures and len(reserve_rows)==31 and not reserve_fail
    rec={"schema_id":"IG_G4_S4_RECURSIVE_P3_MATERIALISATION_AUDIT_V1","status":"PASS" if not failures and len(recursive_rows)==7688 else "FAIL","public_context_count":len(recursive_rows),"expected_public_context_count":7688,"failure_count":len(failures),"failure_examples":failures[:16],"relation_cardinality_histogram":[{"relation_cardinality":k,"public_context_count":v} for k,v in sorted(cards.items())]}; rec["science_sha256"]=canonical_sha256(rec)
    rb={"schema_id":"IG_G4_S4_RECURSIVE_RESERVE_BASIS_AUDIT_V1","status":"PASS" if len(reserve_rows)==31 and not reserve_fail else "FAIL","operator_rows":len(reserve_rows),"failure_count":len(reserve_fail),"failure_examples":reserve_fail[:8]}; rb["science_sha256"]=canonical_sha256(rb)
    result={"schema_id":"IG_G4_S4_COMPOSITION_CLOSURE_RESULT_V1","status":"PASS" if passed else "REVIEW_REQUIRED","stage_ref":"G4:S4","classification":"G4_ONE_STEP_RELATION_VALUED_COMPOSITION_CLOSURE_ON_FROZEN_TIER1_P3_SCOPE_S5_UNLOCKED" if passed else "G4_COMPOSITION_CLOSURE_PREMISE_OR_MATERIALISATION_FAILURE_S5_LOCKED","authority":auth,"challenge_reproduction":dict(reproduction),"pair_candidate_basis":dict(pair_basis),"tier1_three_reservation_sufficiency":local,"recursive_materialisation_audit":rec,"recursive_reserve_basis_audit":rb,"minimal_added_read_at_tested_composition_observer":"NONE_BEYOND_S2_TIER1_CANDIDATE" if passed else None,"candidate_read_status":"BOUNDED_TIER1_CANDIDATE_SURVIVES_S4_NOT_PUBLICLY_PROMOTED" if passed else "REVIEW_REQUIRED","public_descriptor_promoted":False,"topology_promoted":False,"shell_profile_promoted":False,"g4_s5_unlocked":passed,"g4_graduated":False,"next_authorized_stage":"G4:S5" if passed else None,"scope":"EXACT_FROZEN_G4_S0_TWO_TERM_DOMAIN__124_COMPLETE_PAIR_CANDIDATES__7688_EXACT_ORDERED_RECURSIVE_P3_CONTEXTS__31_OPERATOR_POST_RECURSIVE_RESERVE_BASIS","nonclaims":list(s4_spec()["nonclaims"])}
    result["science_sha256"]=canonical_sha256(result); return result


def compare_cold_replay(primary:Mapping[str,Any], cold:Mapping[str,Any])->dict[str,Any]:
    checks={"science_sha_equal":primary.get("science_sha256")==cold.get("science_sha256"),"source_sha_equal":primary.get("source_sha256")==cold.get("source_sha256"),"registry_sha_equal":primary.get("registry_sha256")==cold.get("registry_sha256"),"classification_equal":primary.get("classification")==cold.get("classification"),"pair_basis_equal":primary.get("pair_candidate_basis")==cold.get("pair_candidate_basis"),"local_sufficiency_equal":primary.get("tier1_three_reservation_sufficiency")==cold.get("tier1_three_reservation_sufficiency"),"recursive_audit_equal":primary.get("recursive_materialisation_audit")==cold.get("recursive_materialisation_audit"),"reserve_basis_equal":primary.get("recursive_reserve_basis_audit")==cold.get("recursive_reserve_basis_audit")}
    failures=[k.upper() for k,v in checks.items() if not v]
    out={"schema_id":"IG_G4_S4_COLD_REPLAY_COMPARISON_V1","status":"PASS" if not failures else "FAIL","certification":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED","failures":failures,"experiment_id":"G4:S4","same_registered_decoder_native_experiment":True,**checks,"primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"primary_source_sha256":primary.get("source_sha256"),"cold_source_sha256":cold.get("source_sha256"),"primary_registry_sha256":primary.get("registry_sha256"),"cold_registry_sha256":cold.get("registry_sha256"),"lower_layer_rematerialization":False}
    out["comparison_sha256"]=canonical_sha256(out); return out


def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if replay.get("certification")!="CERTIFIED_PASS": failures.append("REPLAY")
    if primary.get("status")!="PASS" or cold.get("status")!="PASS": failures.append("SCIENCE_STATUS")
    out={"schema_id":"IG_G4_S4_CERTIFIED_CLOSEOUT_V1","status":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED","failures":failures,"experiment_id":"G4:S4","classification":primary.get("classification") if not failures else None,"science_sha256":primary.get("science_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),"comparison_sha256":replay.get("comparison_sha256"),"public_descriptor_promoted":False,"topology_promoted":False,"shell_profile_promoted":False,"g4_s5_unlocked":bool(not failures and primary.get("g4_s5_unlocked")),"g4_graduated":False,"next_authorized_stage":"G4:S5" if not failures and primary.get("g4_s5_unlocked") else None}
    out["closeout_sha256"]=canonical_sha256(out); return out

# Portable worker contexts.
_WORKER_STATES:dict[str,G3TermState]|None=None
_WORKER_PAIRS:dict[int,tuple[G4PairTermState,...]]|None=None

def init_g4_s4_local_worker(payload:Mapping[str,Any])->None:
    global _WORKER_STATES
    _WORKER_STATES={str(k):G3TermState.from_wire(v) for k,v in payload["states"].items()}

def g4_s4_local_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_STATES is None: raise G4S4Error("local worker context absent")
    ref=str(payload["term_ref"]); a,b,c=map(int,(payload["a"],payload["b"],payload["c"]))
    return {"term_ref":ref,"a":a,"b":b,"c":c,"signature":ordered_three_reservation_signature(_WORKER_STATES[ref],a,b,c)}

def init_g4_s4_recursive_worker(payload:Mapping[str,Any])->None:
    global _WORKER_STATES,_WORKER_PAIRS
    _WORKER_STATES={str(k):G3TermState.from_wire(v) for k,v in payload["states"].items()}
    # Pair relations are cheap and deterministically rebuilt from frozen exact refs/operators.
    _WORKER_PAIRS={}
    for row in payload["pair_rows"]:
        i=int(row["pair_key_index"]); lr=str(row["left_ref"]); rr=str(row["right_ref"]); a,b=map(int,row["operator"])
        _WORKER_PAIRS[i]=compose_g4_pair_relation(_WORKER_STATES[lr],_WORKER_STATES[rr],a,b)

def g4_s4_recursive_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_STATES is None or _WORKER_PAIRS is None: raise G4S4Error("recursive worker context absent")
    i=int(payload["pair_key_index"]); ref=str(payload["third_ref"]); op=[int(x) for x in payload["operator"]]
    sig,_=recursive_signature(pair_relation=_WORKER_PAIRS[i],third=_WORKER_STATES[ref],operator=op)
    return {"pair_key_index":i,"third_ref":ref,"operator":op,"signature":sig}

def g4_s4_reserve_worker(payload:Mapping[str,Any])->dict[str,Any]:
    if _WORKER_STATES is None or _WORKER_PAIRS is None: raise G4S4Error("reserve worker context absent")
    i=int(payload["pair_key_index"]); ref=str(payload["third_ref"]); op=[int(x) for x in payload["operator"]]
    sig,outs=recursive_signature(pair_relation=_WORKER_PAIRS[i],third=_WORKER_STATES[ref],operator=op)
    return {"pair_key_index":i,"third_ref":ref,"operator":op,"recursive_signature":sig,"reserve_capability":reserve_capability(outs)}
