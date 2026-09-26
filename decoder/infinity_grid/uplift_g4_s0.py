from __future__ import annotations

"""Registered G4:S0 whole-G3 carrier interface extraction.

Phase 0 inherits only the graduated G3 CAPS7 public interface. Certified hidden
G2-unit tree/fiber structure, including SHELL_PROFILE_MULTISET, is retained strictly
as challenge evidence. S0 performs authority/firewall validation and freezes the
challenge corpus; it does not ask whether a G4 context reads that structure.
"""
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json

from .canon import canonical_sha256
from .g4_term_state import G3TermState, derive_homogeneous_seed_caps
from .uplift_g3_r0 import _tree_canon, _graph_metrics

class G4S0Error(RuntimeError): pass


def _resource(name:str)->Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def _load(name:str)->dict[str,Any]:
    return json.loads(_resource(name).read_text(encoding="utf-8"))


def _checked(name:str, schema_id:str)->dict[str,Any]:
    obj=_load(name)
    if obj.get("schema_id")!=schema_id: raise G4S0Error(f"bad {name} schema")
    field="registry_sha256" if name=="G_UPLIFT_EXPERIMENT_REGISTRY_V1.json" else "science_sha256"
    payload={k:v for k,v in obj.items() if k!=field}
    if canonical_sha256(payload)!=obj.get(field): raise G4S0Error(f"{name} hash mismatch")
    return obj


def phase0_spec()->dict[str,Any]:
    return _checked("G4_PHASE0_ARCHITECTURE_FREEZE_V1.json","IG_G4_PHASE0_ARCHITECTURE_FREEZE_V1")


def challenge_tiers()->dict[str,Any]:
    obj=_checked("G4_CHALLENGE_READ_TIERS_V1.json","IG_G4_CHALLENGE_READ_TIERS_V1")
    if obj["science_sha256"]!=phase0_spec()["challenge_read_tier_science_sha256"]:
        raise G4S0Error("G4 challenge-tier/Phase0 binding mismatch")
    return obj


def s0_spec()->dict[str,Any]:
    obj=_checked("G4_S0_INTERFACE_EXTRACTION_SPEC_V2.json","IG_G4_S0_INTERFACE_EXTRACTION_SPEC_V2")
    if obj.get("phase0_spec_sha256")!=phase0_spec()["science_sha256"]: raise G4S0Error("G4 S0/Phase0 binding mismatch")
    if obj.get("challenge_read_tier_science_sha256")!=challenge_tiers()["science_sha256"]: raise G4S0Error("G4 S0/tier binding mismatch")
    return obj


def s1_placeholder_spec()->dict[str,Any]:
    obj=_checked("G4_S1_HIDDEN_STRUCTURE_CONTEXT_READ_SPEC_V2.json","IG_G4_S1_HIDDEN_STRUCTURE_CONTEXT_READ_SPEC_V2")
    if obj.get("phase0_spec_sha256")!=phase0_spec()["science_sha256"]: raise G4S0Error("G4 S1/Phase0 binding mismatch")
    if obj.get("challenge_read_tier_science_sha256")!=challenge_tiers()["science_sha256"]: raise G4S0Error("G4 S1/tier binding mismatch")
    return obj


def verify_authority(*, r1_result:Mapping[str,Any], r2_result:Mapping[str,Any], r2_replay:Mapping[str,Any], r2_closeout:Mapping[str,Any])->dict[str,Any]:
    p=phase0_spec()["authority"]; failures=[]
    if r1_result.get("schema_id")!="IG_G3_R1_FIBER_MODULI_AUDIT_RESULT_V1" or r1_result.get("status")!="PASS": failures.append("R1_SCHEMA_OR_STATUS")
    if r1_result.get("classification")!=p["g3_r1_classification"] or r1_result.get("science_sha256")!=p["g3_r1_science_sha256"]: failures.append("R1_IDENTITY")
    if r1_result.get("g3_graduation_preserved") is not True or r1_result.get("promotion") is not False or r1_result.get("topology_promoted") is not False: failures.append("R1_FIREWALL")
    ca=r1_result.get("candidate_tree_read_audit",{})
    if ca.get("selected_candidates")!=["SHELL_PROFILE_MULTISET"]: failures.append("R1_SELECTED_READ")

    if r2_result.get("schema_id")!="IG_G3_R2_ADAPTIVE_MATURATION_RESULT_V1" or r2_result.get("status")!="PASS": failures.append("R2_SCHEMA_OR_STATUS")
    if r2_result.get("classification")!=p["g3_r2_certified_closeout_classification"] or r2_result.get("science_sha256")!=p["g3_r2_certified_closeout_science_sha256"]: failures.append("R2_IDENTITY")
    if r2_result.get("g3_graduation_preserved") is not True or r2_result.get("promotion") is not False or r2_result.get("topology_promoted") is not False or r2_result.get("g4_started") is not False: failures.append("R2_FIREWALL")
    if r2_result.get("selected_read_under_stress")!="SHELL_PROFILE_MULTISET": failures.append("R2_SELECTED_READ")

    if r2_replay.get("schema_id")!="IG_G3_R2_RERUN_REPLAY_COMPARISON_V1" or r2_replay.get("certification_status")!="PASS": failures.append("R2_REPLAY_SCHEMA_OR_STATUS")
    if r2_replay.get("comparison_sha256")!=p["g3_r2_replay_comparison_sha256"] or r2_replay.get("science_sha256_match") is not True or r2_replay.get("all_declared_stable_fields_match") is not True: failures.append("R2_REPLAY_IDENTITY_OR_MATCH")

    if r2_closeout.get("schema_id")!="IG_G3_R2_RERUN_CERTIFIED_CLOSEOUT_V1" or r2_closeout.get("status")!="CERTIFIED": failures.append("R2_CLOSEOUT_SCHEMA_OR_STATUS")
    if r2_closeout.get("science_sha256")!=p["g3_r2_certified_closeout_science_sha256"] or r2_closeout.get("classification")!=p["g3_r2_certified_closeout_classification"] or r2_closeout.get("closeout_sha256")!=p["g3_r2_closeout_sha256"]: failures.append("R2_CLOSEOUT_IDENTITY")
    if r2_closeout.get("cold_replay_exact_stable_payload_match") is not True or r2_closeout.get("g3_graduation_preserved") is not True or r2_closeout.get("topology_promoted") is not False or r2_closeout.get("g4_started") is not False: failures.append("R2_CLOSEOUT_FIREWALL")

    out={"schema_id":"IG_G4_S0_AUTHORITY_VERIFICATION_V1","status":"PASS" if not failures else "FAIL","failures":failures,
         "g3_r1_science_sha256":r1_result.get("science_sha256"),"g3_r2_science_sha256":r2_result.get("science_sha256"),
         "g3_r2_replay_comparison_sha256":r2_replay.get("comparison_sha256"),"g3_r2_closeout_sha256":r2_closeout.get("closeout_sha256"),
         "g3_graduation_preserved":True,"topology_promoted":False,"shell_profile_promoted":False}
    out["science_sha256"]=canonical_sha256(out)
    if failures: raise G4S0Error("G4:S0 authority failed: "+",".join(failures))
    return out


def _public_iface(caps:list[int])->dict[str,Any]:
    out={"schema_id":"IG_G2_CAPS7_STATE_V1","coordinates":[int(x) for x in caps]}
    out["science_sha256"]=canonical_sha256(out); return out


def _hidden_record(label:str, row:Mapping[str,Any])->dict[str,Any]:
    m=row["metrics"]
    shell=[list(x) for x in sorted(tuple(int(y) for y in z) for z in m.get("shell_profiles",[]))]
    return {
      "label":label,"g3_carrier_ref":str(row["example_construction_digest"]),"public_interface":_public_iface(list(row["caps7"])),
      "hidden_challenge_diagnostic":{
        "visibility":"NOT_G4_PUBLIC_INTERFACE","g2_unit_count":4,"topology_canon":str(row["topology_canon"]),
        "degree_sequence":[int(x) for x in m["degree_sequence"]],"diameter":int(m["diameter"]),"radius":int(m["radius"]),
        "articulation_count":int(m["articulations"]),"wiener_index":int(m["distance_sum"]),"shell_profile_multiset":shell,
      }}


def _build_certified_term_corpus(r1_result:Mapping[str,Any], A:Mapping[str,Any], B:Mapping[str,Any])->dict[str,Any]:
    fib=r1_result.get("certified_exact_n4_fiber",{})
    base=fib.get("fiber_base",{})
    witness=fib.get("coarse_topology_witness",{})
    n=int(base.get("g2_unit_count",0))
    if n!=4: raise G4S0Error("G4:S0 V2 requires the certified n=4 G3 fiber")
    shared=[int(x) for x in base.get("caps7",[])]
    if shared!=[int(x) for x in witness.get("shared_caps7",[])]: raise G4S0Error("R1 fiber/witness shared CAPS7 mismatch")
    rows={"A":witness.get("topology_A",{}),"B":witness.get("topology_B",{})}
    terms={}
    seed_caps=None
    for key,row in rows.items():
        edges=[tuple(int(x) for x in e) for e in row.get("typed_g3_edges",[])]
        local_seed=derive_homogeneous_seed_caps(final_caps=shared,unit_count=n,typed_edges=edges)
        if seed_caps is None: seed_caps=local_seed
        if local_seed!=seed_caps: raise G4S0Error("path/star imply different atomic G2 seed CAPS7")
        term=G3TermState.from_homogeneous_seed(seed_caps,edges,unit_count=n)
        if list(term.total_caps)!=shared: raise G4S0Error(f"{key} certified term CAPS7 mismatch")
        expected_topo=str(row.get("topology_canon"))
        pairs=[(int(e[0]),int(e[1])) for e in edges]
        observed_topo=_tree_canon(n,pairs)
        if observed_topo!=expected_topo:
            raise G4S0Error(f"{key} G3 term topology does not reproduce certified R1 topology")
        observed_metrics=_graph_metrics(n,pairs)
        certified_metrics=row.get("metrics",{})
        for metric_key, observed_key in (("degree_sequence","degree_sequence"),("diameter","diameter"),("radius","radius"),("articulations","articulations"),("distance_sum","distance_sum")):
            if observed_metrics.get(observed_key)!=certified_metrics.get(metric_key):
                raise G4S0Error(f"{key} G3 term metric mismatch: {metric_key}")
        observed_shell=[list(x) for x in sorted(tuple(int(y) for y in z) for z in observed_metrics.get("shell_profiles",[]))]
        certified_shell=[list(x) for x in sorted(tuple(int(y) for y in z) for z in certified_metrics.get("shell_profiles",[]))]
        if observed_shell!=certified_shell:
            raise G4S0Error(f"{key} G3 term shell-profile mismatch")
        terms[key]={
          "label":"A_PATH_LIKE" if key=="A" else "B_STAR_LIKE",
          "historical_g3_carrier_ref":str(row.get("example_construction_digest")),
          "historical_topology_canon":expected_topo,
          "term":term.to_wire(),
        }
    assert seed_caps is not None
    if terms["A"]["term"]["term_ref"]==terms["B"]["term"]["term_ref"]: raise G4S0Error("G4:S0 V2 term corpus collapsed path/star")
    out={
      "schema_id":"IG_G4_S0_CERTIFIED_G3_TERM_CORPUS_V1",
      "status":"PASS",
      "semantics":"PREVIOUS_LAYER_G2_UNITS_ATOMIC_AT_EARNED_CAPS7_INTERFACE",
      "lower_layer_rematerialization":False,
      "unit_count":n,
      "atomic_g2_seed_caps7":[int(x) for x in seed_caps],
      "shared_public_caps7":shared,
      "g3_bridge_operator":[int(x) for x in base.get("g3_bridge_operator",[])],
      "g3_bridge_count":int(base.get("g3_bridge_count",0)),
      "terms":terms,
    }
    out["science_sha256"]=canonical_sha256(out)
    return out

def run_g4_s0(*, engine:Any, r1_result:Mapping[str,Any], r2_result:Mapping[str,Any], r2_replay:Mapping[str,Any], r2_closeout:Mapping[str,Any])->dict[str,Any]:
    spec=s0_spec(); auth=verify_authority(r1_result=r1_result,r2_result=r2_result,r2_replay=r2_replay,r2_closeout=r2_closeout)
    tiers=challenge_tiers(); witness=r1_result["certified_exact_n4_fiber"]["coarse_topology_witness"]
    A=_hidden_record("A_PATH_LIKE",witness["topology_A"]); B=_hidden_record("B_STAR_LIKE",witness["topology_B"])
    if A["public_interface"]["coordinates"]!=B["public_interface"]["coordinates"]: raise G4S0Error("G4:S0 challenge pair CAPS7 mismatch")
    if A["hidden_challenge_diagnostic"]["topology_canon"]==B["hidden_challenge_diagnostic"]["topology_canon"]: raise G4S0Error("G4:S0 challenge topology collapsed")
    public_keys=set(A["public_interface"])|set(B["public_interface"])
    forbidden={"topology_canon","shell_profile_multiset","degree_sequence","diameter","radius","articulation_count","wiener_index","g3_carrier_ref"}
    if public_keys & forbidden: raise G4S0Error("hidden field leaked into public interface")
    if tiers["ordered_tiers"][4]["name"]!="SHELL_PROFILE_MULTISET": raise G4S0Error("G4 tier order changed")

    term_corpus=_build_certified_term_corpus(r1_result,A,B)
    # S0 now actually freezes the reusable base that S1 consumes.
    A=dict(A); B=dict(B)
    A["g4_term_ref"]=term_corpus["terms"]["A"]["term"]["term_ref"]
    B["g4_term_ref"]=term_corpus["terms"]["B"]["term"]["term_ref"]
    A["historical_g3_carrier_ref"]=A.pop("g3_carrier_ref")
    B["historical_g3_carrier_ref"]=B.pop("g3_carrier_ref")
    result={
      "schema_id":"IG_G4_S0_INTERFACE_EXTRACTION_RESULT_V2","status":"PASS",
      "classification":"G4_BASELINE_G3_TERM_CORPUS_FROZEN_CAPS7_ONLY_HIDDEN_TREE_CHALLENGE_HIERARCHY_EARNED_S1_UNLOCKED",
      "g4_started":True,"g4_graduated":False,"g4_s1_unlocked":True,"promotion":False,
      "authority":auth,"phase0_spec_sha256":phase0_spec()["science_sha256"],"s0_spec_sha256":spec["science_sha256"],
      "challenge_read_tier_science_sha256":tiers["science_sha256"],"g4_unit_definition":phase0_spec()["g4_unit_definition"],
      "baseline_public_observer":"CAPS7_ONLY_INHERITED_FROM_GRADUATED_G3",
      "certified_term_corpus":term_corpus,
      "certified_collision_pair":{"A":A,"B":B,"same_public_interface":True,"different_hidden_tree_topology":True,
        "meaning":"S0 confirms a valid equal-CAPS7 hidden-structure challenge pair. It does not claim that any G4 context distinguishes the pair."},
      "challenge_read_order":[{"tier":x["tier"],"name":x["name"],"public":x["public"]} for x in tiers["ordered_tiers"]],
      "s0_materializes_reusable_base":True,"s1_lower_layer_rematerialization_forbidden":True,
      "selected_preregistered_challenge_read":"SHELL_PROFILE_MULTISET","topology_visibility":"HIDDEN_CHALLENGE_EVIDENCE_ONLY_NOT_PROMOTED",
      "shell_profile_visibility":"HIDDEN_CHALLENGE_EVIDENCE_ONLY_NOT_PROMOTED","next_authorized_stage":"G4:S1",
      "nonclaims":spec["nonclaims"]}
    result["science_sha256"]=canonical_sha256(result); return result


def compare_cold_replay(primary:Mapping[str,Any], cold:Mapping[str,Any])->dict[str,Any]:
    """Certify an exact cold replay of the same registered Decoder-native G4:S0 V2 experiment."""
    failures=[]
    for label,obj in (("PRIMARY",primary),("COLD",cold)):
        if obj.get("schema_id")!="IG_G4_S0_INTERFACE_EXTRACTION_RESULT_V2" or obj.get("status")!="PASS":
            failures.append(f"{label}_SCHEMA_OR_STATUS")
    science_equal=primary.get("science_sha256")==cold.get("science_sha256")
    source_equal=primary.get("source_sha256")==cold.get("source_sha256")
    registry_equal=primary.get("registry_sha256")==cold.get("registry_sha256")
    class_equal=primary.get("classification")==cold.get("classification")
    corpus_equal=(primary.get("certified_term_corpus",{}).get("science_sha256")==cold.get("certified_term_corpus",{}).get("science_sha256"))
    if not science_equal: failures.append("SCIENCE_SHA_MISMATCH")
    if not source_equal: failures.append("SOURCE_SHA_MISMATCH")
    if not registry_equal: failures.append("REGISTRY_SHA_MISMATCH")
    if not class_equal: failures.append("CLASSIFICATION_MISMATCH")
    if not corpus_equal: failures.append("TERM_CORPUS_MISMATCH")
    out={
      "schema_id":"IG_G4_S0_COLD_REPLAY_COMPARISON_V2",
      "status":"PASS" if not failures else "FAIL",
      "certification":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
      "failures":failures,
      "experiment_id":"G4:S0",
      "same_registered_decoder_native_experiment":True,
      "science_sha_equal":science_equal,"source_sha_equal":source_equal,"registry_sha_equal":registry_equal,
      "classification_equal":class_equal,"certified_term_corpus_sha_equal":corpus_equal,
      "primary_status":primary.get("status"),"cold_status":cold.get("status"),
      "primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),
      "primary_source_sha256":primary.get("source_sha256"),"cold_source_sha256":cold.get("source_sha256"),
      "primary_registry_sha256":primary.get("registry_sha256"),"cold_registry_sha256":cold.get("registry_sha256"),
      "certified_term_corpus_sha256":primary.get("certified_term_corpus",{}).get("science_sha256"),
      "stage_specific_external_science_runner":False,
      "lower_layer_rematerialization":False,
    }
    out["comparison_sha256"]=canonical_sha256(out)
    return out


def certified_closeout(primary:Mapping[str,Any], cold:Mapping[str,Any], replay:Mapping[str,Any])->dict[str,Any]:
    """Close G4:S0 V2 only after exact primary/cold identity certification."""
    failures=[]
    if replay.get("schema_id")!="IG_G4_S0_COLD_REPLAY_COMPARISON_V2" or replay.get("status")!="PASS" or replay.get("certification")!="CERTIFIED_PASS": failures.append("REPLAY")
    if primary.get("science_sha256")!=cold.get("science_sha256"): failures.append("SCIENCE")
    if primary.get("source_sha256")!=cold.get("source_sha256"): failures.append("SOURCE")
    if primary.get("registry_sha256")!=cold.get("registry_sha256"): failures.append("REGISTRY")
    corpus_sha=primary.get("certified_term_corpus",{}).get("science_sha256")
    if not corpus_sha or corpus_sha!=cold.get("certified_term_corpus",{}).get("science_sha256"): failures.append("TERM_CORPUS")
    out={
      "schema_id":"IG_G4_S0_CERTIFIED_CLOSEOUT_V2",
      "status":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED",
      "failures":failures,"experiment_id":"G4:S0","decoder_version":"0.30.35",
      "science_sha256":primary.get("science_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),
      "classification":primary.get("classification"),"certified_term_corpus_sha256":corpus_sha,
      "same_registered_decoder_native_experiment":replay.get("same_registered_decoder_native_experiment") is True,
      "science_sha_equal":replay.get("science_sha_equal") is True,"source_sha_equal":replay.get("source_sha_equal") is True,"registry_sha_equal":replay.get("registry_sha_equal") is True,
      "g4_s1_unlocked":primary.get("g4_s1_unlocked") is True,"g4_graduated":False,
      "baseline_public_observer":primary.get("baseline_public_observer"),
      "s0_materializes_reusable_base":primary.get("s0_materializes_reusable_base") is True,
      "s1_lower_layer_rematerialization_forbidden":primary.get("s1_lower_layer_rematerialization_forbidden") is True,
      "lower_layer_rematerialization":False,"next_authorized_stage":"G4:S1" if not failures else None,
      "stage_specific_external_science_runner":False,
    }
    out["closeout_sha256"]=canonical_sha256(out)
    if failures: raise G4S0Error("G4:S0 V2 closeout failed: "+",".join(failures))
    return out
