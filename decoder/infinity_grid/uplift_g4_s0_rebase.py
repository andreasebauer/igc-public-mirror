from __future__ import annotations

"""G4:S0.REBASE — restart the G4 leaf boundary from certified G3:R3 authority.

This stage is deliberately non-promoting.  G3 public state remains CAPS7.  The
repaired hidden tree algebra H is frozen only as challenge evidence for later G4
context-read tests.  S0.REBASE materializes a reusable four-term corpus:
(1) the certified n=4 path/star split and (2) a deterministic n=9 pair with the
same old Tier-1 scalar summaries but different repaired hidden H.
"""
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g4_term_state import G3TermState, derive_homogeneous_seed_caps
from .uplift_g3_r0 import _tree_canon, _graph_metrics
from .uplift_g3_r3 import hidden_state

class G4S0RebaseError(RuntimeError): pass

def _resource(name:str)->Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))

def _checked(name:str,schema:str)->dict[str,Any]:
    obj=json.loads(_resource(name).read_text(encoding="utf-8"))
    if obj.get("schema_id")!=schema: raise G4S0RebaseError(f"bad {name} schema")
    field="registry_sha256" if name=="G_UPLIFT_EXPERIMENT_REGISTRY_V1.json" else "science_sha256"
    payload={k:v for k,v in obj.items() if k!=field}
    if canonical_sha256(payload)!=obj.get(field): raise G4S0RebaseError(f"{name} hash mismatch")
    return obj

def s0_rebase_spec()->dict[str,Any]:
    return _checked("G4_S0_REBASE_INTERFACE_FREEZE_SPEC_V1.json","IG_G4_S0_REBASE_INTERFACE_FREEZE_SPEC_V1")

def rebase_state()->dict[str,Any]:
    return _checked("G4_REBASE_AUTHORITY_STATE_V2.json","IG_G4_REBASE_AUTHORITY_STATE_V2")

def _tier1(n:int,edges:Sequence[Sequence[int]])->dict[str,int]:
    m=_graph_metrics(n,[(int(e[0]),int(e[1])) for e in edges]); deg=[int(x) for x in m["degree_sequence"]]
    return {"MAX_DEGREE":max(deg),"LEAF_COUNT":sum(1 for x in deg if x==1),"diameter":int(m["diameter"]),"radius":int(m["radius"]),"articulation_count":int(m["articulations"]),"wiener_index":int(m["distance_sum"])}

def verify_authority(*,r1_result:Mapping[str,Any],r3_result:Mapping[str,Any],r3_replay:Mapping[str,Any],r3_closeout:Mapping[str,Any])->dict[str,Any]:
    fail=[]; state=rebase_state(); spec=s0_rebase_spec()
    # R1 remains authority only for the exact material n=4 G3 term witness.
    if r1_result.get("schema_id")!="IG_G3_R1_FIBER_MODULI_AUDIT_RESULT_V1" or r1_result.get("status")!="PASS": fail.append("R1_SCHEMA_OR_STATUS")
    if r1_result.get("classification")!="G3_HIDDEN_TREE_MODULI_PRESENT_RELATION_VALUED_FIBER_GRAFTING_SHELL_PROFILE_READ_CANDIDATE_EARNED_G4_NOT_STARTED" or r1_result.get("science_sha256")!="c9df07857a83ab21dbfaf9cb1755e459557ba400d34536900c3d3a5e6631a5ed": fail.append("R1_IDENTITY")
    if r1_result.get("g3_graduation_preserved") is not True or r1_result.get("promotion") is not False or r1_result.get("topology_promoted") is not False: fail.append("R1_FIREWALL")
    # R3 is now the current hidden structural authority.
    if r3_result.get("schema_id")!="IG_G3_R3_STRUCTURAL_ALGEBRA_REBASE_RESULT_V1" or r3_result.get("status")!="PASS": fail.append("R3_SCHEMA_OR_STATUS")
    if r3_result.get("science_sha256")!=state["g3_r3_primary_science_sha256"]: fail.append("R3_SCIENCE_IDENTITY")
    if r3_result.get("classification")!="G3_R3_REPAIRED_HIDDEN_TREE_BINARY_GRAFT_ALGEBRA_EARNED_G3_CAPS7_GRADUATION_PRESERVED_G4_REBASE_REQUIRED": fail.append("R3_CLASSIFICATION")
    if r3_result.get("g3_public_caps7_graduation_preserved") is not True or r3_result.get("hidden_state_promoted_to_public_g3") is not False: fail.append("R3_FIREWALL")
    if r3_result.get("repaired_hidden_state")!=spec["corrected_hidden_authority"]["state"]: fail.append("R3_HIDDEN_STATE")
    if r3_replay.get("schema_id")!="IG_G3_R3_REPLAY_COMPARISON_V1" or r3_replay.get("status")!="PASS" or r3_replay.get("stable_scientific_payload_exact_equal") is not True: fail.append("R3_REPLAY")
    if r3_replay.get("primary_science_sha256")!=r3_result.get("science_sha256") or r3_replay.get("cold_science_sha256")!=r3_result.get("science_sha256"): fail.append("R3_REPLAY_BINDING")
    if r3_closeout.get("schema_id")!="IG_G3_R3_CERTIFIED_CLOSEOUT_V1" or r3_closeout.get("status")!="CERTIFIED_PASS": fail.append("R3_CLOSEOUT")
    if r3_closeout.get("science_sha256")!=state["g3_r3_certified_closeout_science_sha256"] or r3_closeout.get("primary_science_sha256")!=r3_result.get("science_sha256") or r3_closeout.get("cold_science_sha256")!=r3_result.get("science_sha256"): fail.append("R3_CLOSEOUT_BINDING")
    if int(r3_closeout.get("mismatch_count",-1))!=0 or int(r3_closeout.get("ordered_rooted_binary_graft_action_count",-1))!=202816: fail.append("R3_EXHAUSTIVE_GATE")
    out={"schema_id":"IG_G4_S0_REBASE_AUTHORITY_VERIFICATION_V1","status":"PASS" if not fail else "FAIL","failures":fail,"g3_r1_science_sha256":r1_result.get("science_sha256"),"g3_r3_science_sha256":r3_result.get("science_sha256"),"g3_r3_replay_science_sha256":r3_replay.get("science_sha256"),"g3_r3_closeout_science_sha256":r3_closeout.get("science_sha256"),"g3_public_descriptor":"CAPS7","repaired_hidden_state":spec["corrected_hidden_authority"]["state"],"hidden_state_promoted":False,"historical_g4_forward_use":"BLOCKED_PENDING_REBASE"}
    out["science_sha256"]=canonical_sha256(out)
    if fail: raise G4S0RebaseError("G4:S0.REBASE authority failed: "+",".join(fail))
    return out

def _public(caps:Sequence[int])->dict[str,Any]:
    o={"schema_id":"IG_G2_CAPS7_STATE_V1","coordinates":[int(x) for x in caps]}; o["science_sha256"]=canonical_sha256(o); return o

def _diag(n:int,edges:Sequence[Sequence[int]])->dict[str,Any]:
    pairs=[(int(e[0]),int(e[1])) for e in edges]; m=_graph_metrics(n,pairs); H=hidden_state(n,pairs)
    o={"visibility":"NONPUBLIC_G4_CHALLENGE_EVIDENCE","unit_count":n,"topology_canon":_tree_canon(n,pairs),"tier1_scalar_tree_summaries":_tier1(n,pairs),"repaired_hidden_state":H}
    o["science_sha256"]=canonical_sha256(o); return o

def _term(seed:Sequence[int], n:int, edges:Sequence[Sequence[int]])->G3TermState:
    typed=[(int(u),int(v),0,0) for u,v in edges]
    return G3TermState.from_homogeneous_seed([int(x) for x in seed],typed,unit_count=n)

def build_rebased_term_corpus(r1_result:Mapping[str,Any])->dict[str,Any]:
    fib=r1_result.get("certified_exact_n4_fiber",{}); base=fib.get("fiber_base",{}); wit=fib.get("coarse_topology_witness",{})
    if int(base.get("g2_unit_count",0))!=4: raise G4S0RebaseError("R1 exact n=4 fiber missing")
    shared=[int(x) for x in base.get("caps7",[])]; rows={"A":wit.get("topology_A",{}),"B":wit.get("topology_B",{})}
    seed=None; terms={}; diagnostics={}
    for key,row in rows.items():
        typed=[tuple(int(x) for x in e) for e in row.get("typed_g3_edges",[])]
        local=derive_homogeneous_seed_caps(final_caps=shared,unit_count=4,typed_edges=typed)
        if seed is None: seed=local
        if local!=seed: raise G4S0RebaseError("R1 n4 witness seed mismatch")
        pairs=[(e[0],e[1]) for e in typed]; st=G3TermState.from_homogeneous_seed(seed,typed,unit_count=4)
        if list(st.total_caps)!=shared: raise G4S0RebaseError("R1 n4 term CAPS7 mismatch")
        terms[key]={"label":"A_N4_PATH" if key=="A" else "B_N4_STAR","term":st.to_wire(),"source":"CERTIFIED_G3_R1_N4_FIBER"}
        diagnostics[key]=_diag(4,pairs)
    if seed is None: raise G4S0RebaseError("seed missing")
    lane=next(x for x in s0_rebase_spec()["challenge_corpus"]["lanes"] if x["lane_id"]=="N9_TIER1_COLLISION_REPAIRED_H_SPLIT")
    for key,field,label in (("C","A_edges","C_N9_TIER1_COLLISION_LEFT"),("D","B_edges","D_N9_TIER1_COLLISION_RIGHT")):
        edges=[tuple(int(y) for y in e) for e in lane[field]]; st=_term(seed,9,edges)
        terms[key]={"label":label,"term":st.to_wire(),"source":"DETERMINISTIC_G3_R3_TREE_PLUS_GRADUATED_HOMOGENEOUS_G3_GRAFT"}; diagnostics[key]=_diag(9,edges)
    # Exact rebase gates.
    A,B,C,D=[G3TermState.from_wire(terms[k]["term"]) for k in "ABCD"]
    if A.total_caps!=B.total_caps or diagnostics["A"]["topology_canon"]==diagnostics["B"]["topology_canon"]: raise G4S0RebaseError("n4 challenge collapsed")
    if C.total_caps!=D.total_caps: raise G4S0RebaseError("n9 public CAPS7 mismatch")
    if diagnostics["C"]["tier1_scalar_tree_summaries"]!=diagnostics["D"]["tier1_scalar_tree_summaries"]: raise G4S0RebaseError("n9 old Tier1 collision missing")
    if diagnostics["C"]["tier1_scalar_tree_summaries"]!={str(k):int(v) for k,v in lane["old_tier1_value"].items()}: raise G4S0RebaseError("n9 Tier1 freeze mismatch")
    if diagnostics["C"]["repaired_hidden_state"]["science_sha256"]==diagnostics["D"]["repaired_hidden_state"]["science_sha256"]: raise G4S0RebaseError("n9 repaired H failed to separate")
    out={"schema_id":"IG_G4_S0_REBASE_G3_TERM_CORPUS_V1","status":"PASS","semantics":"PREVIOUS_LAYER_G2_UNITS_ATOMIC_AT_EARNED_CAPS7_INTERFACE","lower_layer_rematerialization":False,"atomic_g2_seed_caps7":[int(x) for x in seed],"terms":terms,"hidden_diagnostics":diagnostics,"challenge_pairs":[{"pair_id":"N4_PATH_STAR","left":"A","right":"B","same_caps7":True,"same_old_tier1":diagnostics["A"]["tier1_scalar_tree_summaries"]==diagnostics["B"]["tier1_scalar_tree_summaries"],"different_repaired_H":diagnostics["A"]["repaired_hidden_state"]["science_sha256"]!=diagnostics["B"]["repaired_hidden_state"]["science_sha256"]},{"pair_id":"N9_TIER1_COLLISION_REPAIRED_H_SPLIT","left":"C","right":"D","same_caps7":True,"same_old_tier1":True,"different_repaired_H":True}],"public_interface":"CAPS7_ONLY","corrected_hidden_state":"SHELL_PROFILE_MULTISET_PLUS_PAIRED_SUPPORT_LOAD_DISTANCE_SIGNATURE_MULTISET","corrected_hidden_state_visibility":"NONPUBLIC_CHALLENGE_ONLY"}
    out["science_sha256"]=canonical_sha256(out); return out

def run_s0_rebase(*,r1_result:Mapping[str,Any],r3_result:Mapping[str,Any],r3_replay:Mapping[str,Any],r3_closeout:Mapping[str,Any])->dict[str,Any]:
    auth=verify_authority(r1_result=r1_result,r3_result=r3_result,r3_replay=r3_replay,r3_closeout=r3_closeout); corpus=build_rebased_term_corpus(r1_result)
    out={"schema_id":"IG_G4_S0_REBASE_INTERFACE_RESULT_V1","status":"PASS","stage_ref":"G4:S0.REBASE","classification":s0_rebase_spec()["outcomes"]["pass"],"authority":auth,"baseline_public_observer":"CAPS7_ONLY_INHERITED_FROM_GRADUATED_G3","corrected_hidden_authority":"G3:R3_REPAIRED_HIDDEN_TREE_BINARY_GRAFT_ALGEBRA","repaired_hidden_state":"SHELL_PROFILE_MULTISET_PLUS_PAIRED_SUPPORT_LOAD_DISTANCE_SIGNATURE_MULTISET","hidden_state_promoted":False,"certified_term_corpus":corpus,"challenge_pair_count":2,"old_tier1_sufficiency_assumed":False,"s0_materializes_reusable_base":True,"s1_lower_layer_rematerialization_forbidden":True,"g4_s1_rebase_unlocked":True,"g4_rebase_complete":False,"historical_g4_forward_use":"BLOCKED_UNTIL_REBASE_COMPLETES","g4_graduated":False,"next_authorized_stage":"G4:S1.REBASE","nonclaims":list(s0_rebase_spec()["nonclaims"])}
    out["science_sha256"]=canonical_sha256(out); return out

def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={"primary_pass":primary.get("status")=="PASS","cold_pass":cold.get("status")=="PASS","science_sha_equal":primary.get("science_sha256")==cold.get("science_sha256"),"source_sha_equal":primary.get("source_sha256")==cold.get("source_sha256"),"registry_sha_equal":primary.get("registry_sha256")==cold.get("registry_sha256"),"classification_equal":primary.get("classification")==cold.get("classification"),"term_corpus_equal":primary.get("certified_term_corpus",{}).get("science_sha256")==cold.get("certified_term_corpus",{}).get("science_sha256"),"authority_equal":primary.get("authority")==cold.get("authority")}
    ok=all(checks.values()); out={"schema_id":"IG_G4_S0_REBASE_REPLAY_COMPARISON_V1","status":"PASS" if ok else "FAIL","certification":"CERTIFIED_PASS" if ok else "NOT_CERTIFIED","checks":checks,"primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"primary_source_sha256":primary.get("source_sha256"),"cold_source_sha256":cold.get("source_sha256"),"primary_registry_sha256":primary.get("registry_sha256"),"cold_registry_sha256":cold.get("registry_sha256"),"term_corpus_sha256":primary.get("certified_term_corpus",{}).get("science_sha256")}; out["science_sha256"]=canonical_sha256(out); return out

def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    ok=replay.get("status")=="PASS" and replay.get("certification")=="CERTIFIED_PASS" and primary.get("status")=="PASS" and cold.get("status")=="PASS"
    out={"schema_id":"IG_G4_S0_REBASE_CERTIFIED_CLOSEOUT_V1","status":"CERTIFIED_PASS" if ok else "CERTIFICATION_FAIL","classification":primary.get("classification"),"primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"replay_comparison_science_sha256":replay.get("science_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),"certified_term_corpus_sha256":primary.get("certified_term_corpus",{}).get("science_sha256"),"g3_public_caps7_graduation_preserved":True,"hidden_state_promoted":False,"historical_g4_forward_use":"BLOCKED_UNTIL_REBASE_COMPLETES","g4_s1_rebase_unlocked":ok,"g4_rebase_complete":False,"next_authorized_stage":"G4:S1.REBASE" if ok else None}; out["science_sha256"]=canonical_sha256(out); return out
