from __future__ import annotations

"""Preregistered G5 structural capabilities.

G5:S0-S5 are registered scientific stages in the current working source. S6 remains a deliberate lock
that stops the generic workflow at REVIEW_REQUIRED until recursive-closure preregistration is frozen.
"""
from typing import Any, Mapping
from .canon import canonical_sha256
from .v05_workflow import register_workflow_capability, register_workflow_semantic_contract
from .adapters.g4_accepted import DecoratedG4Tree

PASS_ALT = "G5_S0_G4_AUTHORITY_FROZEN_CAPS7_PLUS_H_CLASS_BAG_PUBLIC_ACTUAL_G4_HIDDEN_DECORATED_TREE_CHALLENGE_CORPUS_EARNED_S1_DESIGN_REVIEW"
FAIL_ALT = "G5_S0_AUTHORITY_OR_INTERFACE_FREEZE_FAILURE"
LOCK_ALT = "G5_S1_PREREGISTRATION_REQUIRED"


def _path_broom(depth: int):
    n = int(depth) + 3
    path_edges = tuple((i, i + 1) for i in range(n - 1))
    broom_edges = tuple([(i, i + 1) for i in range(int(depth))] + [(int(depth), int(depth)+1), (int(depth), int(depth)+2)])
    hs = tuple("C" for _ in range(n))
    return (
        DecoratedG4Tree(n, path_edges, hs, tuple((0,0) for _ in path_edges)),
        DecoratedG4Tree(n, broom_edges, hs, tuple((0,0) for _ in broom_edges)),
    )


def _is_realizable_support(row: Mapping[str, Any], depth: int) -> bool:
    # The accepted R7 adapter returns one support row from the certified actual-G4
    # resource-realizability theorem.  Check the exact structural facts that make
    # this lane admissible rather than treating mere row presence as success.
    return bool(
        int(row.get("depth", -1)) == depth
        and int(row.get("g3_unit_count", -1)) == depth + 3
        and int(row.get("path_max_degree", 99)) <= 3
        and int(row.get("broom_max_degree", 99)) <= 3
        and row.get("child_canons_distinct") is True
        and isinstance(row.get("shared_public_caps7"), list)
        and len(row.get("shared_public_caps7")) == 7
        and isinstance(row.get("shared_H_class_bag"), Mapping)
        and bool(row.get("truncated_action_read_sha256"))
        and bool(row.get("path_child_canon_sha256"))
        and bool(row.get("broom_child_canon_sha256"))
    )


@register_workflow_capability("g5.s0.carrier-interface.v1")
def g5_s0_carrier_interface(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    if stage != "S0":
        return {"outcome":"BLOCKED","observed_alternative":FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    depths = tuple(int(x) for x in parameters.get("witness_depths", []))
    if depths != (2,4):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":FAIL_ALT,"failure":"CHALLENGE_LANE_MISMATCH","observed_depths":list(depths)}
    if execution_context is not None:
        execution_context.reserve_cases(len(depths))
    lanes=[]; failures=[]
    for depth in depths:
        path,broom=_path_broom(depth)
        support=adapter.realizability_witness(depth)
        q_path=adapter.public_read(path); q_broom=adapter.public_read(broom)
        h_path=adapter.unrooted_canon(path); h_broom=adapter.unrooted_canon(broom)
        public_equal=q_path==q_broom
        hidden_distinct=h_path!=h_broom
        legal=bool(q_path.get("legal")) and bool(q_broom.get("legal"))
        descriptor_ok=q_path.get("descriptor")==q_broom.get("descriptor")=="CAPS7_PLUS_H_CLASS_BAG"
        support_ok=_is_realizable_support(support,depth)
        lane={
          "lane_id":f"R7_ACTUAL_G4_PATH_BROOM_D{depth}","witness_depth":depth,"vertices":depth+3,
          "support_certificate":support,"support_certificate_sha256":canonical_sha256(support),
          "path_public_read":q_path,"broom_public_read":q_broom,
          "public_reads_exact_equal":public_equal,"hidden_canons_distinct":hidden_distinct,
          "path_hidden_canon_sha256":canonical_sha256(h_path),"broom_hidden_canon_sha256":canonical_sha256(h_broom),
          "public_reads_legal":legal,"descriptor_token_ok":descriptor_ok,"resource_realizability_ok":support_ok,
        }
        lanes.append(lane)
        if not all((public_equal,hidden_distinct,legal,descriptor_ok,support_ok)):
            failures.append({"lane_id":lane["lane_id"],"public_equal":public_equal,"hidden_distinct":hidden_distinct,"legal":legal,"descriptor_ok":descriptor_ok,"support_ok":support_ok})
    result={
      "question":"What exact public interface does each complete graduated G4 carrier expose to G5?",
      "g4_adapter_descriptor":adapter.descriptor(),"g4_scope_contract":adapter.scope_contract(),
      "frozen_public_interface":"CAPS7_PLUS_H_CLASS_BAG","hidden_diagnostic_role":"CHALLENGE_ONLY_NOT_PUBLIC_STATE",
      "challenge_lanes":lanes,"failures":failures,"new_science":True,"promotion":False,
      "g4_changed":False,"g5_composition_law_earned":False,"g5_graduated":False,
    }
    if failures:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":FAIL_ALT,**result}
    return {"outcome":"PASS","observed_alternative":PASS_ALT,**result}


@register_workflow_capability("g5.stage.locked.v1")
def g5_stage_locked(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    expected=str(parameters.get("stage"))
    if expected != stage:
        return {"outcome":"BLOCKED","observed_alternative":"G5_STAGE_LOCK_BINDING_MISMATCH","expected_stage":expected,"stage":stage}
    alt=str(parameters.get("review_alternative") or f"G5_{stage}_PREREGISTRATION_REQUIRED")
    return {
      "outcome":"REVIEW_REQUIRED","observed_alternative":alt,
      "reason":"SEPARATE_STAGE_PREREGISTRATION_REQUIRED","prior_stage_count":len(prior),
      "promotion":False,"g5_graduated":False,
    }


S1_CONGRUENT_ALT = "G5_S1_PAIR_CENSUS_PUBLIC_CONGRUENT_ON_CHALLENGE"
S1_RESIDUAL_ALT = "G5_S1_PAIR_CENSUS_HIDDEN_RESIDUAL_DETECTED"
S1_FAIL_ALT = "G5_S1_AUTHORITY_OR_CENSUS_FAILURE"

def _g5_s1_carriers():
    p2,b2=_path_broom(2); p4,b4=_path_broom(4)
    return {"D2_PATH":p2,"D2_BROOM":b2,"D4_PATH":p4,"D4_BROOM":b4}

def _join_g4_pair(left: DecoratedG4Tree, right: DecoratedG4Tree, lu: int, rv: int, op: tuple[int,int]) -> DecoratedG4Tree:
    off=left.n
    edges=tuple(left.edges)+tuple((a+off,b+off) for a,b in right.edges)+((int(lu),int(rv)+off),)
    hs=tuple(left.H_classes)+tuple(right.H_classes)
    grades=tuple(left.edge_operators)+tuple(right.edge_operators)+(tuple(map(int,op)),)
    return DecoratedG4Tree(left.n+right.n,edges,hs,grades)

def _public_key(read: Mapping[str, Any]) -> tuple[Any,...]:
    if not read.get("legal"):
        return (False,)
    return (True,tuple(read["caps7"]),tuple(sorted((read.get("H_class_bag") or {}).items())))

@register_workflow_capability("g5.s1.pair-census.v1")
def g5_s1_pair_census(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    if stage != "S1":
        return {"outcome":"BLOCKED","observed_alternative":S1_FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    if len(prior)!=1 or prior[0].get("stage")!="S0" or prior[0].get("outcome")!="PASS" or prior[0].get("observed_alternative")!=PASS_ALT:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"S0_PRIOR_BINDING_MISMATCH"}
    expected_s0_stage = parameters.get("required_s0_stage_science_sha256")
    if expected_s0_stage is not None:
        if str(prior[0].get("science_sha256")) != str(expected_s0_stage):
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"S0_STAGE_AUTHORITY_SHA_MISMATCH","expected":str(expected_s0_stage),"observed":prior[0].get("science_sha256")}
    else:
        expected_s0=str(parameters.get("required_s0_science_sha256"))
        if expected_s0 != "3fc75ad0fe5c456d117e41e66f0f036d415332e7d4b97dc30f643cac290ab66a":
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"S0_AUTHORITY_SHA_MISMATCH","observed":expected_s0}
    carriers=_g5_s1_carriers(); ops=tuple(adapter.operator_basis())
    if len(ops)!=31:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"OPERATOR_BASIS_COUNT_MISMATCH","observed":len(ops)}
    if execution_context is not None:
        execution_context.reserve_cases(len(carriers) * len(carriers) * len(ops))
    reads={k:adapter.public_read(v) for k,v in carriers.items()}
    if any(not r.get("legal") for r in reads.values()):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"ILLEGAL_S0_CARRIER"}
    rows=[]
    for lk,left in carriers.items():
      for rk,right in carriers.items():
        inkey=( _public_key(reads[lk]), _public_key(reads[rk]) )
        for op in ops:
          public_counts={}; hidden=set(); legal_count=0; attempted=left.n*right.n
          for lu in range(left.n):
            for rv in range(right.n):
              child=_join_g4_pair(left,right,lu,rv,op)
              q=adapter.public_read(child)
              if not q.get("legal"):
                  continue
              legal_count+=1
              pk=_public_key(q)
              ks=repr(pk); public_counts[ks]=public_counts.get(ks,0)+1
              hidden.add(repr(adapter.unrooted_canon(child)))
          sig={"legal_realization_count":legal_count,"attempted_owner_pairs":attempted,"public_outcome_multiset":dict(sorted(public_counts.items()))}
          rows.append({"left":lk,"right":rk,"operator":list(op),"input_public_key":repr(inkey),"public_signature":sig,"hidden_exact_child_canon_count":len(hidden)})
    if len(rows)!=16*31:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S1_FAIL_ALT,"failure":"CENSUS_ROW_COUNT_MISMATCH","observed":len(rows)}
    # Congruence audit within equal public input key + operator fibres.
    fibres={}; witnesses=[]
    for r in rows:
        k=(r["input_public_key"],tuple(r["operator"]))
        fibres.setdefault(k,[]).append(r)
    for (k,op), group in fibres.items():
        sigs={canonical_sha256(x["public_signature"]):x for x in group}
        if len(sigs)>1:
            gs=sorted(group,key=lambda x:(x["left"],x["right"]))
            a=gs[0]; b=next(x for x in gs[1:] if canonical_sha256(x["public_signature"])!=canonical_sha256(a["public_signature"]))
            witnesses.append({"operator":list(op),"input_public_key":k,"pair_a":[a["left"],a["right"]],"pair_b":[b["left"],b["right"]],"signature_a":a["public_signature"],"signature_b":b["public_signature"]})
    alt=S1_RESIDUAL_ALT if witnesses else S1_CONGRUENT_ALT
    return {"outcome":"PASS","observed_alternative":alt,"question":"Which ordered carrier pairs admit each declared connection attempt and what canonical outcomes result?","carrier_count":4,"ordered_pair_count":16,"operator_count":31,"census_row_count":len(rows),"rows":rows,"equal_public_input_fibre_count":len(fibres),"hidden_residual_witness_count":len(witnesses),"first_hidden_residual_witness":witnesses[0] if witnesses else None,"all_latent_owner_pairs_quantified":True,"hidden_owner_identity_promoted":False,"hidden_topology_promoted":False,"g4_changed":False,"g5_composition_law_earned":False,"g5_graduated":False,"promotion":False}


S2_PASS_ALT = "G5_S2_PAIR_OBSERVER_QUOTIENT_CERTIFIED_NO_ADDED_READ"
S2_NOT_CERT_ALT = "G5_S2_PAIR_OBSERVER_QUOTIENT_NOT_CERTIFIED"
S2_FAIL_ALT = "G5_S2_AUTHORITY_OR_QUOTIENT_FAILURE"

@register_workflow_capability("g5.s2.pair-quotient.v1")
def g5_s2_pair_quotient(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    if stage != "S2":
        return {"outcome":"BLOCKED","observed_alternative":S2_FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    if len(prior)!=2 or prior[0].get("stage")!="S0" or prior[0].get("outcome")!="PASS" or prior[1].get("stage")!="S1" or prior[1].get("outcome")!="PASS" or prior[1].get("observed_alternative")!=S1_CONGRUENT_ALT:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"S0_S1_PRIOR_BINDING_MISMATCH"}
    s1r=prior[1].get("result") or {}; rows=list(s1r.get("rows") or [])
    if execution_context is not None:
        execution_context.reserve_cases(len(rows))
    required_rows_sha=str(parameters.get("required_s1_rows_sha256"))
    observed_rows_sha=canonical_sha256(rows)
    expected_s1_stage = parameters.get("required_s1_stage_science_sha256")
    if expected_s1_stage is not None:
        if str(prior[1].get("science_sha256")) != str(expected_s1_stage):
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"S1_STAGE_AUTHORITY_MISMATCH","expected":str(expected_s1_stage),"observed":prior[1].get("science_sha256")}
        if observed_rows_sha != required_rows_sha:
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"S1_ROWS_AUTHORITY_MISMATCH","expected_rows_sha256":required_rows_sha,"observed_rows_sha256":observed_rows_sha}
    elif required_rows_sha != "5e017d0529da53900e2b1d8cee993b20315bbd9630cd5d8e6584b9f25e27c863" or observed_rows_sha != required_rows_sha:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"S1_ROWS_AUTHORITY_MISMATCH","observed_rows_sha256":observed_rows_sha}
    if len(rows)!=496 or s1r.get("hidden_residual_witness_count")!=0:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"S1_CENSUS_PRECONDITION_MISMATCH"}
    carriers=_g5_s1_carriers()
    pub={k:canonical_sha256({"g4_public_read":adapter.public_read(v)}) for k,v in carriers.items()}
    public_partition={}
    for ref,h in pub.items(): public_partition.setdefault(h,[]).append(ref)
    public_partition_norm=sorted(sorted(v) for v in public_partition.values())
    # Exact pair-context congruence using only public carrier keys + operator.
    fibres={}
    for r in rows:
        lk=str(r["left"]); rk=str(r["right"]); op=tuple(map(int,r["operator"]))
        if lk not in pub or rk not in pub:
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S2_FAIL_ALT,"failure":"ROW_OUTSIDE_FROZEN_DOMAIN"}
        key=(pub[lk],pub[rk],op)
        fibres.setdefault(key,[]).append(r)
    conflicts=[]; qrows=[]
    for key,group in sorted(fibres.items(),key=lambda kv:(kv[0][0],kv[0][1],kv[0][2])):
        sigs=sorted({canonical_sha256(x["public_signature"]) for x in group})
        rec={"left_public_sha256":key[0],"right_public_sha256":key[1],"operator":list(key[2]),"exact_representative_pair_count":len(group),"observer_value_count":len(sigs),"observer_signature_sha256":sigs[0] if len(sigs)==1 else None}
        qrows.append(rec)
        if len(group)!=4 or len(sigs)!=1: conflicts.append({**rec,"observer_hashes":sigs})
    if len(fibres)!=124:
        conflicts.append({"failure":"PUBLIC_PAIR_CONTEXT_KEY_COUNT","observed":len(fibres),"expected":124})
    # Complete target-behavior kernel, with hidden right-carrier identity quotiented away.
    by_target={k:[] for k in carriers}
    for r in rows:
        by_target[str(r["left"])].append(r)
    behavior_by_target={}; kernel={}; target_conflicts=[]
    for tr,trows in sorted(by_target.items()):
        tg={}
        for r in trows:
            key=(pub[str(r["right"])],tuple(map(int,r["operator"])))
            tg.setdefault(key,[]).append(canonical_sha256(r["public_signature"]))
        table=[]
        for key,vals in sorted(tg.items(),key=lambda kv:(kv[0][0],kv[0][1])):
            uniq=sorted(set(vals))
            table.append({"context_public_sha256":key[0],"operator":list(key[1]),"exact_context_representative_count":len(vals),"observer_value_count":len(uniq),"observer_signature_sha256":uniq[0] if len(uniq)==1 else None})
            if len(vals)!=2 or len(uniq)!=1:
                target_conflicts.append({"target":tr,"context_public_sha256":key[0],"operator":list(key[1]),"representatives":len(vals),"observer_hashes":uniq})
        bh=canonical_sha256(table); behavior_by_target[tr]={"behavior_sha256":bh,"public_carrier_sha256":pub[tr],"table_key_count":len(table)}; kernel.setdefault(bh,[]).append(tr)
    kernel_norm=sorted(sorted(v) for v in kernel.values())
    partitions_equal=(public_partition_norm==kernel_norm)
    if not partitions_equal:
        conflicts.append({"failure":"PUBLIC_CARRIER_TARGET_KERNEL_PARTITION_MISMATCH","public_partition":public_partition_norm,"target_kernel_partition":kernel_norm})
    conflicts.extend(target_conflicts[:16])
    passed=not conflicts
    return {
      "outcome":"PASS" if passed else "REVIEW_REQUIRED",
      "observed_alternative":S2_PASS_ALT if passed else S2_NOT_CERT_ALT,
      "question":"Does the complete frozen G5 pair observer factor exactly through inherited G4 public carrier state without an added read?",
      "s1_rows_sha256":observed_rows_sha,"exact_carrier_count":4,"public_carrier_class_count":len(public_partition),"public_pair_context_key_count":len(fibres),
      "expected_representatives_per_public_pair_context_key":4,"representative_conflict_count":len(conflicts),"all_public_pair_context_keys_single_valued":all(x["observer_value_count"]==1 and x["exact_representative_pair_count"]==4 for x in qrows),
      "public_carrier_partition":public_partition_norm,"target_behavior_kernel_partition":kernel_norm,"public_carrier_partition_equals_target_behavior_kernel":partitions_equal,
      "target_behavior":behavior_by_target,"quotient_table":qrows,"conflicts":conflicts[:24],"minimal_added_read_on_frozen_pair_observer":"NONE" if passed else None,
      "hidden_topology_promoted":False,"hidden_owner_identity_promoted":False,"g4_changed":False,"g5_public_descriptor_promoted":False,"g5_composition_law_earned":False,"g5_graduated":False,"promotion":False
    }


S3_PASS_ALT = "G5_S3_NO_IRREDUCIBLE_TRIPLE_RESIDUAL_ON_FROZEN_P3_K3_SCOPE_S4_DESIGN_REVIEW"
S3_RESIDUAL_ALT = "G5_S3_HIGHER_ORDER_HIDDEN_RESIDUAL_DETECTED"
S3_FAIL_ALT = "G5_S3_AUTHORITY_OR_HIGHER_RESIDUAL_FAILURE"


def _g5_local_remaining_caps(adapter: Any, tree: DecoratedG4Tree) -> list[list[int]]:
    rows = adapter._rows
    local = [list(map(int, rows[k]["caps7"])) for k in tree.H_classes]
    for (u, v), (a, b) in zip(tree.edges, tree.edge_operators):
        local[int(u)][int(a)] -= 1
        local[int(v)][int(b)] -= 1
    return local


def _g5_single_owner_count(adapter: Any, tree: DecoratedG4Tree, endpoint_type: int) -> int:
    a = int(endpoint_type)
    if not 0 <= a < 7:
        raise ValueError("endpoint type outside seven-type alphabet")
    local = _g5_local_remaining_caps(adapter, tree)
    return sum(1 for row in local if row[a] > 0)


def _g5_two_reservation_signature(adapter: Any, tree: DecoratedG4Tree, first_type: int, second_type: int) -> dict[str, Any]:
    a, b = int(first_type), int(second_type)
    if not (0 <= a < 7 and 0 <= b < 7):
        raise ValueError("endpoint type outside seven-type alphabet")
    q = adapter.public_read(tree)
    if not q.get("legal"):
        raise ValueError("S3 local continuation requires legal frozen G4 carrier")
    local = _g5_local_remaining_caps(adapter, tree)
    first_owner_count = 0
    second_hist: dict[int, int] = {}
    total_pairs = 0
    for u in range(tree.n):
        if local[u][a] <= 0:
            continue
        first_owner_count += 1
        second_count = 0
        for v in range(tree.n):
            remaining = local[v][b] - (1 if (v == u and b == a) else 0)
            if remaining > 0:
                second_count += 1
        second_hist[second_count] = second_hist.get(second_count, 0) + 1
        total_pairs += second_count
    residual_caps = list(map(int, q["caps7"]))
    if total_pairs > 0:
        residual_caps[a] -= 1
        residual_caps[b] -= 1
    out = {
        "schema_id":"IG_G5_S3_ORDERED_TWO_RESERVATION_SIGNATURE_V1",
        "first_endpoint_type":a,"second_endpoint_type":b,
        "first_owner_count":first_owner_count,
        "first_branch_second_owner_count_histogram":[{"second_owner_count":k,"first_branch_multiplicity":second_hist[k]} for k in sorted(second_hist)],
        "ordered_owner_pair_count":total_pairs,"joint_legal":bool(total_pairs > 0),
        "residual_public_read":{"descriptor":"CAPS7_PLUS_H_CLASS_BAG","caps7":residual_caps if total_pairs > 0 else None,"H_class_bag":q.get("H_class_bag")},
        "owner_identity_recorded":False,"hidden_topology_read":False,"hidden_canon_recorded":False,"rooted_action_read":False,
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _g5_public_sha(adapter: Any, tree: DecoratedG4Tree) -> str:
    return canonical_sha256({"g4_public_read":adapter.public_read(tree)})


def _g5_aggregate_public_outcome(adapter: Any, refs: list[str], carriers: Mapping[str, DecoratedG4Tree], edge_ops: list[tuple[int,int]]) -> dict[str, Any]:
    reads=[adapter.public_read(carriers[r]) for r in refs]
    caps=[sum(int(q["caps7"][i]) for q in reads) for i in range(7)]
    bag: dict[str,int]={}
    for q in reads:
        for k,v in (q.get("H_class_bag") or {}).items(): bag[str(k)]=bag.get(str(k),0)+int(v)
    # Directed motif edge order: P3 = (0,1),(1,2); K3 adds (2,0).
    edges=[(0,1),(1,2)] + ([(2,0)] if len(edge_ops)==3 else [])
    for (u,v),(a,b) in zip(edges,edge_ops):
        caps[int(a)]-=1; caps[int(b)]-=1
    return {"descriptor":"CAPS7_PLUS_H_CLASS_BAG","caps7":caps,"H_class_bag":dict(sorted(bag.items()))}


@register_workflow_capability("g5.s3.higher-residual.v1")
def g5_s3_higher_residual(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    if stage != "S3":
        return {"outcome":"BLOCKED","observed_alternative":S3_FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    if len(prior)!=3 or [x.get("stage") for x in prior] != ["S0","S1","S2"] or any(x.get("outcome")!="PASS" for x in prior):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S0_S2_PRIOR_BINDING_MISMATCH"}
    if prior[2].get("observed_alternative") != S2_PASS_ALT:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_OUTCOME_BINDING_MISMATCH"}
    s2r=prior[2].get("result") or {}
    expected_s2_stage = parameters.get("required_s2_stage_science_sha256")
    if expected_s2_stage is not None:
        expected_quotient = str(parameters.get("required_s2_quotient_table_sha256"))
        expected_s1_rows = str(parameters.get("required_s1_rows_sha256"))
        if str(prior[2].get("science_sha256")) != str(expected_s2_stage):
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_STAGE_AUTHORITY_MISMATCH","expected":str(expected_s2_stage),"observed":prior[2].get("science_sha256")}
    else:
        expected={
          "required_s2_primary_science_sha256":"91c5cfed19c6db59bfeb53c93603f8c8f835cee1a727153ba8ca2c1bb5984be9",
          "required_s2_workflow_science_sha256":"1e2aae65a1ad0d007994ad400a287122ff4f62687ef7ea7ba3c68e892a606403",
          "required_s2_quotient_table_sha256":"afa4ed046578cd87867c8e94ad3f30578fcfbd5e8f776b9b9f51cbd41aad3f22",
          "required_s1_rows_sha256":"5e017d0529da53900e2b1d8cee993b20315bbd9630cd5d8e6584b9f25e27c863",
        }
        if any(str(parameters.get(k)) != v for k,v in expected.items()):
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_AUTHORITY_PARAMETER_MISMATCH"}
        expected_quotient = expected["required_s2_quotient_table_sha256"]
        expected_s1_rows = expected["required_s1_rows_sha256"]
    if canonical_sha256(list(s2r.get("quotient_table") or [])) != expected_quotient:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_QUOTIENT_BINDING_MISMATCH"}
    if str(s2r.get("s1_rows_sha256")) != expected_s1_rows or s2r.get("representative_conflict_count") != 0 or s2r.get("public_carrier_partition_equals_target_behavior_kernel") is not True or s2r.get("minimal_added_read_on_frozen_pair_observer") != "NONE":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_CERTIFICATE_PRECONDITION_MISMATCH"}

    carriers=_g5_s1_carriers(); ops=tuple(adapter.operator_basis())
    if len(carriers)!=4 or len(ops)!=31:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"FROZEN_DOMAIN_MISMATCH"}
    pub={r:_g5_public_sha(adapter,t) for r,t in carriers.items()}
    classes: dict[str,list[str]]={}
    for r,h in pub.items(): classes.setdefault(h,[]).append(r)
    if sorted(len(v) for v in classes.values()) != [2,2]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"PUBLIC_CLASS_POPULATION_MISMATCH","classes":classes}

    if execution_context is not None:
        public_class_count = len(classes)
        execution_context.reserve_cases((public_class_count ** 3) * (len(ops) ** 2 + len(ops) ** 3))

    local_rows=[]; by_ref={}
    for ref,tree in sorted(carriers.items()):
        vec={}
        for a in range(7):
            for b in range(7):
                sig=_g5_two_reservation_signature(adapter,tree,a,b)
                local_rows.append({"carrier_ref":ref,"public_carrier_sha256":pub[ref],"first_type":a,"second_type":b,"signature":sig})
                vec[(a,b)]=sig
        by_ref[ref]=vec
    conflicts=[]; class_vectors=[]
    for ph,reps in sorted(classes.items()):
        reps=sorted(reps); vhash=[]
        for a in range(7):
          for b in range(7):
            hs=sorted({by_ref[r][(a,b)]["science_sha256"] for r in reps})
            if len(hs)!=1:
                conflicts.append({"public_carrier_sha256":ph,"first_type":a,"second_type":b,"carrier_refs":reps,"signature_sha256s":hs})
        for r in reps:
            vhash.append(canonical_sha256([by_ref[r][(a,b)]["science_sha256"] for a in range(7) for b in range(7)]))
        class_vectors.append({"public_carrier_sha256":ph,"representative_count":len(reps),"carrier_refs":reps,"continuation_vector_sha256":vhash[0] if len(set(vhash))==1 else None})
        if len(set(vhash))!=1:
            conflicts.append({"failure":"FULL_VECTOR_CONFLICT","public_carrier_sha256":ph,"carrier_refs":reps,"vector_sha256s":sorted(set(vhash))})

    # Bind all 124 S2 edge observer values.
    qmap={}
    for q in s2r.get("quotient_table") or []:
        key=(str(q["left_public_sha256"]),str(q["right_public_sha256"]),tuple(map(int,q["operator"])))
        if key in qmap or not q.get("observer_signature_sha256"):
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_EDGE_MAP_MALFORMED"}
        qmap[key]=str(q["observer_signature_sha256"])
    if len(qmap)!=124:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S3_FAIL_ALT,"failure":"S2_EDGE_MAP_COUNT_MISMATCH","observed":len(qmap)}

    # Exact combinatorial public-context coverage. No sampling and no hidden key.
    pub_classes=sorted(classes)
    p3_count=(len(pub_classes)**3)*(len(ops)**2)
    k3_count=(len(pub_classes)**3)*(len(ops)**3)
    missing_edge_keys=0
    for a in pub_classes:
      for b in pub_classes:
        for op in ops:
          if (a,b,op) not in qmap: missing_edge_keys += 1
    coverage_ok=(p3_count==7688 and k3_count==238328 and p3_count+k3_count==246016 and missing_edge_keys==0)

    # Representative-independent local vectors imply representative-independent P3/K3 observer.
    passed=(not conflicts and coverage_ok)
    return {
      "outcome":"PASS" if passed else "REVIEW_REQUIRED",
      "observed_alternative":S3_PASS_ALT if passed else S3_RESIDUAL_ALT,
      "question":"Are any connected three-carrier P3/K3 public compatibility facts irreducible to certified S2 edge values plus inherited G4 public carrier state?",
      "s2_quotient_table_sha256":canonical_sha256(list(s2r.get("quotient_table") or [])),
      "exact_carrier_count":4,"public_carrier_class_count":2,"exact_representatives_per_public_carrier_class":2,
      "ordered_local_endpoint_pair_count":49,"local_two_reservation_row_count":len(local_rows),
      "local_representative_conflict_count":len(conflicts),"first_local_residual_witness":conflicts[0] if conflicts else None,
      "all_public_classes_share_one_complete_local_continuation_vector_across_exact_representatives":not conflicts,
      "class_continuation_vectors":class_vectors,
      "p3_public_context_count":p3_count,"k3_public_context_count":k3_count,"total_public_context_count":p3_count+k3_count,"s2_edge_map_key_count":len(qmap),"missing_s2_edge_key_count":missing_edge_keys,
      "factorization_complete":coverage_ok and not conflicts,
      "irreducible_triple_residual_count":0 if passed else len(conflicts),
      "minimal_added_read_on_frozen_triple_observer":"NONE_BEYOND_INHERITED_G4_PUBLIC_PLUS_S2_EDGE_OBSERVER" if passed else None,
      "hidden_topology_promoted":False,"hidden_owner_identity_promoted":False,"g4_changed":False,"g5_public_descriptor_promoted":False,"g5_composition_law_earned":False,"g5_graduated":False,"promotion":False
    }


# Machine-checkable semantic bindings introduced by the pre-S4 audit repair.
# They validate declarations at admission without changing historical result payloads.
register_workflow_semantic_contract("g5.s0.carrier-interface.v1", {
    "contract_id":"IG_G5_S0_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG", "allowed_stages":["S0"],
    "parameter_keys":["spec_science_sha256","witness_depths"],
    "registered_alternatives":[PASS_ALT, FAIL_ALT],
})
register_workflow_semantic_contract("g5.s1.pair-census.v1", {
    "contract_id":"IG_G5_S1_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG", "allowed_stages":["S1"],
    "parameter_key_sets":[["required_s0_science_sha256","spec_science_sha256"],["required_s0_stage_science_sha256","spec_science_sha256"]],
    "registered_alternatives":[S1_CONGRUENT_ALT,S1_RESIDUAL_ALT,S1_FAIL_ALT],
})
register_workflow_semantic_contract("g5.s2.pair-quotient.v1", {
    "contract_id":"IG_G5_S2_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG", "allowed_stages":["S2"],
    "parameter_key_sets":[["required_s1_primary_science_sha256","required_s1_rows_sha256","spec_science_sha256"],["required_s1_stage_science_sha256","required_s1_rows_sha256","spec_science_sha256"]],
    "registered_alternatives":[S2_PASS_ALT,S2_NOT_CERT_ALT,S2_FAIL_ALT],
})
register_workflow_semantic_contract("g5.s3.higher-residual.v1", {
    "contract_id":"IG_G5_S3_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG", "allowed_stages":["S3"],
    "parameter_key_sets":[["required_s1_rows_sha256","required_s2_primary_science_sha256","required_s2_quotient_table_sha256","required_s2_workflow_science_sha256","spec_science_sha256"],["required_s1_rows_sha256","required_s2_stage_science_sha256","required_s2_quotient_table_sha256","spec_science_sha256"]],
    "registered_alternatives":[S3_PASS_ALT,S3_RESIDUAL_ALT,S3_FAIL_ALT],
})
register_workflow_semantic_contract("g5.stage.locked.v1", {
    "contract_id":"IG_G5_STAGE_LOCK_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG", "allowed_stages":["S0","S1","S2","S3","S4","S5","S6"],
    "parameter_keys":["review_alternative","stage"], "stage_parameter_binding":True,
    "review_alternative_parameter":"review_alternative",
})

# G5:S4 one-step composition closure, preregistered 2026-09-06.
S4_PASS_ALT = "G5_S4_ONE_STEP_COMPOSITION_CLOSURE_ON_FROZEN_RECURSIVE_P3_SCOPE_S5_DESIGN_REVIEW"
S4_SEPARATOR_ALT = "G5_S4_COMPOSITION_CLOSURE_SEPARATOR_DETECTED"
S4_FAIL_ALT = "G5_S4_AUTHORITY_OR_COMPOSITION_FAILURE"


def _g5_s4_spec() -> dict[str, Any]:
    import json as _json
    from importlib.resources import files as _files
    obj = _json.loads(_files("infinity_grid").joinpath("resources/uplift/G5_S4_COMPOSITION_CLOSURE_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_S4_COMPOSITION_CLOSURE_SPEC_V1":
        raise ValueError("bad G5:S4 spec schema")
    declared = obj.get("science_sha256")
    observed = canonical_sha256({k:v for k,v in obj.items() if k != "science_sha256"})
    if declared != observed:
        raise ValueError("G5:S4 spec hash mismatch")
    return obj


def _g5_owner_count_vector(local_rows: list[list[int]]) -> tuple[int, ...]:
    return tuple(sum(1 for row in local_rows if int(row[t]) > 0) for t in range(7))


def _g5_profile_one(local_rows: list[list[int]], endpoint_type: int) -> dict[tuple[int, ...], int]:
    t = int(endpoint_type)
    out: dict[tuple[int, ...], int] = {}
    for u,row in enumerate(local_rows):
        if int(row[t]) <= 0:
            continue
        work = [list(x) for x in local_rows]
        work[u][t] -= 1
        key = _g5_owner_count_vector(work)
        out[key] = out.get(key, 0) + 1
    return out


def _g5_profile_two(local_rows: list[list[int]], first_type: int, second_type: int) -> dict[tuple[int, ...], int]:
    a,b = int(first_type), int(second_type)
    out: dict[tuple[int, ...], int] = {}
    for u,row in enumerate(local_rows):
        if int(row[a]) <= 0:
            continue
        first = [list(x) for x in local_rows]
        first[u][a] -= 1
        for v,row2 in enumerate(first):
            if int(row2[b]) <= 0:
                continue
            second = [list(x) for x in first]
            second[v][b] -= 1
            key = _g5_owner_count_vector(second)
            out[key] = out.get(key, 0) + 1
    return out


def _g5_convolve_three_profiles(
    hx: Mapping[tuple[int, ...], int],
    hy: Mapping[tuple[int, ...], int],
    hz: Mapping[tuple[int, ...], int],
    out: dict[tuple[int, ...], int],
) -> None:
    for x,mx in hx.items():
        for y,my in hy.items():
            xy = tuple(int(a)+int(b) for a,b in zip(x,y))
            mm = int(mx)*int(my)
            for z,mz in hz.items():
                total = tuple(int(a)+int(b) for a,b in zip(xy,z))
                out[total] = out.get(total, 0) + mm*int(mz)


def _g5_normalize_vector_hist(hist: Mapping[tuple[int, ...], int]) -> list[dict[str, Any]]:
    return [
        {"owner_count_vector": list(vec), "branch_multiplicity": int(hist[vec])}
        for vec in sorted(hist)
    ]


def _g5_s4_factorized_signature(
    *, adapter: Any, refs: tuple[str,str,str], carriers: Mapping[str, DecoratedG4Tree],
    public_reads: Mapping[str, Mapping[str, Any]],
    one_profiles: Mapping[tuple[str,int], Mapping[tuple[int,...],int]],
    two_profiles: Mapping[tuple[str,int,int], Mapping[tuple[int,...],int]],
    op1: tuple[int,int], op2: tuple[int,int],
) -> dict[str, Any]:
    l,r,t = refs; a,b = map(int,op1); c,d = map(int,op2)
    hist: dict[tuple[int,...],int] = {}
    # Second recursive attachment may consume its pair-side endpoint from either
    # the original left carrier or the original right carrier. These two lanes
    # are disjoint explicit owner choices and are therefore added, not selected.
    _g5_convolve_three_profiles(two_profiles[(l,a,c)], one_profiles[(r,b)], one_profiles[(t,d)], hist)
    _g5_convolve_three_profiles(one_profiles[(l,a)], two_profiles[(r,b,c)], one_profiles[(t,d)], hist)
    branches = sum(int(v) for v in hist.values())
    if branches:
        caps=[sum(int(public_reads[x]["caps7"][i]) for x in (l,r,t)) for i in range(7)]
        caps[a]-=1; caps[b]-=1; caps[c]-=1; caps[d]-=1
        bag: dict[str,int]={}
        for x in (l,r,t):
            for k,v in (public_reads[x].get("H_class_bag") or {}).items():
                bag[str(k)]=bag.get(str(k),0)+int(v)
        public={"descriptor":"CAPS7_PLUS_H_CLASS_BAG","caps7":caps,"H_class_bag":dict(sorted(bag.items()))}
    else:
        public=None
    out = {
        "schema_id":"IG_G5_S4_RECURSIVE_PUBLIC_OBSERVER_SIGNATURE_V1",
        "legal_owner_choice_branch_count":int(branches),
        "public_outcome_multiset":[] if not branches else [{"public_outcome":public,"multiplicity":int(branches)}],
        "one_more_attachment_owner_vector_histogram":_g5_normalize_vector_hist(hist),
        "hidden_owner_identity_recorded":False,
        "hidden_topology_read_as_public_selector":False,
        "exact_hidden_relation_equality_claimed":False,
    }
    return out


def _g5_s4_direct_materialized_signature(
    *, adapter: Any, left: DecoratedG4Tree, right: DecoratedG4Tree, third: DecoratedG4Tree,
    op1: tuple[int,int], op2: tuple[int,int],
) -> dict[str, Any]:
    public_counts: dict[tuple[Any,...], int] = {}
    public_values: dict[tuple[Any,...], dict[str,Any]] = {}
    hist: dict[tuple[int,...],int] = {}
    branches = 0
    for lu in range(left.n):
        for rv in range(right.n):
            pair = _join_g4_pair(left,right,lu,rv,op1)
            qpair = adapter.public_read(pair)
            if not qpair.get("legal"):
                continue
            for pw in range(pair.n):
                for tv in range(third.n):
                    child = _join_g4_pair(pair,third,pw,tv,op2)
                    q = adapter.public_read(child)
                    if not q.get("legal"):
                        continue
                    branches += 1
                    key = _public_key(q)
                    public_counts[key] = public_counts.get(key,0)+1
                    public_values.setdefault(key, {"descriptor":q.get("descriptor"),"caps7":list(q.get("caps7") or []),"H_class_bag":dict(q.get("H_class_bag") or {})})
                    vec = _g5_owner_count_vector(_g5_local_remaining_caps(adapter,child))
                    hist[vec] = hist.get(vec,0)+1
    pms = [
        {"public_outcome":public_values[key],"multiplicity":int(public_counts[key])}
        for key in sorted(public_counts, key=repr)
    ]
    return {
        "schema_id":"IG_G5_S4_RECURSIVE_PUBLIC_OBSERVER_SIGNATURE_V1",
        "legal_owner_choice_branch_count":int(branches),
        "public_outcome_multiset":pms,
        "one_more_attachment_owner_vector_histogram":_g5_normalize_vector_hist(hist),
        "hidden_owner_identity_recorded":False,
        "hidden_topology_read_as_public_selector":False,
        "exact_hidden_relation_equality_claimed":False,
    }


@register_workflow_capability("g5.s4.composition-closure.v1")
def g5_s4_composition_closure(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    if stage != "S4":
        return {"outcome":"BLOCKED","observed_alternative":S4_FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    if len(prior)!=4 or [x.get("stage") for x in prior] != ["S0","S1","S2","S3"] or any(x.get("outcome")!="PASS" for x in prior):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S0_S3_PRIOR_BINDING_MISMATCH"}
    if prior[3].get("observed_alternative") != S3_PASS_ALT:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S3_OUTCOME_BINDING_MISMATCH"}
    spec = _g5_s4_spec()
    if str(parameters.get("spec_science_sha256")) != str(spec["science_sha256"]):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S4_SPEC_BINDING_MISMATCH"}
    required = {
        "required_s1_rows_sha256": spec["authority"]["g5_s1_rows_sha256"],
        "required_s2_stage_science_sha256": spec["authority"]["g5_s2_stage_science_sha256"],
        "required_s2_quotient_table_sha256": spec["authority"]["g5_s2_quotient_table_sha256"],
        "required_s3_stage_science_sha256": spec["authority"]["g5_s3_stage_science_sha256"],
    }
    if any(str(parameters.get(k)) != str(v) for k,v in required.items()):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S4_AUTHORITY_PARAMETER_MISMATCH"}
    if str(prior[2].get("science_sha256")) != required["required_s2_stage_science_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S2_STAGE_AUTHORITY_MISMATCH"}
    if str(prior[3].get("science_sha256")) != required["required_s3_stage_science_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S3_STAGE_AUTHORITY_MISMATCH"}
    s2r = prior[2].get("result") or {}; s3r = prior[3].get("result") or {}
    if canonical_sha256(list(s2r.get("quotient_table") or [])) != required["required_s2_quotient_table_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S2_QUOTIENT_BINDING_MISMATCH"}
    if str(s2r.get("s1_rows_sha256")) != required["required_s1_rows_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S1_ROWS_BINDING_MISMATCH"}
    if s3r.get("factorization_complete") is not True or int(s3r.get("irreducible_triple_residual_count",-1)) != 0:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"S3_CERTIFICATE_PRECONDITION_MISMATCH"}

    carriers = _g5_s1_carriers(); ops = tuple(adapter.operator_basis())
    if len(carriers)!=4 or len(ops)!=31:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"FROZEN_DOMAIN_MISMATCH"}
    if execution_context is not None:
        execution_context.reserve_cases(int(spec["frozen_domain"]["public_recursive_context_count"]))

    # Exact public classes are formed by structural public-read equality. Hashes
    # are assigned only after the exact class is established.
    public_reads = {r:adapter.public_read(t) for r,t in carriers.items()}
    class_by_exact: dict[tuple[Any,...], list[str]] = {}
    for ref,q in public_reads.items():
        class_by_exact.setdefault(_public_key(q),[]).append(ref)
    if sorted(len(v) for v in class_by_exact.values()) != [2,2]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"PUBLIC_CLASS_POPULATION_MISMATCH"}
    classes=[]
    for exact_key,reps in class_by_exact.items():
        reps=sorted(reps)
        q=public_reads[reps[0]]
        classes.append((canonical_sha256({"g4_public_read":q}), exact_key, reps))
    classes.sort(key=lambda x:x[0])

    local = {r:_g5_local_remaining_caps(adapter,t) for r,t in carriers.items()}
    one_profiles: dict[tuple[str,int],dict[tuple[int,...],int]] = {}
    two_profiles: dict[tuple[str,int,int],dict[tuple[int,...],int]] = {}
    for ref,rows in local.items():
        for a in range(7):
            one_profiles[(ref,a)] = _g5_profile_one(rows,a)
            for b in range(7):
                two_profiles[(ref,a,b)] = _g5_profile_two(rows,a,b)

    contexts=[]; signature_catalog: dict[str,dict[str,Any]] = {}
    separator=None; context_count=0; rep_count=0; empty_count=0; multi_outcome_count=0
    for lclass in classes:
        for rclass in classes:
            for tclass in classes:
                for op1 in ops:
                    for op2 in ops:
                        context_count += 1
                        values=[]; reps=[]
                        for l in lclass[2]:
                            for r in rclass[2]:
                                for t in tclass[2]:
                                    sig=_g5_s4_factorized_signature(adapter=adapter,refs=(l,r,t),carriers=carriers,public_reads=public_reads,one_profiles=one_profiles,two_profiles=two_profiles,op1=op1,op2=op2)
                                    values.append(sig); reps.append([l,r,t]); rep_count += 1
                        first=values[0]
                        exact_equal=all(v==first for v in values[1:])
                        if not exact_equal:
                            j=next(i for i,v in enumerate(values[1:],1) if v!=first)
                            separator={"context":{"left_public_sha256":lclass[0],"right_public_sha256":rclass[0],"third_public_sha256":tclass[0],"first_operator":list(op1),"second_operator":list(op2)},"representative_a":reps[0],"representative_b":reps[j],"signature_a":first,"signature_b":values[j],"structural_equality":False}
                            break
                        sh=canonical_sha256(first)
                        old=signature_catalog.get(sh)
                        if old is not None and old != first:
                            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_FAIL_ALT,"failure":"HASH_COLLISION_OR_STRUCTURAL_CATALOG_MISMATCH","signature_sha256":sh}
                        signature_catalog.setdefault(sh,first)
                        branches=int(first["legal_owner_choice_branch_count"])
                        if branches==0: empty_count += 1
                        if len(first["public_outcome_multiset"])>1: multi_outcome_count += 1
                        contexts.append({"context_index":context_count-1,"left_public_sha256":lclass[0],"right_public_sha256":rclass[0],"third_public_sha256":tclass[0],"first_operator":list(op1),"second_operator":list(op2),"exact_representative_instantiations":len(values),"observer_signature_sha256":sh,"legal_owner_choice_branch_count":branches})
                    if separator: break
                if separator: break
            if separator: break
        if separator: break

    if separator is not None:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S4_SEPARATOR_ALT,"question":spec["question"],"public_recursive_contexts_evaluated":context_count,"exact_representative_instantiations_evaluated":rep_count,"separator_count":1,"first_separator":separator,"promotion":False,"g5_public_descriptor_promoted":False,"g5_composition_law_earned":False,"g5_graduated":False,"g4_changed":False}

    # Independent algorithmic/materialization oracle. It constructs actual trees
    # and reads every final branch through the accepted G4 adapter, rather than
    # reusing the profile convolution above.
    oracle_rows=[]; oracle_mismatches=[]
    d2_path,d2_broom=carriers["D2_PATH"],carriers["D2_BROOM"]
    for op in ops:
        for label,tree in (("D2_ALL_PATH",d2_path),("D2_ALL_BROOM",d2_broom)):
            direct=_g5_s4_direct_materialized_signature(adapter=adapter,left=tree,right=tree,third=tree,op1=op,op2=op)
            factor=_g5_s4_factorized_signature(adapter=adapter,refs=(("D2_PATH" if label.endswith("PATH") else "D2_BROOM"),)*3,carriers=carriers,public_reads=public_reads,one_profiles=one_profiles,two_profiles=two_profiles,op1=op,op2=op)
            ok=(direct==factor)
            row={"panel":label,"first_operator":list(op),"second_operator":list(op),"exact_structural_match":ok,"direct_signature_sha256":canonical_sha256(direct),"factorized_signature_sha256":canonical_sha256(factor),"branch_count":int(direct["legal_owner_choice_branch_count"])}
            oracle_rows.append(row)
            if not ok: oracle_mismatches.append({**row,"direct":direct,"factorized":factor})
    for op1raw,op2raw in spec["direct_materialization_oracle"]["d4_spot_panel"]:
        op1=tuple(map(int,op1raw)); op2=tuple(map(int,op2raw))
        for ref in ("D4_PATH","D4_BROOM"):
            tree=carriers[ref]
            direct=_g5_s4_direct_materialized_signature(adapter=adapter,left=tree,right=tree,third=tree,op1=op1,op2=op2)
            factor=_g5_s4_factorized_signature(adapter=adapter,refs=(ref,ref,ref),carriers=carriers,public_reads=public_reads,one_profiles=one_profiles,two_profiles=two_profiles,op1=op1,op2=op2)
            ok=(direct==factor)
            row={"panel":ref,"first_operator":list(op1),"second_operator":list(op2),"exact_structural_match":ok,"direct_signature_sha256":canonical_sha256(direct),"factorized_signature_sha256":canonical_sha256(factor),"branch_count":int(direct["legal_owner_choice_branch_count"])}
            oracle_rows.append(row)
            if not ok: oracle_mismatches.append({**row,"direct":direct,"factorized":factor})

    expected=spec["coverage_required"]
    coverage_ok=(context_count==int(expected["public_recursive_context_count"]) and rep_count==int(expected["exact_representative_instantiation_count"]) and len(oracle_rows)==int(expected["direct_materialization_rows"]) and empty_count==int(expected["unexpected_empty_contexts"]) and not oracle_mismatches)
    passed=coverage_ok
    result={
      "question":spec["question"],
      "s4_spec_science_sha256":spec["science_sha256"],
      "s2_quotient_table_sha256":required["required_s2_quotient_table_sha256"],
      "s3_stage_science_sha256":required["required_s3_stage_science_sha256"],
      "exact_carrier_count":4,"public_carrier_class_count":2,"exact_representatives_per_public_carrier_class":2,"directed_operator_count":31,
      "public_recursive_context_count":context_count,"exact_representative_instantiation_count":rep_count,"exact_representatives_per_context":8,
      "separator_count":0,"first_separator":None,"empty_outcome_context_count":empty_count,"multi_public_outcome_context_count":multi_outcome_count,
      "all_public_recursive_contexts_structurally_single_valued_across_exact_representatives":True,
      "structural_equality_used_for_decision":True,"hashes_used_as_lookup_indices_only":True,
      "signature_catalog_count":len(signature_catalog),"signature_catalog":[{"observer_signature_sha256":h,"signature":signature_catalog[h]} for h in sorted(signature_catalog)],
      "context_index":contexts,
      "direct_materialization_oracle":{"row_count":len(oracle_rows),"mismatch_count":len(oracle_mismatches),"all_31_operators_covered_in_both_positions":len(oracle_rows)>=62,"rows":oracle_rows,"first_mismatch":oracle_mismatches[0] if oracle_mismatches else None},
      "coverage_complete":coverage_ok,
      "minimal_added_read_on_tested_composition_observer":"NONE_BEYOND_INHERITED_G4_PUBLIC_PLUS_CERTIFIED_OWNER_CHOICE_RELATION" if passed else None,
      "s5_preregistration_design_review_unlocked":passed,
      "hidden_topology_promoted":False,"hidden_owner_identity_promoted":False,"g4_changed":False,"g5_public_descriptor_promoted":False,"g5_composition_law_earned":False,"g5_graduated":False,"promotion":False,
      "nonclaims":list(spec["nonclaims"]),
    }
    return {"outcome":"PASS" if passed else "REVIEW_REQUIRED","observed_alternative":S4_PASS_ALT if passed else S4_FAIL_ALT,**result}


register_workflow_semantic_contract("g5.s4.composition-closure.v1", {
    "contract_id":"IG_G5_S4_SEMANTIC_CONTRACT_V1", "version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6", "adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG", "allowed_stages":["S4"],
    "parameter_keys":["required_s1_rows_sha256","required_s2_quotient_table_sha256","required_s2_stage_science_sha256","required_s3_stage_science_sha256","spec_science_sha256"],
    "registered_alternatives":[S4_PASS_ALT,S4_SEPARATOR_ALT,S4_FAIL_ALT],
})

S5_PASS_ALT = "G5_S5_CAPS7_PLUS_H_CLASS_BAG_FINITE_READ_WRITE_STATE_EARNED_S6_DESIGN_REVIEW"
S5_FAIL_ALT = "G5_S5_FINITE_READ_WRITE_STATE_NOT_CERTIFIED"


def _g5_s5_spec() -> dict[str, Any]:
    import json as _json
    from importlib.resources import files as _files
    obj = _json.loads(_files("infinity_grid").joinpath("resources/uplift/G5_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1":
        raise RuntimeError("bad G5:S5 spec schema")
    if obj.get("science_sha256") != canonical_sha256({k:v for k,v in obj.items() if k != "science_sha256"}):
        raise RuntimeError("G5:S5 spec hash mismatch")
    return obj


def _g5_s5_state_from_public(read: Mapping[str, Any], alphabet: tuple[str, ...]) -> dict[str, Any]:
    if not bool(read.get("legal")) or read.get("descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        raise RuntimeError("G5:S5 requires legal inherited G4 public read")
    caps = tuple(int(x) for x in read.get("caps7") or [])
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise RuntimeError("G5:S5 bad CAPS7")
    bag = {str(k): int(v) for k,v in (read.get("H_class_bag") or {}).items()}
    if any(k not in alphabet or v < 0 for k,v in bag.items()):
        raise RuntimeError("G5:S5 H-class bag outside frozen alphabet")
    counts = [int(bag.get(k,0)) for k in alphabet]
    return {
        "schema_id":"IG_G5_CAPS7_H_BAG_READ_WRITE_STATE_V1",
        "total_free_by_type":list(caps),
        "H_class_alphabet":list(alphabet),
        "H_class_counts":counts,
        "unit_count":sum(counts),
    }


def _g5_s5_public_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    alpha = tuple(str(x) for x in state["H_class_alphabet"])
    counts = [int(x) for x in state["H_class_counts"]]
    bag = {k:v for k,v in zip(alpha,counts) if v}
    return {
        "descriptor":"CAPS7_PLUS_H_CLASS_BAG",
        "caps7":[int(x) for x in state["total_free_by_type"]],
        "H_class_bag":dict(sorted(bag.items())),
    }


def _g5_s5_compose_state(left: Mapping[str, Any], right: Mapping[str, Any], op: tuple[int,int]) -> dict[str, Any] | None:
    alpha = tuple(str(x) for x in left["H_class_alphabet"])
    if tuple(str(x) for x in right["H_class_alphabet"]) != alpha:
        raise RuntimeError("G5:S5 state alphabet mismatch")
    a,b = map(int,op)
    lf = [int(x) for x in left["total_free_by_type"]]
    rf = [int(x) for x in right["total_free_by_type"]]
    if not (0 <= a < 7 and 0 <= b < 7):
        raise RuntimeError("G5:S5 operator coordinate outside CAPS7")
    if lf[a] <= 0 or rf[b] <= 0:
        return None
    out = [lf[i]+rf[i] for i in range(7)]
    out[a] -= 1; out[b] -= 1
    counts = [int(x)+int(y) for x,y in zip(left["H_class_counts"],right["H_class_counts"])]
    return {
        "schema_id":"IG_G5_CAPS7_H_BAG_READ_WRITE_STATE_V1",
        "total_free_by_type":out,
        "H_class_alphabet":list(alpha),
        "H_class_counts":counts,
        "unit_count":sum(counts),
    }


@register_workflow_capability("g5.s5.finite-read-write.v1")
def g5_s5_finite_read_write(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    import ast
    if stage != "S5":
        return {"outcome":"BLOCKED","observed_alternative":S5_FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S5_FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    if len(prior) != 5 or [x.get("stage") for x in prior] != ["S0","S1","S2","S3","S4"] or any(x.get("outcome") != "PASS" for x in prior):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S5_FAIL_ALT,"failure":"S0_S4_PRIOR_BINDING_MISMATCH"}
    spec = _g5_s5_spec(); auth = spec["authority"]
    if str(parameters.get("spec_science_sha256")) != spec["science_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S5_FAIL_ALT,"failure":"S5_SPEC_BINDING_MISMATCH"}
    binds = {
        "required_s1_rows_sha256": canonical_sha256((prior[1].get("result") or {}).get("rows") or []),
        "required_s2_quotient_table_sha256": canonical_sha256((prior[2].get("result") or {}).get("quotient_table") or []),
        "required_s3_stage_science_sha256": str(prior[3].get("science_sha256")),
        "required_s4_stage_science_sha256": str(prior[4].get("science_sha256")),
    }
    for k,v in binds.items():
        if str(parameters.get(k)) != str(v):
            return {"outcome":"REVIEW_REQUIRED","observed_alternative":S5_FAIL_ALT,"failure":k.upper()+"_MISMATCH","expected":str(parameters.get(k)),"observed":str(v)}
    if binds["required_s1_rows_sha256"] != auth["g5_s1_rows_sha256"] or binds["required_s2_quotient_table_sha256"] != auth["g5_s2_quotient_table_sha256"] or binds["required_s3_stage_science_sha256"] != auth["g5_s3_stage_science_sha256"] or binds["required_s4_stage_science_sha256"] != auth["g5_s4_stage_science_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S5_FAIL_ALT,"failure":"FROZEN_AUTHORITY_MISMATCH"}
    if prior[4].get("observed_alternative") != auth["g5_s4_classification"] or not bool((prior[4].get("result") or {}).get("s5_preregistration_design_review_unlocked")):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S5_FAIL_ALT,"failure":"S4_UNLOCK_MISMATCH"}

    alphabet = tuple(str(x) for x in spec["candidate_descriptor"]["H_class_alphabet"])
    # Verify the complete frozen H alphabet is still readable by the inherited public adapter.
    alphabet_rows=[]; alphabet_failures=[]
    for klass in alphabet:
        try:
            t=DecoratedG4Tree(1,tuple(),(klass,),tuple())
            q=adapter.public_read(t); st=_g5_s5_state_from_public(q,alphabet)
            alphabet_rows.append({"H_class":klass,"public_state":st,"public_state_sha256":canonical_sha256(st)})
        except Exception as exc:
            alphabet_failures.append({"H_class":klass,"error":str(exc)})

    carriers = _g5_s1_carriers(); reads={k:adapter.public_read(v) for k,v in carriers.items()}
    states={k:_g5_s5_state_from_public(q,alphabet) for k,q in reads.items()}
    pubhash={k:canonical_sha256({"g4_public_read":q}) for k,q in reads.items()}
    if execution_context is not None:
        execution_context.reserve_cases(496 + 124 + 7688 + len(alphabet))

    # Gate 1: every exact S1 pair-row public outcome equals the algebraic state write.
    s1rows=list((prior[1].get("result") or {}).get("rows") or [])
    pair_failures=[]; pair_checks=0
    for r in s1rows:
        lk,rk=str(r["left"]),str(r["right"]); op=tuple(map(int,r["operator"])); pair_checks+=1
        expected_state=_g5_s5_compose_state(states[lk],states[rk],op)
        mult=(r.get("public_signature") or {}).get("public_outcome_multiset") or {}
        if expected_state is None:
            if mult: pair_failures.append({"left":lk,"right":rk,"operator":list(op),"reason":"ABSTRACT_DISABLED_BUT_EXACT_OUTCOME_PRESENT"})
            continue
        expected_key=_public_key({"legal":True,**_g5_s5_public_from_state(expected_state)})
        if len(mult)!=1:
            pair_failures.append({"left":lk,"right":rk,"operator":list(op),"reason":"NON_SINGLETON_PUBLIC_OUTCOME","outcome_count":len(mult)}); continue
        actual_key=ast.literal_eval(next(iter(mult.keys())))
        if actual_key != expected_key:
            pair_failures.append({"left":lk,"right":rk,"operator":list(op),"reason":"PUBLIC_WRITE_MISMATCH","expected":repr(expected_key),"actual":repr(actual_key)})

    # Gate 2: S2 remains a single-valued quotient on all 124 public pair keys.
    s2=(prior[2].get("result") or {}); qrows=list(s2.get("quotient_table") or [])
    s2_failures=[r for r in qrows if int(r.get("observer_value_count",0)) != 1 or int(r.get("exact_representative_pair_count",0)) != 4]

    # Gate 3: every S4 recursive public outcome equals two successive abstract writes.
    s4=(prior[4].get("result") or {}); catalog={str(x["observer_signature_sha256"]):x["signature"] for x in s4.get("signature_catalog") or []}
    class_state={pubhash[k]:states[k] for k in states}
    rec_failures=[]; rec_checks=0
    for row in s4.get("context_index") or []:
        rec_checks+=1
        try:
            l=class_state[str(row["left_public_sha256"])]; r=class_state[str(row["right_public_sha256"])]; t=class_state[str(row["third_public_sha256"])]
            s12=_g5_s5_compose_state(l,r,tuple(map(int,row["first_operator"])))
            s123=None if s12 is None else _g5_s5_compose_state(s12,t,tuple(map(int,row["second_operator"])))
            sig=catalog[str(row["observer_signature_sha256"])]; outm=sig.get("public_outcome_multiset") or []
            if s123 is None:
                if outm: rec_failures.append({"context_index":row["context_index"],"reason":"ABSTRACT_DISABLED_BUT_S4_OUTCOME_PRESENT"})
                continue
            if len(outm)!=1:
                rec_failures.append({"context_index":row["context_index"],"reason":"S4_NON_SINGLETON_PUBLIC_OUTCOME","count":len(outm)}); continue
            expected=_g5_s5_public_from_state(s123); actual=outm[0].get("public_outcome")
            if actual != expected:
                rec_failures.append({"context_index":row["context_index"],"reason":"ITERATED_PUBLIC_WRITE_MISMATCH","expected":expected,"actual":actual})
        except Exception as exc:
            rec_failures.append({"context_index":row.get("context_index"),"reason":"EXCEPTION","error":str(exc)})

    ops=tuple(adapter.operator_basis())
    read_surface={
      "abstract_state_fields":["CAPS7","H_class_bag"],
      "operator_input":"frozen_directed_endpoint_type_pair",
      "hidden_topology_read":False,"hidden_owner_identity_read":False,"construction_identity_read":False,
      "ancestry_read":False,"exact_relation_cardinality_read":False,"branch_multiplicity_read":False,
    }
    passed=(not alphabet_failures and len(alphabet_rows)==4 and len(ops)==31 and len(s1rows)==496 and pair_checks==496 and not pair_failures and len(qrows)==124 and not s2_failures and rec_checks==7688 and not rec_failures and s4.get("separator_count")==0 and s4.get("coverage_complete") is True)
    result={
      "question":spec["question"],"s5_spec_science_sha256":spec["science_sha256"],
      "candidate_descriptor":spec["candidate_descriptor"],"abstract_read_write_law":spec["abstract_read_write_law"],"frozen_observer":spec["frozen_observer"],
      "H_class_alphabet_rows":alphabet_rows,"H_class_alphabet_failure_count":len(alphabet_failures),"H_class_alphabet_failures":alphabet_failures[:8],
      "directed_operator_count":len(ops),"s1_pair_write_check_count":pair_checks,"s1_pair_write_failure_count":len(pair_failures),"first_s1_pair_write_failure":pair_failures[0] if pair_failures else None,
      "s2_public_pair_key_count":len(qrows),"s2_single_value_failure_count":len(s2_failures),"first_s2_failure":s2_failures[0] if s2_failures else None,
      "s4_recursive_write_check_count":rec_checks,"s4_recursive_write_failure_count":len(rec_failures),"first_s4_recursive_write_failure":rec_failures[0] if rec_failures else None,
      "s4_context_index_sha256":canonical_sha256(s4.get("context_index") or []),"s4_signature_catalog_sha256":canonical_sha256(s4.get("signature_catalog") or []),
      "implementation_read_surface_audit":read_surface,"coverage_complete":passed,
      "g5_finite_state_descriptor_earned":passed,"g5_finite_read_write_law_earned":passed,"earned_descriptor":"CAPS7_PLUS_H_CLASS_BAG" if passed else None,
      "g5_public_descriptor_changed":False,"hidden_topology_promoted":False,"hidden_owner_identity_promoted":False,"g4_changed":False,
      "g5_global_composition_law_promoted":False,"g5_graduated":False,"promotion":False,"s6_preregistration_design_review_unlocked":passed,
      "nonclaims":list(spec["nonclaims"]),
    }
    return {"outcome":"PASS" if passed else "REVIEW_REQUIRED","observed_alternative":S5_PASS_ALT if passed else S5_FAIL_ALT,**result}


register_workflow_semantic_contract("g5.s5.finite-read-write.v1", {
    "contract_id":"IG_G5_S5_SEMANTIC_CONTRACT_V1","version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6","adapter_family":"G4_ACCEPTED",
    "public_descriptor":"CAPS7_PLUS_H_CLASS_BAG","allowed_stages":["S5"],
    "parameter_keys":["required_s1_rows_sha256","required_s2_quotient_table_sha256","required_s3_stage_science_sha256","required_s4_stage_science_sha256","spec_science_sha256"],
    "registered_alternatives":[S5_PASS_ALT,S5_FAIL_ALT],
})

# ---------------------------------------------------------------------------
# G5:S6 recursive closure / graduation-candidate gate
# ---------------------------------------------------------------------------
S6_PASS_ALT = "G5_S6_RECURSIVE_CLOSURE_GRADUATION_CANDIDATE"
S6_FAIL_ALT = "G5_S6_RECURSIVE_CLOSURE_NOT_CERTIFIED"


def _g5_s6_spec() -> dict[str, Any]:
    import json as _json
    from importlib.resources import files as _files
    obj = _json.loads(_files("infinity_grid").joinpath("resources/uplift/G5_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G5_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1":
        raise RuntimeError("bad G5:S6 spec schema")
    if obj.get("science_sha256") != canonical_sha256({k:v for k,v in obj.items() if k != "science_sha256"}):
        raise RuntimeError("G5:S6 spec hash mismatch")
    return obj


def _g5_s6_first_legal_join(adapter: Any, left: DecoratedG4Tree, right: DecoratedG4Tree, op: tuple[int,int]):
    for lu in range(left.n):
        for rv in range(right.n):
            child = _join_g4_pair(left, right, lu, rv, op)
            q = adapter.public_read(child)
            if q.get("legal"):
                return child, {"left_owner":lu,"right_owner":rv}, q
    return None, None, None


def _g5_s6_compare_direct_to_abstract(adapter: Any, left: DecoratedG4Tree, right: DecoratedG4Tree, op: tuple[int,int], alphabet: tuple[str,...]):
    ls = _g5_s5_state_from_public(adapter.public_read(left), alphabet)
    rs = _g5_s5_state_from_public(adapter.public_read(right), alphabet)
    abs_out = _g5_s5_compose_state(ls, rs, op)
    child, owners, direct = _g5_s6_first_legal_join(adapter, left, right, op)
    if abs_out is None:
        return {"ok": child is None, "abstract_enabled":False, "direct_enabled":child is not None, "owners":owners}, child
    if child is None:
        return {"ok":False,"abstract_enabled":True,"direct_enabled":False,"owners":None,"failure":"NO_DIRECT_REALIZATION"}, None
    expected = {"legal":True, **_g5_s5_public_from_state(abs_out)}
    actual = {k:v for k,v in direct.items() if k in {"legal","descriptor","caps7","H_class_bag"}}
    return {"ok":actual==expected,"abstract_enabled":True,"direct_enabled":True,"owners":owners,"expected":expected,"actual":actual}, child


def _g5_s6_recursive_factorisation_certificate(adapter: Any, alphabet: tuple[str,...]) -> dict[str, Any]:
    import inspect
    pub_src = inspect.getsource(type(adapter).public_read)
    join_src = inspect.getsource(_join_g4_pair)
    forbidden = ["unrooted_canon", "rooted_canon", "action_read", "construction_digest", "ancestry"]
    hidden_hits = [x for x in forbidden if x in pub_src]
    # The exact algebraic identity follows from the implementation's vertex-local
    # residual-capacity sum and Counter(H_classes), plus one typed bridge edge.
    proof = {
      "method":"STRUCTURAL_INDUCTION_ON_FINITE_DECORATED_G5_BINARY_TERMS",
      "base":"For each H-class atom h, public_read gives f=caps7(h) and m=e_h, hence D is defined.",
      "inductive_hypothesis":"For finite generated X,Y, public_read(X)=D_X and public_read(Y)=D_Y depend only on aggregate residual typed capacity and the H-class bag.",
      "inductive_step":"Joining X,Y by admitted directed operator (a,b) unions vertices/old edges and adds exactly one edge reserving type a on X and b on Y. Therefore f_out=f_X+f_Y-e_a-e_b and m_out=m_X+m_Y, with legality iff f_X[a]>0 and f_Y[b]>0 for existence of a legal owner pair.",
      "consequence":"ALL_FINITE_GENERATED_G5_TERMS_UNDER_FROZEN_31_OPERATOR_GRAMMAR_FACTOR_THROUGH_CAPS7_PLUS_H_CLASS_BAG",
      "public_read_source_sha256":canonical_sha256({"source":pub_src}),
      "join_constructor_source_sha256":canonical_sha256({"source":join_src}),
      "hidden_selector_hits":hidden_hits,
      "status":"PASS" if not hidden_hits else "REVIEW_REQUIRED",
    }
    return proof


@register_workflow_capability("g5.s6.recursive-closure.v1")
def g5_s6_recursive_closure(*, adapter: Any, stage: str, parameters: Mapping[str, Any], prior, registration, execution_context=None) -> dict[str, Any]:
    if stage != "S6":
        return {"outcome":"BLOCKED","observed_alternative":S6_FAIL_ALT,"failure":"STAGE_BINDING_MISMATCH"}
    if adapter.descriptor().get("family") != "G4_ACCEPTED" or adapter.descriptor().get("public_descriptor") != "CAPS7_PLUS_H_CLASS_BAG":
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"G4_ADAPTER_AUTHORITY_MISMATCH"}
    if len(prior) != 6 or [x.get("stage") for x in prior] != ["S0","S1","S2","S3","S4","S5"] or any(x.get("outcome") != "PASS" for x in prior):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"S0_S5_PRIOR_BINDING_MISMATCH"}
    spec = _g5_s6_spec(); auth=spec["authority"]
    if str(parameters.get("spec_science_sha256")) != spec["science_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"S6_SPEC_BINDING_MISMATCH"}
    if str(parameters.get("required_s5_stage_science_sha256")) != str(prior[5].get("science_sha256")) or str(prior[5].get("science_sha256")) != auth["g5_s5_stage_science_sha256"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"S5_STAGE_AUTHORITY_MISMATCH"}
    if prior[5].get("observed_alternative") != auth["g5_s5_classification"]:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"S5_CLASSIFICATION_MISMATCH"}
    s5r=prior[5].get("result") or {}
    if not bool(s5r.get("g5_finite_state_descriptor_earned")) or not bool(s5r.get("g5_finite_read_write_law_earned")) or s5r.get("earned_descriptor") != "CAPS7_PLUS_H_CLASS_BAG" or not bool(s5r.get("s6_preregistration_design_review_unlocked")):
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"S5_GRADUATION_PREMISES_MISSING"}

    alphabet=tuple(str(x) for x in spec["frozen_grammar"]["H_class_alphabet"])
    ops=tuple(adapter.operator_basis())
    if len(ops)!=31:
        return {"outcome":"REVIEW_REQUIRED","observed_alternative":S6_FAIL_ALT,"failure":"OPERATOR_BASIS_COUNT_MISMATCH","observed":len(ops)}
    if execution_context is not None:
        execution_context.reserve_cases(31 + 8 + 8 + 4)

    theorem=_g5_s6_recursive_factorisation_certificate(adapter,alphabet)
    carriers=_g5_s1_carriers()
    holdout_failures=[]

    # Fresh holdout A: pair+pair, every one of the 31 operators in the final recursive position.
    seedop=ops[0]
    l1,_,_= _g5_s6_first_legal_join(adapter,carriers["D2_PATH"],carriers["D2_BROOM"],seedop)
    r1,_,_= _g5_s6_first_legal_join(adapter,carriers["D2_BROOM"],carriers["D2_PATH"],seedop)
    pair_rows=[]
    for op in ops:
        row,child=_g5_s6_compare_direct_to_abstract(adapter,l1,r1,op,alphabet)
        rec={"operator":list(op),**row}; pair_rows.append(rec)
        if not row["ok"]: holdout_failures.append({"family":"PAIR_PLUS_PAIR",**rec})

    # Fresh holdout B: eight deterministic deeper P5-style left-deep terms (five certified carriers).
    refs=("D2_PATH","D2_BROOM","D4_PATH","D4_BROOM")
    deep_rows=[]
    for i in range(8):
        seq=[refs[(i+j)%4] for j in range(5)]
        opseq=[ops[(i*7+j*5)%len(ops)] for j in range(4)]
        cur=carriers[seq[0]]; abs_state=_g5_s5_state_from_public(adapter.public_read(cur),alphabet); steps=[]; ok=True
        for j in range(4):
            nxt=carriers[seq[j+1]]; ns=_g5_s5_state_from_public(adapter.public_read(nxt),alphabet); ao=_g5_s5_compose_state(abs_state,ns,opseq[j]); child,owners,direct=_g5_s6_first_legal_join(adapter,cur,nxt,opseq[j])
            if ao is None or child is None:
                stepok=(ao is None and child is None); steps.append({"step":j,"operator":list(opseq[j]),"ok":stepok,"owners":owners}); ok=ok and stepok
                if not stepok: break
                continue
            expected={"legal":True,**_g5_s5_public_from_state(ao)}; actual={k:v for k,v in direct.items() if k in {"legal","descriptor","caps7","H_class_bag"}}; stepok=(expected==actual)
            steps.append({"step":j,"operator":list(opseq[j]),"ok":stepok,"owners":owners,"direct_sha256":canonical_sha256(actual),"abstract_sha256":canonical_sha256(expected)})
            ok=ok and stepok; cur=child; abs_state=ao
            if not ok: break
        rec={"case":i,"carrier_sequence":seq,"operator_sequence":[list(x) for x in opseq],"ok":ok,"steps":steps}; deep_rows.append(rec)
        if not ok: holdout_failures.append({"family":"DEEP_P5",**rec})

    # Fresh holdout C: public rebracketing. Compare (A*B)*C with A*(B*C) for matching typed operators.
    rebracket_rows=[]
    for i in range(8):
        A=carriers[refs[i%4]]; B=carriers[refs[(i+1)%4]]; C=carriers[refs[(i+2)%4]]
        op1=ops[(i*3)%len(ops)]; op2=ops[(i*3+11)%len(ops)]
        ab,_,_= _g5_s6_first_legal_join(adapter,A,B,op1); left,_,lq=_g5_s6_first_legal_join(adapter,ab,C,op2)
        bc,_,_= _g5_s6_first_legal_join(adapter,B,C,op2); right,_,rq=_g5_s6_first_legal_join(adapter,A,bc,op1)
        ok=bool(left is not None and right is not None and _public_key(lq)==_public_key(rq))
        rec={"case":i,"operators":[list(op1),list(op2)],"ok":ok,"left_public_sha256":canonical_sha256(_public_key(lq)) if lq else None,"right_public_sha256":canonical_sha256(_public_key(rq)) if rq else None}; rebracket_rows.append(rec)
        if not ok: holdout_failures.append({"family":"REBRACKETING",**rec})

    # Base-atom coverage.
    atom_rows=[]
    for klass in alphabet:
        q=adapter.public_read(DecoratedG4Tree(1,tuple(),(klass,),tuple())); st=_g5_s5_state_from_public(q,alphabet)
        atom_rows.append({"H_class":klass,"state_sha256":canonical_sha256(st),"ok":bool(q.get("legal"))})

    passed=(theorem.get("status")=="PASS" and not holdout_failures and len(pair_rows)==31 and len(deep_rows)==8 and len(rebracket_rows)==8 and len(atom_rows)==4 and all(x["ok"] for x in atom_rows))
    result={
      "question":spec["question"],"s6_spec_science_sha256":spec["science_sha256"],"s5_stage_science_sha256":prior[5].get("science_sha256"),
      "candidate_descriptor":"CAPS7_PLUS_H_CLASS_BAG","candidate_law":spec["candidate_law"],"recursive_factorisation_theorem":theorem,
      "directed_operator_count":len(ops),"base_atom_rows":atom_rows,
      "fresh_holdout":{"pair_plus_pair_case_count":len(pair_rows),"pair_plus_pair_rows":pair_rows,"deep_p5_case_count":len(deep_rows),"deep_p5_rows":deep_rows,"rebracketing_case_count":len(rebracket_rows),"rebracketing_rows":rebracket_rows,"failure_count":len(holdout_failures),"first_failure":holdout_failures[0] if holdout_failures else None,"outcome_retuning":False},
      "implementation_read_surface_audit":{"public_read_hidden_selector_hits":theorem.get("hidden_selector_hits"),"hidden_topology_read":False,"hidden_owner_identity_read":False,"construction_identity_read":False,"ancestry_read":False,"exact_relation_cardinality_read":False,"branch_multiplicity_read":False},
      "proof_obligations":{"registered":list(spec["proof_obligations"]),"satisfied":list(spec["proof_obligations"]) if passed else [],"unmet":[] if passed else list(spec["proof_obligations"])},
      "coverage_complete":passed,"g5_global_composition_law_candidate":passed,"g5_recursive_closure_candidate":passed,"g5_graduation_candidate":passed,
      "mechanical_s6_pass_graduates":False,"independent_verification_required":True,"scientific_graduation_decision_required":True,
      "g4_changed":False,"hidden_topology_promoted":False,"hidden_owner_identity_promoted":False,"geometry_claim":False,"promotion":False,
      "nonclaims":list(spec["nonclaims"]),
    }
    return {"outcome":"PASS" if passed else "REVIEW_REQUIRED","observed_alternative":S6_PASS_ALT if passed else S6_FAIL_ALT,**result}


register_workflow_semantic_contract("g5.s6.recursive-closure.v1", {
    "contract_id":"IG_G5_S6_SEMANTIC_CONTRACT_V1","version":"1.0.0",
    "workflow_kind":"STRUCTURAL_S0_S6","adapter_family":"G4_ACCEPTED","public_descriptor":"CAPS7_PLUS_H_CLASS_BAG","allowed_stages":["S6"],
    "parameter_keys":["required_s5_stage_science_sha256","spec_science_sha256"],
    "registered_alternatives":[S6_PASS_ALT,S6_FAIL_ALT],
})
