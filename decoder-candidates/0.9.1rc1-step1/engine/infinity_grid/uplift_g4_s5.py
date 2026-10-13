from __future__ import annotations

"""Registered G4:S5 finite read/write descriptor audit.

S4 certified one-step relation-valued composition closure on the frozen P3 domain.
S5 asks whether the surviving S2 Tier-1 scalar-tree read can now be promoted into
an actual finite-coordinate G4 transition state without reopening hidden G3 topology.

The preregistered candidate is

    (CAPS7, BAG(TIER1_CLASS))

where the bag records only multiplicities of the already-earned S2 Tier-1 public
classes of constituent whole G3 units.  It deliberately does *not* observe exact
relation identity/cardinality, branch multiplicity, owner witnesses, construction
identity, shell profile, or hidden topology.

Reserve and binary writes are algebraic on this state:
  reserve_t(f,m) = (f-e_t,m)
  compose_{a,b}((f,m),(g,n)) = (f+g-e_a-e_b,m+n)

S6 owns recursive closure/graduation and all fresh holdout stress.
"""

from collections import Counter
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import gzip
import inspect
import json

from .canon import canonical_sha256
from .uplift_g4_s0 import phase0_spec


class G4S5Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s5_spec() -> dict[str, Any]:
    obj = json.loads(_resource("G4_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G4_S5_FINITE_READ_WRITE_DESCRIPTOR_SPEC_V1":
        raise G4S5Error("bad G4:S5 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G4S5Error("G4:S5 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G4S5Error("G4:S5/Phase0 binding mismatch")
    return obj


def _caps7(v: Sequence[Any], field: str = "CAPS7") -> tuple[int, ...]:
    out = tuple(int(x) for x in v)
    if len(out) != 7 or any(x < 0 for x in out):
        raise G4S5Error(f"{field} must be seven nonnegative integers")
    return out


def _tier1_rows(s3_result: Mapping[str, Any]) -> tuple[list[str], dict[str, str], dict[str, dict[str, int]]]:
    idx = s3_result.get("candidate_interface_index", {})
    if idx.get("schema_id") != "IG_G4_S3_CANDIDATE_INTERFACE_INDEX_V1" or idx.get("status") != "PASS":
        raise G4S5Error("missing certified S3 Tier-1 candidate interface index")
    if idx.get("science_sha256") != s5_spec()["authority"]["g4_s3_candidate_interface_index_sha256"]:
        raise G4S5Error("S3 candidate interface index binding mismatch")
    rows = idx.get("rows", [])
    if len(rows) != 2:
        raise G4S5Error("frozen G4:S5 Tier-1 alphabet must contain exactly two classes")
    ref_to_label: dict[str, str] = {}
    values: dict[str, dict[str, int]] = {}
    for row in rows:
        ref = str(row["term_ref"])
        val = {str(k): int(v) for k, v in row["tier1_value"].items()}
        label = canonical_sha256({"schema_id": "IG_G4_TIER1_STATIC_CLASS_V1", "tier1_value": val})
        ref_to_label[ref] = label
        values[label] = val
    alphabet = sorted(values)
    if len(alphabet) != 2 or len(ref_to_label) != 2:
        raise G4S5Error("Tier-1 label collision in frozen alphabet")
    return alphabet, ref_to_label, values


def descriptor_from_public_parts(*, caps7: Sequence[Any], tier1_labels: Sequence[str], alphabet: Sequence[str]) -> dict[str, Any]:
    caps = _caps7(caps7)
    alpha = tuple(sorted(str(x) for x in alphabet))
    if len(alpha) != len(set(alpha)) or not alpha:
        raise G4S5Error("descriptor alphabet must be nonempty and unique")
    counts = Counter(str(x) for x in tier1_labels)
    unknown = sorted(set(counts) - set(alpha))
    if unknown:
        raise G4S5Error("descriptor uses unknown Tier-1 class")
    vec = [int(counts.get(label, 0)) for label in alpha]
    out = {
        "schema_id": "IG_G4_CAPS7_TIER1_BAG_READ_WRITE_STATE_V1",
        "total_free_by_type": list(caps),
        "tier1_class_alphabet": list(alpha),
        "tier1_class_counts": vec,
        "g3_unit_count": int(sum(vec)),
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def _descriptor_parts(desc: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]]:
    if desc.get("schema_id") != "IG_G4_CAPS7_TIER1_BAG_READ_WRITE_STATE_V1":
        raise G4S5Error("bad G4 descriptor schema")
    caps = _caps7(desc.get("total_free_by_type", []))
    alpha = tuple(str(x) for x in desc.get("tier1_class_alphabet", []))
    counts = tuple(int(x) for x in desc.get("tier1_class_counts", []))
    if len(alpha) != len(counts) or any(x < 0 for x in counts) or int(desc.get("g3_unit_count", -1)) != sum(counts):
        raise G4S5Error("malformed G4 Tier-1 bag descriptor")
    if list(alpha) != sorted(alpha) or len(alpha) != len(set(alpha)):
        raise G4S5Error("noncanonical Tier-1 alphabet")
    expected = canonical_sha256({k: v for k, v in desc.items() if k != "science_sha256"})
    if desc.get("science_sha256") != expected:
        raise G4S5Error("descriptor hash mismatch")
    return caps, alpha, counts


def reserve_write(desc: Mapping[str, Any], endpoint_type: int) -> dict[str, Any] | None:
    caps, alpha, counts = _descriptor_parts(desc)
    t = int(endpoint_type)
    if not 0 <= t < 7:
        raise G4S5Error("endpoint type outside alphabet")
    if caps[t] <= 0:
        return None
    nxt = list(caps); nxt[t] -= 1
    labels: list[str] = []
    for label, n in zip(alpha, counts): labels.extend([label] * n)
    return descriptor_from_public_parts(caps7=nxt, tier1_labels=labels, alphabet=alpha)


def binary_write(left: Mapping[str, Any], right: Mapping[str, Any], a: int, b: int, operator_basis: Sequence[Sequence[int]]) -> dict[str, Any] | None:
    lf, la, lm = _descriptor_parts(left); rf, ra, rm = _descriptor_parts(right)
    if la != ra:
        raise G4S5Error("descriptor alphabets differ")
    aa, bb = int(a), int(b)
    basis = {(int(x), int(y)) for x, y in operator_basis}
    if (aa, bb) not in basis or lf[aa] <= 0 or rf[bb] <= 0:
        return None
    out = [lf[i] + rf[i] for i in range(7)]
    out[aa] -= 1; out[bb] -= 1
    labels: list[str] = []
    for label, n in zip(la, tuple(lm[i] + rm[i] for i in range(len(lm)))):
        labels.extend([label] * n)
    return descriptor_from_public_parts(caps7=out, tier1_labels=labels, alphabet=la)


def verify_s5_authority(*, s3_result: Mapping[str, Any], s4_result: Mapping[str, Any], s4_replay: Mapping[str, Any], s4_closeout: Mapping[str, Any]) -> dict[str, Any]:
    spec = s5_spec(); a = spec["authority"]; failures: list[str] = []
    if s4_result.get("schema_id") != "IG_G4_S4_COMPOSITION_CLOSURE_RESULT_V1": failures.append("S4_SCHEMA")
    if s4_result.get("status") != "PASS": failures.append("S4_STATUS")
    if s4_result.get("classification") != a["g4_s4_classification"]: failures.append("S4_CLASSIFICATION")
    if s4_result.get("science_sha256") != a["g4_s4_science_sha256"]: failures.append("S4_SCIENCE")
    if s4_result.get("g4_s5_unlocked") is not True: failures.append("S5_NOT_UNLOCKED")
    if s4_result.get("g4_graduated") is not False: failures.append("G4_GRADUATION_FIREWALL")
    if s4_result.get("public_descriptor_promoted") is not False: failures.append("PREMATURE_PUBLIC_PROMOTION")
    if s4_result.get("topology_promoted") is not False or s4_result.get("shell_profile_promoted") is not False: failures.append("STRUCTURE_PROMOTION_FIREWALL")
    if s4_replay.get("schema_id") != "IG_G4_S4_COLD_REPLAY_COMPARISON_V1": failures.append("S4_REPLAY_SCHEMA")
    if s4_replay.get("certification") != "CERTIFIED_PASS" or s4_replay.get("status") != "PASS": failures.append("S4_REPLAY_STATUS")
    if s4_replay.get("science_sha_equal") is not True or s4_replay.get("source_sha_equal") is not True or s4_replay.get("registry_sha_equal") is not True: failures.append("S4_REPLAY_IDENTITY")
    if s4_replay.get("comparison_sha256") != a["g4_s4_replay_comparison_sha256"]: failures.append("S4_REPLAY_HASH")
    if s4_closeout.get("schema_id") != "IG_G4_S4_CERTIFIED_CLOSEOUT_V1" or s4_closeout.get("status") != "CERTIFIED_PASS": failures.append("S4_CLOSEOUT_STATUS")
    if s4_closeout.get("closeout_sha256") != a["g4_s4_closeout_sha256"]: failures.append("S4_CLOSEOUT_HASH")
    try: _tier1_rows(s3_result)
    except G4S5Error as exc: failures.append("S3_TIER1_INDEX:" + str(exc))
    out = {"schema_id":"IG_G4_S5_AUTHORITY_V1","status":"PASS" if not failures else "FAIL","failures":failures,
           "g4_s4_science_sha256":s4_result.get("science_sha256"),"g4_s4_certified":not failures,"g4_graduated":False,
           "public_descriptor_promoted":False,"topology_promoted":False,"shell_profile_promoted":False}
    out["science_sha256"] = canonical_sha256(out)
    if failures: raise G4S5Error("G4:S5 authority failed: " + ",".join(failures))
    return out


def _class_summary(hashes: Sequence[str]) -> dict[str, Any]:
    c=Counter(hashes); sh=Counter(c.values())
    return {"record_count":len(hashes),"descriptor_class_count":len(c),"multi_record_descriptor_class_count":sum(1 for n in c.values() if n>1),
            "records_in_multi_record_descriptor_classes":sum(n for n in c.values() if n>1),"largest_descriptor_class_size":max(c.values()) if c else 0,
            "descriptor_class_size_histogram":[{"record_count":int(k),"descriptor_class_count":int(v)} for k,v in sorted(sh.items())]}


def certify_candidate_census(*, s3_result: Mapping[str, Any], pair_basis: Mapping[str, Any], recursive_rows_path: str|Path) -> dict[str, Any]:
    alphabet, ref_to_label, values = _tier1_rows(s3_result)
    if pair_basis.get("schema_id") != "IG_G4_S4_COMPLETE_PAIR_CANDIDATE_BASIS_V1" or pair_basis.get("status") != "PASS":
        raise G4S5Error("bad S4 pair basis")
    pair_rows=pair_basis.get("rows",[])
    if len(pair_rows)!=124: raise G4S5Error("S4 pair basis must contain 124 rows")
    pair_desc: dict[int,dict[str,Any]]={}; pair_hashes=[]; failures=[]
    for row in pair_rows:
        idx=int(row["pair_key_index"]); capsset=row.get("public_output_caps7_set",[])
        if len(capsset)!=1: failures.append({"scope":"PAIR","index":idx,"reason":"NON_SINGLETON_CAPS7"}); continue
        labels=[ref_to_label[str(row["left_ref"])],ref_to_label[str(row["right_ref"])]]
        d=descriptor_from_public_parts(caps7=capsset[0],tier1_labels=labels,alphabet=alphabet); pair_desc[idx]=d; pair_hashes.append(d["science_sha256"])
    with gzip.open(Path(recursive_rows_path),"rt",encoding="utf-8") as fh: obj=json.load(fh)
    if obj.get("schema_id")!="IG_G4_S4_RECURSIVE_P3_INDEX_V1" or int(obj.get("task_count",-1))!=7688:
        raise G4S5Error("bad S4 recursive P3 index")
    rec_hashes=[]; collisions: dict[tuple[str,str,str],set[str]]={}
    for row in obj.get("rows",[]):
        sig=row.get("signature",{}); sh=sig.get("science_sha256")
        if canonical_sha256({k:v for k,v in sig.items() if k!="science_sha256"})!=sh: raise G4S5Error("S4 recursive signature hash mismatch")
        capsset=sig.get("public_output_caps7_set",[]); i=int(row["pair_key_index"])
        if len(capsset)!=1 or i not in pair_desc: failures.append({"scope":"RECURSIVE","pair_key_index":i,"reason":"MISSING_PAIR_OR_NON_SINGLETON_CAPS7"}); continue
        prow=pair_rows[i]
        labels=[ref_to_label[str(prow["left_ref"])],ref_to_label[str(prow["right_ref"])],ref_to_label[str(row["third_ref"])]]
        d=descriptor_from_public_parts(caps7=capsset[0],tier1_labels=labels,alphabet=alphabet); rec_hashes.append(d["science_sha256"])
        # Reduced observer intentionally ignores exact relation cardinality/branch multiplicity.
        k=(pair_desc[i]["science_sha256"],ref_to_label[str(row["third_ref"])],f"{int(row['operator'][0])}>{int(row['operator'][1])}")
        collisions.setdefault(k,set()).add(d["science_sha256"])
    conflicts=[{"abstract_input":list(k),"output_descriptor_hashes":sorted(v)} for k,v in collisions.items() if len(v)>1]
    passed=not failures and not conflicts and len(pair_hashes)==124 and len(rec_hashes)==7688
    out={"schema_id":"IG_G4_S5_CANDIDATE_DESCRIPTOR_CENSUS_V1","status":"PASS" if passed else "FAIL",
         "candidate":"CAPS7_PLUS_TIER1_CLASS_BAG","tier1_class_alphabet":[{"class_sha256":x,"tier1_value":values[x]} for x in alphabet],
         "pair_candidate_summary":_class_summary(pair_hashes),"recursive_p3_candidate_summary":_class_summary(rec_hashes),
         "abstract_recursive_write_conflict_count":len(conflicts),"abstract_recursive_write_conflict_examples":conflicts[:16],
         "failure_count":len(failures),"failure_examples":failures[:16],"exact_relation_cardinality_observed":False,"branch_multiplicity_observed":False}
    out["science_sha256"]=canonical_sha256(out); return out


def verify_reserve_factorisation(*, reserve_basis: Mapping[str, Any]) -> dict[str, Any]:
    if reserve_basis.get("schema_id")!="IG_G4_S4_RESERVE_BASIS_INDEX_V1": raise G4S5Error("bad S4 reserve basis schema")
    rows=reserve_basis.get("rows",[]); failures=[]; checks=0
    if len(rows)!=31: failures.append("BASIS_ROW_COUNT")
    for r in rows:
        rc=r.get("reserve_capability",{}); checks+=int(rc.get("reserve_checks",0))
        if rc.get("status")!="PASS" or int(rc.get("failure_count",-1))!=0: failures.append("RESERVE_CAPABILITY")
    out={"schema_id":"IG_G4_S5_RESERVE_FACTORISATION_V1","status":"PASS" if not failures else "FAIL","failures":failures,
         "operator_basis_rows":len(rows),"exact_branch_reservation_checks_in_certified_s4_evidence":checks,
         "law":"For state (f,m), reserve_t is enabled iff f_t>0 and every exact owner branch maps to (f-e_t,m); Tier-1 class bag m is invariant under external reservation.",
         "exact_relation_cardinality_observed":False,"branch_multiplicity_observed":False,"hidden_owner_observed":False}
    out["science_sha256"]=canonical_sha256(out); return out


def implementation_read_surface_audit() -> dict[str, Any]:
    dsrc=inspect.getsource(descriptor_from_public_parts); rsrc=inspect.getsource(reserve_write); bsrc=inspect.getsource(binary_write)
    failures=[]
    for token in ("topology","node_caps","typed_edges","construction_digest","owner","ancestry","shell"):
        if token in dsrc or token in rsrc or token in bsrc: failures.append("FORBIDDEN_ABSTRACT_READ:"+token)
    out={"schema_id":"IG_G4_S5_IMPLEMENTATION_READ_SURFACE_AUDIT_V1","status":"PASS" if not failures else "FAIL","failures":failures,
         "abstract_state_fields":["CAPS7","Tier1_class_bag"],"construction_identity_role":"EXACT_RELATION_DEDUP_ONLY_NOT_ABSTRACT_SELECTOR",
         "direct_hidden_topology_read":False,"direct_shell_profile_read":False,"owner_witness_read":False,"lower_layer_carrier_read":False}
    out["science_sha256"]=canonical_sha256(out); return out


def factorisation_argument(*, census: Mapping[str,Any], reserve: Mapping[str,Any], implementation: Mapping[str,Any]) -> dict[str,Any]:
    passed=all(x.get("status")=="PASS" for x in (census,reserve,implementation))
    out={"schema_id":"IG_G4_S5_FACTORISATION_ARGUMENT_V1","status":"PASS" if passed else "FAIL",
         "state":"D=(f,m) with f in N^7 and m the finite bag of already-earned S2 Tier-1 scalar-tree classes of constituent whole G3 units.",
         "reserve_law":"reserve_t(f,m)=(f-e_t,m) whenever f_t>0.",
         "binary_law":"compose_{a,b}((f,m),(g,n))=(f+g-e_a-e_b,m+n) for a frozen earned bridge operator with available endpoints.",
         "why_no_hidden_read":"The static class bag is inherited from certified S2 public labels; reserve never changes those labels, and composition only adds their multiplicities. No topology canon, shell profile, owner witness, construction identity, ancestry or lower-layer carrier is read by the abstract law.",
         "observer_note":"Exact relation cardinality and branch multiplicity were useful in S1-S4 as adversarial/closure observables, but are deliberately outside the S5 transition-state observer, exactly as observer-relative abstraction permits. S6 must test whether this coarser promoted state survives fresh recursive contexts.",
         "scope":"FROZEN_ONE_STEP_RELATION_VALUED_G4_GRAMMAR_CERTIFIED_BY_S4; recursive closure, pair+pair, rebracketing and graduation remain S6.",
         "not_claimed":["minimality","raw exact relation identity","exact branch multiplicity identity","global Tier-1 sufficiency","pair+pair closure","unbounded recursive closure","topology erasure","G4 graduation"]}
    out["science_sha256"]=canonical_sha256(out); return out


def finalize_s5_result(*, authority:Mapping[str,Any], census:Mapping[str,Any], reserve:Mapping[str,Any], implementation:Mapping[str,Any], argument:Mapping[str,Any])->dict[str,Any]:
    spec=s5_spec(); passed=all(x.get("status")=="PASS" for x in (authority,census,reserve,implementation,argument))
    out={"schema_id":"IG_G4_S5_FINITE_READ_WRITE_DESCRIPTOR_RESULT_V1","schema_version":"1.0.0","date":"2026-09-04","stage_ref":"G4:S5",
         "status":"PASS" if passed else "REVIEW_REQUIRED",
         "classification":"G4_CAPS7_PLUS_TIER1_BAG_FINITE_READ_WRITE_DESCRIPTOR_EARNED_ON_FROZEN_GRAMMAR_S6_UNLOCKED" if passed else "G4_FINITE_DESCRIPTOR_GATE_FAILED_S6_LOCKED",
         "candidate_descriptor":{"schema_id":"IG_G4_CAPS7_TIER1_BAG_READ_WRITE_STATE_V1","name":"CAPS7_PLUS_TIER1_CLASS_BAG","coordinate_domain":"N",
             "frozen_coordinate_count":9,"fields":["total_free_by_type[0..6]","tier1_class_counts[0..1]"],
             "tier1_class_source":"CERTIFIED_G4_S2_TIER1_SCALAR_TREE_SUMMARIES","relation_semantics":"complete exact relations are quotiented only by common abstract state under the reduced S5 transition observer"},
         "observer_scope":spec["frozen_observer"],"authority":dict(authority),"candidate_descriptor_census":dict(census),"reserve_factorisation":dict(reserve),
         "implementation_read_surface_audit":dict(implementation),"factorisation_argument":dict(argument),
         "public_descriptor_promoted":passed,"promoted_descriptor":"CAPS7_PLUS_TIER1_CLASS_BAG" if passed else None,
         "topology_promoted":False,"shell_profile_promoted":False,"g4_s6_unlocked":passed,"g4_graduated":False,"next_authorized_stage":"G4:S6" if passed else None,
         "nonclaims":spec["nonclaims"],"g4_s5_spec_sha256":spec["science_sha256"]}
    out["science_sha256"]=canonical_sha256(out); return out


def stable_science_core(result:Mapping[str,Any])->dict[str,Any]:
    volatile={"source_sha256","source_version","registry_sha256","execution_metadata","native_engine_science_sha256"}
    return {k:v for k,v in result.items() if k not in volatile and k!="science_sha256"}


def compare_cold_replay(primary:Mapping[str,Any],cold:Mapping[str,Any])->dict[str,Any]:
    checks={"science_sha_equal":primary.get("science_sha256")==cold.get("science_sha256"),"source_sha_equal":primary.get("source_sha256")==cold.get("source_sha256"),"registry_sha_equal":primary.get("registry_sha256")==cold.get("registry_sha256"),"classification_equal":primary.get("classification")==cold.get("classification"),"descriptor_equal":primary.get("candidate_descriptor")==cold.get("candidate_descriptor"),"stable_science_core_equal":stable_science_core(primary)==stable_science_core(cold)}
    failures=[k.upper() for k,v in checks.items() if not v]
    out={"schema_id":"IG_G4_S5_COLD_REPLAY_COMPARISON_V1","status":"PASS" if not failures else "FAIL","certification":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED","failures":failures,"experiment_id":"G4:S5","same_registered_decoder_native_experiment":True,**checks,
         "primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"primary_source_sha256":primary.get("source_sha256"),"cold_source_sha256":cold.get("source_sha256"),"primary_registry_sha256":primary.get("registry_sha256"),"cold_registry_sha256":cold.get("registry_sha256")}
    out["comparison_sha256"]=canonical_sha256(out); return out


def certified_closeout(primary:Mapping[str,Any],cold:Mapping[str,Any],replay:Mapping[str,Any])->dict[str,Any]:
    failures=[]
    if replay.get("certification")!="CERTIFIED_PASS": failures.append("REPLAY")
    if primary.get("status")!="PASS" or cold.get("status")!="PASS": failures.append("SCIENCE_STATUS")
    out={"schema_id":"IG_G4_S5_CERTIFIED_CLOSEOUT_V1","status":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED","failures":failures,"experiment_id":"G4:S5",
         "classification":primary.get("classification") if not failures else None,"science_sha256":primary.get("science_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),"comparison_sha256":replay.get("comparison_sha256"),
         "public_descriptor_promoted":bool(not failures and primary.get("public_descriptor_promoted")),"promoted_descriptor":primary.get("promoted_descriptor") if not failures else None,
         "topology_promoted":False,"shell_profile_promoted":False,"g4_s6_unlocked":bool(not failures and primary.get("g4_s6_unlocked")),"g4_graduated":False,"next_authorized_stage":"G4:S6" if not failures and primary.get("g4_s6_unlocked") else None}
    out["closeout_sha256"]=canonical_sha256(out); return out
