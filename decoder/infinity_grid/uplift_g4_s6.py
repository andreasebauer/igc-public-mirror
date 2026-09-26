from __future__ import annotations

"""G4:S6 recursive closure and graduation audit.

This is the only frozen G4 Phase-0 stage allowed to graduate G4.  It consumes the
certified S5 public descriptor D=(CAPS7, Tier-1-class bag), proves recursive
complete-relation factorisation by structural induction, attacks that proof with
fresh exact P4/P5/rebracketing holdouts, and still requires a cold rerun of this
same registered Decoder-native experiment before a graduation certificate exists.

Hidden G3 topology, shell profile, owner identity, exact relation cardinality and
branch multiplicity remain outside the promoted observer. Exact construction
identity is permitted only for operational deduplication of the challenge engine.
"""

from dataclasses import dataclass
from functools import cached_property
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import hashlib
import inspect
import json

from .canon import canonical_sha256, write_json_atomic
from .g4_term_state import G3TermState, G4PairTermState, compose_g4_pair_relation
from .uplift_g4_s0 import phase0_spec
from .uplift_g4_s1 import load_s0_term_states
from .uplift_g4_s5 import descriptor_from_public_parts, reserve_write, binary_write


class G4S6Error(RuntimeError):
    pass


_SPEC = "G4_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1.json"
_WORKER_CONTEXT: dict[str, Any] | None = None


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s6_spec() -> dict[str, Any]:
    obj = json.loads(_resource(_SPEC).read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G4_S6_RECURSIVE_CLOSURE_GRADUATION_SPEC_V1":
        raise G4S6Error("bad G4:S6 spec schema")
    if canonical_sha256({k: v for k, v in obj.items() if k != "science_sha256"}) != obj.get("science_sha256"):
        raise G4S6Error("G4:S6 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G4S6Error("G4:S6/Phase0 binding mismatch")
    return obj


def _caps7(st: Any) -> tuple[int, ...]:
    out = tuple(int(x) for x in st.total_caps)
    if len(out) != 7 or any(x < 0 for x in out):
        raise G4S6Error("malformed exact G4 CAPS7")
    return out


def _digest(st: Any) -> str:
    d = str(st.construction_digest)
    if len(d) != 64:
        raise G4S6Error("exact G4 state lacks canonical construction digest")
    return d


def _reserve_state(st: Any, endpoint_type: int) -> tuple[Any, ...]:
    fn = getattr(st, "reserve_external_relation", None)
    if fn is None:
        raise G4S6Error("exact G4 state lacks relation-valued reserve")
    return tuple(fn(int(endpoint_type)))


def _dedupe(states: Sequence[Any]) -> tuple[Any, ...]:
    out: dict[str, Any] = {}
    for st in states:
        out.setdefault(_digest(st), st)
    return tuple(out[k] for k in sorted(out))


@dataclass(frozen=True)
class G4RecursiveTermState:
    """Generic finite G4 exact construction tree used only by S6 holdout transfer.

    Children are already-reserved exact branches. The constructor itself never
    selects an owner. Relation-level composition below enumerates every eligible
    left/right reserve branch before constructing and exact-deduplicating outputs.
    """

    left: Any
    right: Any
    operator: tuple[int, int]

    def __post_init__(self) -> None:
        a, b = map(int, self.operator)
        if not (0 <= a < 7 and 0 <= b < 7):
            raise G4S6Error("bad recursive G4 operator")
        object.__setattr__(self, "operator", (a, b))

    @cached_property
    def total_caps(self) -> tuple[int, ...]:
        lf, rf = _caps7(self.left), _caps7(self.right)
        return tuple(lf[t] + rf[t] for t in range(7))

    @cached_property
    def construction_digest(self) -> str:
        return canonical_sha256({
            "schema_id": "IG_G4_S6_RECURSIVE_TERM_CANON_V1",
            "left": _digest(self.left),
            "right": _digest(self.right),
            "operator": list(self.operator),
        })

    def reserve_external_relation(self, endpoint_type: int) -> tuple["G4RecursiveTermState", ...]:
        t = int(endpoint_type)
        if not 0 <= t < 7:
            raise G4S6Error("endpoint type outside seven-type alphabet")
        out: dict[str, G4RecursiveTermState] = {}
        for ls in _reserve_state(self.left, t):
            st = G4RecursiveTermState(ls, self.right, self.operator)
            out.setdefault(st.construction_digest, st)
        for rs in _reserve_state(self.right, t):
            st = G4RecursiveTermState(self.left, rs, self.operator)
            out.setdefault(st.construction_digest, st)
        return tuple(out[k] for k in sorted(out))


def iter_complete_compose(left_rel: Sequence[Any], right_rel: Sequence[Any], op: Sequence[int]):
    """Yield every raw exact branch of complete relation-valued G4 composition."""
    a, b = map(int, op)
    if not (0 <= a < 7 and 0 <= b < 7):
        raise G4S6Error("operator outside seven-type alphabet")
    for left in left_rel:
        if _caps7(left)[a] <= 0:
            continue
        lres = _reserve_state(left, a)
        for right in right_rel:
            if _caps7(right)[b] <= 0:
                continue
            rres = _reserve_state(right, b)
            for ls in lres:
                for rs in rres:
                    yield G4RecursiveTermState(ls, rs, (a, b))


def compose_complete_relations(left_rel: Sequence[Any], right_rel: Sequence[Any], op: Sequence[int]) -> tuple[Any, ...]:
    return _dedupe(tuple(iter_complete_compose(left_rel, right_rel, op)))


def relation_caps7(rel: Sequence[Any]) -> tuple[int, ...]:
    if not rel:
        raise G4S6Error("empty exact G4 relation")
    vals = {_caps7(st) for st in rel}
    if len(vals) != 1:
        raise G4S6Error("complete exact G4 relation is not singleton CAPS7")
    return next(iter(vals))


def tier1_index(s3_result: Mapping[str, Any]) -> tuple[list[str], dict[str, str], dict[str, dict[str, int]]]:
    idx = s3_result.get("candidate_interface_index", {})
    if idx.get("schema_id") != "IG_G4_S3_CANDIDATE_INTERFACE_INDEX_V1" or idx.get("status") != "PASS":
        raise G4S6Error("missing certified S3 Tier-1 index")
    if idx.get("science_sha256") != s6_spec()["authority"]["g4_s3_candidate_interface_index_sha256"]:
        raise G4S6Error("S3 Tier-1 index hash mismatch")
    ref_to_label: dict[str, str] = {}
    vals: dict[str, dict[str, int]] = {}
    for row in idx.get("rows", []):
        val = {str(k): int(v) for k, v in row["tier1_value"].items()}
        label = canonical_sha256({"schema_id": "IG_G4_TIER1_STATIC_CLASS_V1", "tier1_value": val})
        ref_to_label[str(row["term_ref"])] = label
        vals[label] = val
    alpha = sorted(vals)
    if len(alpha) != 2 or len(ref_to_label) != 2:
        raise G4S6Error("frozen S6 Tier-1 alphabet must contain two classes")
    return alpha, ref_to_label, vals


def leaf_descriptor(ref: str, states: Mapping[str, G3TermState], ref_to_label: Mapping[str, str], alphabet: Sequence[str]) -> dict[str, Any]:
    if ref not in states or ref not in ref_to_label:
        raise G4S6Error("unknown certified G4 leaf")
    return descriptor_from_public_parts(caps7=states[ref].total_caps, tier1_labels=[ref_to_label[ref]], alphabet=alphabet)


def descriptor_for_relation(rel: Sequence[Any], refs: Sequence[str], ref_to_label: Mapping[str, str], alphabet: Sequence[str]) -> dict[str, Any]:
    return descriptor_from_public_parts(caps7=relation_caps7(rel), tier1_labels=[ref_to_label[str(r)] for r in refs], alphabet=alphabet)


def _desc_public(desc: Mapping[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    caps = tuple(int(x) for x in desc.get("total_free_by_type", []))
    bag = tuple(int(x) for x in desc.get("tier1_class_counts", []))
    if len(caps) != 7 or len(bag) != 2:
        raise G4S6Error("malformed public D descriptor")
    return caps, bag


def verify_authority(*, s0_result: Mapping[str, Any], s3_result: Mapping[str, Any], s4_pair_basis: Mapping[str, Any], s5_result: Mapping[str, Any], s5_replay: Mapping[str, Any], s5_closeout: Mapping[str, Any]) -> dict[str, Any]:
    a = s6_spec()["authority"]
    failures: list[str] = []
    if s0_result.get("schema_id") != "IG_G4_S0_INTERFACE_EXTRACTION_RESULT_V2" or s0_result.get("status") != "PASS": failures.append("S0_AUTHORITY")
    if s0_result.get("certified_term_corpus", {}).get("science_sha256") != a["g4_s0_term_corpus_sha256"]: failures.append("S0_TERM_CORPUS")
    try:
        states, load_meta = load_s0_term_states(s0_result)
        alpha, ref_labels, _vals = tier1_index(s3_result)
    except Exception as exc:
        failures.append("LEAF_OR_TIER1_LOAD:" + str(exc)); states={}; load_meta={}; alpha=[]; ref_labels={}
    if s4_pair_basis.get("schema_id") != "IG_G4_S4_COMPLETE_PAIR_CANDIDATE_BASIS_V1" or s4_pair_basis.get("status") != "PASS": failures.append("S4_PAIR_BASIS_SCHEMA")
    if s4_pair_basis.get("science_sha256") != a["g4_s4_pair_basis_sha256"]: failures.append("S4_PAIR_BASIS_HASH")
    ops = sorted({tuple(map(int, r["operator"])) for r in s4_pair_basis.get("rows", [])})
    if len(ops) != 31 or [list(x) for x in ops] != s6_spec()["frozen_grammar"]["operator_basis"]: failures.append("OPERATOR_BASIS")
    if s5_result.get("schema_id") != "IG_G4_S5_FINITE_READ_WRITE_DESCRIPTOR_RESULT_V1" or s5_result.get("status") != "PASS": failures.append("S5_SCHEMA_STATUS")
    if s5_result.get("classification") != a["g4_s5_classification"]: failures.append("S5_CLASSIFICATION")
    if s5_result.get("science_sha256") != a["g4_s5_science_sha256"]: failures.append("S5_SCIENCE")
    if s5_result.get("g4_s6_unlocked") is not True or s5_result.get("g4_graduated") is not False: failures.append("S5_STAGE_FIREWALL")
    if s5_result.get("public_descriptor_promoted") is not True or s5_result.get("promoted_descriptor") != a["required_promoted_descriptor"]: failures.append("S5_DESCRIPTOR_PROMOTION")
    if s5_result.get("topology_promoted") is not False or s5_result.get("shell_profile_promoted") is not False: failures.append("S5_STRUCTURE_FIREWALL")
    if s5_replay.get("schema_id") != a["g4_s5_replay_schema_id"] or s5_replay.get("certification") != "CERTIFIED_PASS" or s5_replay.get("status") != "PASS": failures.append("S5_REPLAY")
    if s5_replay.get("comparison_sha256") != a["g4_s5_replay_comparison_sha256"]: failures.append("S5_REPLAY_HASH")
    if not all(s5_replay.get(k) is True for k in ("science_sha_equal","source_sha_equal","registry_sha_equal","stable_science_core_equal")): failures.append("S5_REPLAY_IDENTITY")
    if s5_closeout.get("schema_id") != a["g4_s5_closeout_schema_id"] or s5_closeout.get("status") != "CERTIFIED_PASS": failures.append("S5_CLOSEOUT")
    if s5_closeout.get("closeout_sha256") != a["g4_s5_closeout_sha256"]: failures.append("S5_CLOSEOUT_HASH")
    if s5_closeout.get("g4_s6_unlocked") is not True or s5_closeout.get("g4_graduated") is not False: failures.append("S5_CLOSEOUT_FIREWALL")
    out={
      "schema_id":"IG_G4_S6_AUTHORITY_VERIFICATION_V1","status":"PASS" if not failures else "FAIL","failures":failures,
      "certified_s0_term_corpus_sha256":s0_result.get("certified_term_corpus",{}).get("science_sha256"),
      "s0_term_load_sha256":load_meta.get("science_sha256"),"certified_leaf_refs":sorted(states),
      "tier1_alphabet":alpha,"tier1_ref_labels":ref_labels,
      "s4_pair_basis_sha256":s4_pair_basis.get("science_sha256"),"operator_count":len(ops),
      "g4_s5_science_sha256":s5_result.get("science_sha256"),"g4_s5_replay_comparison_sha256":s5_replay.get("comparison_sha256"),
      "public_descriptor_promoted_before_s6":True,"g4_graduated_before_s6":False,"topology_promoted":False,"shell_profile_promoted":False,
    }
    out["science_sha256"] = canonical_sha256(out)
    if failures:
        raise G4S6Error("G4:S6 authority failed: " + ",".join(failures))
    return out


def verify_regression_evidence(reg: Mapping[str, Any], *, current_source_sha256: str) -> dict[str, Any]:
    failures=[]
    if reg.get("schema_id") != "IG_G4_S6_REGRESSION_GATE_V1": failures.append("SCHEMA")
    if reg.get("status") != "PASS" or int(reg.get("return_code", -1)) != 0: failures.append("STATUS")
    if reg.get("source_sha256") != current_source_sha256: failures.append("SOURCE")
    if int(reg.get("passed", 0)) <= 0: failures.append("NO_PASS_COUNT")
    out={"schema_id":"IG_G4_S6_REGRESSION_VERIFICATION_V1","status":"PASS" if not failures else "FAIL","failures":failures,
         "source_sha256":current_source_sha256,"passed":int(reg.get("passed",0)),"skipped":int(reg.get("skipped",0)),"return_code":int(reg.get("return_code",-1)),
         "stable_projection":"SOURCE_STATUS_PASS_SKIP_COUNTS_V1","operational_fields_excluded_from_science_identity":["duration_seconds","suite_summary"]}
    out["science_sha256"]=canonical_sha256(out)
    if failures: raise G4S6Error("G4:S6 regression gate failed: "+",".join(failures))
    return out


def recursive_descriptor_factorisation_theorem(s5_result: Mapping[str, Any]) -> dict[str, Any]:
    premises={
      "authority":s5_result.get("authority",{}).get("status"),
      "candidate_descriptor_census":s5_result.get("candidate_descriptor_census",{}).get("status"),
      "reserve_factorisation":s5_result.get("reserve_factorisation",{}).get("status"),
      "implementation_read_surface_audit":s5_result.get("implementation_read_surface_audit",{}).get("status"),
      "factorisation_argument":s5_result.get("factorisation_argument",{}).get("status"),
    }
    failures=[k for k,v in premises.items() if v!="PASS"]
    cd=s5_result.get("candidate_descriptor",{})
    if cd.get("name")!="CAPS7_PLUS_TIER1_CLASS_BAG" or int(cd.get("frozen_coordinate_count",-1))!=9: failures.append("DESCRIPTOR")
    proof={
      "method":"STRUCTURAL_INDUCTION_ON_FINITE_COMPLETE_G4_RELATION_TERMS_AND_ACTION_SYNTAX",
      "base":"Every certified G3 unit U is a legal G4 leaf. Its public state is D(U)=(CAPS7(U), one token in its already-certified Tier-1 class). The singleton exact relation {U} is uniform-D.",
      "reserve_step":"Assume complete exact G4 relation R is uniform at D=(f,m). If f_t>0, relation-valued reserve enumerates every eligible recursive owner branch. Every raw successor loses exactly one type-t capacity and changes no constituent Tier-1 class token, hence the complete successor relation is nonempty and uniformly (f-e_t,m).",
      "binary_step":"Assume complete exact relations R,S are uniform at (f,m),(g,n). For any frozen operator (a,b), complete relation-valued composition enumerates all exact branch pairs and all eligible owner reservations. When enabled, every raw output is uniformly (f+g-e_a-e_b,m+n).",
      "dedupe_step":"Exact construction identity is used only after raw branch generation. Exact set deduplication cannot change nonemptiness or a singleton public D image.",
      "closure":"Enabled writes preserve N^7 and add/subtract only finite N^2 Tier-1 bags, so every finite quotient successor remains in the same state carrier.",
      "context_consequence":"By structural induction, D equality is a congruence for every finite admitted G4 reservation/composition context. Exact identity, relation cardinality, branch multiplicity, hidden incidence topology and shell profile may differ but are outside this observer.",
      "rebracketing_scope":"At the D observer, construction trees with the same leaf Tier-1 bag, leaf CAPS7 sum and multiset of directed endpoint consumptions have the same public state. No raw exact-relation or raw-topology associativity is claimed.",
      "finite_action_alphabet":{"reservation_actions":7,"binary_bridge_operators":31,"total_action_schemata":38},
      "minimality_claimed":False,"topology_erasure_claimed":False,
    }
    out={"schema_id":"IG_G4_S6_RECURSIVE_DESCRIPTOR_FACTORISATION_THEOREM_V1","status":"PASS" if not failures else "FAIL","failures":failures,"premises":premises,"proof":proof,
         "consequence":"CAPS7_PLUS_TIER1_CLASS_BAG_IS_A_RECURSIVE_RELATION_QUOTIENT_CONGRUENCE_FOR_ALL_FINITE_TERMS_OF_THE_FROZEN_G4_GRAMMAR" if not failures else "PREMISES_NOT_MET"}
    out["science_sha256"]=canonical_sha256(out); return out


def _expected_descriptor(refs: Sequence[str], states: Mapping[str,G3TermState], ref_to_label: Mapping[str,str], alphabet: Sequence[str], ops: Sequence[Sequence[int]]) -> dict[str,Any]:
    descs=[leaf_descriptor(str(r),states,ref_to_label,alphabet) for r in refs]
    # Public final state depends only on total leaf resources, class bag, and total endpoint consumptions.
    caps=[0]*7; labels=[]
    for r,d in zip(refs,descs):
        dc, _bag=_desc_public(d); caps=[caps[t]+dc[t] for t in range(7)]; labels.append(ref_to_label[str(r)])
    for a,b in (tuple(map(int,x)) for x in ops):
        caps[a]-=1; caps[b]-=1
    if any(x<0 for x in caps): raise G4S6Error("abstract holdout underflow")
    return descriptor_from_public_parts(caps7=caps,tier1_labels=labels,alphabet=alphabet)


def _relation_uniform_descriptor(rel: Sequence[Any], expected: Mapping[str,Any], refs: Sequence[str], ref_to_label: Mapping[str,str], alphabet: Sequence[str]) -> dict[str,Any]:
    if not rel: return {"status":"FAIL","reason":"EMPTY_RELATION"}
    got=descriptor_for_relation(rel,refs,ref_to_label,alphabet)
    return {"status":"PASS" if got==expected else "FAIL","exact_state_count":len(rel),"observed_descriptor_sha256":got.get("science_sha256"),"expected_descriptor_sha256":expected.get("science_sha256"),"caps7_uniform":True,"tier1_bag_invariant":_desc_public(got)[1]==_desc_public(expected)[1]}


def _reservation_sentinel(states: Sequence[Any], expected: Mapping[str,Any], *, max_states: int = 24) -> dict[str,Any]:
    caps, bag=_desc_public(expected); checked=0; succ_count=0; failures=[]
    for st in tuple(states)[:max(1,int(max_states))]:
        if _caps7(st)!=caps:
            failures.append({"reason":"FINAL_CAPS7"}); continue
        checked+=1
        for t in range(7):
            if caps[t]<=0: continue
            succ=_reserve_state(st,t); succ_count+=len(succ)
            q=list(caps); q[t]-=1; q=tuple(q)
            if not succ or any(_caps7(x)!=q for x in succ): failures.append({"endpoint_type":t,"reason":"RESERVE_WRITE"})
    out={"schema_id":"IG_G4_S6_BOUNDED_EXACT_RESERVATION_SENTINEL_V1","status":"PASS" if checked>0 and not failures else "FAIL","selected_exact_states_checked":checked,"exact_successor_states_checked":succ_count,
         "expected_tier1_class_counts":list(bag),"tier1_bag_invariant_by_static_leaf_membership":True,"selector":"CANONICAL_GENERATION_PREFIX_ONLY_FOR_IMPLEMENTATION_TRANSFER_SENTINEL","sentinel_non_authoritative_for_GENERAL_RECURSIVE_PROOF":True,
         "failure_count":len(failures),"failure_examples":failures[:16]}
    out["science_sha256"]=canonical_sha256(out); return out


def execute_pair_plus_pair_case(case: Mapping[str,Any]) -> dict[str,Any]:
    ctx=_require_worker_context(); states=ctx["states"]; ref_labels=ctx["ref_labels"]; alpha=ctx["alphabet"]; basis=ctx["operator_basis"]
    refs=[str(x) for x in case["refs"]]; lop=case["left_operator"]; rop=case["right_operator"]; root=case["root_operator"]
    l=compose_g4_pair_relation(states[refs[0]],states[refs[1]],*map(int,lop)); r=compose_g4_pair_relation(states[refs[2]],states[refs[3]],*map(int,rop))
    if not l or not r: raise G4S6Error("pair+pair child relation empty")
    final=compose_complete_relations(l,r,root)
    expected=_expected_descriptor(refs,states,ref_labels,alpha,[lop,rop,root])
    obs=_relation_uniform_descriptor(final,expected,refs,ref_labels,alpha); sent=_reservation_sentinel(final,expected,max_states=24)
    passed=obs["status"]=="PASS" and sent["status"]=="PASS"
    return {"case_id":case["case_id"],"kind":"PAIR_PLUS_PAIR_P4","status":"PASS" if passed else "FAIL","refs":refs,"operators":[lop,rop,root],"left_pair_cardinality":len(l),"right_pair_cardinality":len(r),"final_relation_cardinality":len(final),"descriptor_audit":obs,"reservation_sentinel":sent,"raw_exact_identity_publicly_observed":False}


def execute_deep_p5_case(case: Mapping[str,Any]) -> dict[str,Any]:
    ctx=_require_worker_context(); states=ctx["states"]; ref_labels=ctx["ref_labels"]; alpha=ctx["alphabet"]
    refs=[str(x) for x in case["refs"]]; ops=[list(map(int,x)) for x in case["operator_sequence"]]
    rel:tuple[Any,...]=(states[refs[0]],)
    # Materialize only through P4. P5 is streamed exhaustively without retaining a full identity set.
    for i in range(1,4):
        rel=compose_complete_relations(rel,(states[refs[i]],),ops[i-1])
        if not rel: raise G4S6Error("deep P5 prefix relation empty")
    p4_card=len(rel); expected=_expected_descriptor(refs,states,ref_labels,alpha,ops)
    exp_caps,_bag=_desc_public(expected); raw=0; sentinels=[]
    for st in iter_complete_compose(rel,(states[refs[4]],),ops[3]):
        raw+=1
        if _caps7(st)!=exp_caps: raise G4S6Error("streamed P5 branch CAPS7 mismatch")
        if len(sentinels)<24: sentinels.append(st)
    if raw<=0: raise G4S6Error("deep P5 final relation empty")
    sent=_reservation_sentinel(sentinels,expected,max_states=24)
    passed=sent["status"]=="PASS"
    return {"case_id":case["case_id"],"kind":"LEFT_DEEP_P5","status":"PASS" if passed else "FAIL","refs":refs,"operators":ops,"p4_exact_relation_cardinality":p4_card,"raw_p5_branches_checked":raw,"all_raw_p5_branches_descriptor_correct":True,"full_p5_exact_identity_frontier_materialized":False,"reservation_sentinel":sent,"expected_descriptor_sha256":expected["science_sha256"],"raw_exact_identity_publicly_observed":False}


def execute_rebracket_case(case: Mapping[str,Any]) -> dict[str,Any]:
    ctx=_require_worker_context(); states=ctx["states"]; ref_labels=ctx["ref_labels"]; alpha=ctx["alphabet"]
    refs=[str(x) for x in case["refs"]]; o0,o1,o2=[list(map(int,x)) for x in case["operator_sequence"]]
    # left-deep: (((0 o0 1) o1 2) o2 3)
    left=compose_complete_relations((states[refs[0]],),(states[refs[1]],),o0)
    left=compose_complete_relations(left,(states[refs[2]],),o1)
    left=compose_complete_relations(left,(states[refs[3]],),o2)
    # balanced: ((0 o0 1) o2 (2 o1 3)) -- same operator multiset/leaf bag.
    a=compose_complete_relations((states[refs[0]],),(states[refs[1]],),o0)
    b=compose_complete_relations((states[refs[2]],),(states[refs[3]],),o1)
    bal=compose_complete_relations(a,b,o2)
    expected=_expected_descriptor(refs,states,ref_labels,alpha,[o0,o1,o2])
    la=_relation_uniform_descriptor(left,expected,refs,ref_labels,alpha); ba=_relation_uniform_descriptor(bal,expected,refs,ref_labels,alpha)
    ls=_reservation_sentinel(left,expected,max_states=16); bs=_reservation_sentinel(bal,expected,max_states=16)
    same=(la.get("observed_descriptor_sha256")==ba.get("observed_descriptor_sha256")==expected.get("science_sha256"))
    passed=all(x.get("status")=="PASS" for x in (la,ba,ls,bs)) and same
    return {"case_id":case["case_id"],"kind":"P4_REBRACKET","status":"PASS" if passed else "FAIL","refs":refs,"operators":[o0,o1,o2],"left_deep_relation_cardinality":len(left),"balanced_relation_cardinality":len(bal),"left_descriptor_audit":la,"balanced_descriptor_audit":ba,"same_public_D":same,"left_reservation_sentinel":ls,"balanced_reservation_sentinel":bs,"raw_relation_equality_claimed":False,"raw_topology_associativity_claimed":False}


def _require_worker_context()->dict[str,Any]:
    if _WORKER_CONTEXT is None: raise G4S6Error("G4:S6 worker context missing")
    return _WORKER_CONTEXT


def init_holdout_worker_from_fork(_: Any) -> None:
    _require_worker_context()


def holdout_worker_context_probe(payload: Mapping[str,Any]) -> dict[str,Any]:
    return {"marker":_require_worker_context().get("probe_marker"),"payload":dict(payload)}



def _checkpoint_name(kind: str, case_id: str) -> str:
    safe = ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(case_id))
    return f"{kind}__{safe}.json"

def write_case_checkpoint(*, checkpoint_dir: str|Path, binding_sha256: str, kind: str, case_id: str, result: Mapping[str,Any]) -> Path:
    p=Path(checkpoint_dir)/_checkpoint_name(kind,case_id)
    obj={"schema_id":"IG_G4_S6_CASE_CHECKPOINT_V1","binding_sha256":str(binding_sha256),"kind":str(kind),"case_id":str(case_id),"result":dict(result)}
    obj["checkpoint_sha256"]=canonical_sha256(obj)
    write_json_atomic(p,obj); return p

def load_case_checkpoint(*, checkpoint_dir: str|Path, binding_sha256: str, kind: str, case_id: str) -> dict[str,Any]|None:
    p=Path(checkpoint_dir)/_checkpoint_name(kind,case_id)
    if not p.exists(): return None
    try: obj=json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None
    got=obj.get("checkpoint_sha256"); exp=canonical_sha256({k:v for k,v in obj.items() if k!="checkpoint_sha256"})
    if got!=exp or obj.get("schema_id")!="IG_G4_S6_CASE_CHECKPOINT_V1" or obj.get("binding_sha256")!=binding_sha256 or obj.get("kind")!=kind or obj.get("case_id")!=case_id: return None
    r=obj.get("result")
    return dict(r) if isinstance(r,dict) else None

def g4_s6_holdout_worker(payload: Mapping[str,Any]) -> dict[str,Any]:
    kind=str(payload["kind"]); case=payload["case"]
    if kind=="PAIR_PLUS_PAIR": result=execute_pair_plus_pair_case(case)
    elif kind=="DEEP_P5": result=execute_deep_p5_case(case)
    elif kind=="REBRACKET": result=execute_rebracket_case(case)
    else: raise G4S6Error("unknown G4:S6 holdout task kind "+kind)
    ctx=_require_worker_context()
    if ctx.get("checkpoint_dir") and ctx.get("binding_sha256"):
        write_case_checkpoint(checkpoint_dir=ctx["checkpoint_dir"],binding_sha256=ctx["binding_sha256"],kind=kind,case_id=str(case["case_id"]),result=result)
    return result


def aggregate_holdout(*, pair_plus_pair: Sequence[Mapping[str,Any]], deep_p5: Sequence[Mapping[str,Any]], rebracket: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    failures=[]
    for family,rows,expected in (("PAIR_PLUS_PAIR",pair_plus_pair,31),("DEEP_P5",deep_p5,8),("REBRACKET",rebracket,8)):
        if len(rows)!=expected: failures.append(family+"_COUNT")
        if any(r.get("status")!="PASS" for r in rows): failures.append(family+"_FAIL")
    out={"schema_id":"IG_G4_S6_FRESH_RECURSIVE_HOLDOUT_RESULT_V1","status":"PASS" if not failures else "FAIL","failures":failures,
         "freshness_scope":s6_spec()["fresh_holdout"]["freshness_scope"],"pair_plus_pair_case_count":len(pair_plus_pair),"deep_p5_case_count":len(deep_p5),"rebracketing_case_count":len(rebracket),
         "pair_plus_pair_cases_sha256":canonical_sha256(list(pair_plus_pair)),"deep_p5_cases_sha256":canonical_sha256(list(deep_p5)),"rebracket_cases_sha256":canonical_sha256(list(rebracket)),
         "all_pair_plus_pair_operator_cases_covered":len(pair_plus_pair)==31,"all_deep_p5_raw_branches_checked":all(bool(r.get("all_raw_p5_branches_descriptor_correct")) for r in deep_p5),
         "all_rebracket_public_D_equal":all(bool(r.get("same_public_D")) for r in rebracket),"topology_promoted":False,"shell_profile_promoted":False}
    out["science_sha256"]=canonical_sha256(out); return out


def implementation_no_hidden_selector_audit() -> dict[str,Any]:
    from . import uplift_g4_s5
    abstract_src=inspect.getsource(uplift_g4_s5.descriptor_from_public_parts)+inspect.getsource(uplift_g4_s5.reserve_write)+inspect.getsource(uplift_g4_s5.binary_write)
    reserve_src=inspect.getsource(G4RecursiveTermState.reserve_external_relation)
    compose_src=inspect.getsource(iter_complete_compose)
    failures=[]
    for tok in ("topology_canon","typed_edges","node_caps","shell_profile","ancestry","owner_witness"):
        if tok in abstract_src: failures.append("ABSTRACT_FORBIDDEN_READ_"+tok.upper())
    required_res=("for ls in _reserve_state(self.left, t)","for rs in _reserve_state(self.right, t)","out.setdefault(st.construction_digest, st)")
    if any(tok not in reserve_src for tok in required_res): failures.append("RESERVE_COMPLETE_OWNER_ENUMERATION_SOURCE_SHAPE")
    required_comp=("for left in left_rel","lres = _reserve_state(left, a)","for right in right_rel","rres = _reserve_state(right, b)","for ls in lres","for rs in rres")
    if any(tok not in compose_src for tok in required_comp): failures.append("COMPOSE_COMPLETE_CARTESIAN_SOURCE_SHAPE")
    out={"schema_id":"IG_G4_S6_NO_HIDDEN_SELECTOR_READ_AUDIT_V1","status":"PASS" if not failures else "FAIL","failures":failures,
         "promotable_abstract_reads":["CAPS7","Tier1_class_bag","endpoint types","frozen 31-operator basis"],
         "promotable_abstract_does_not_read":["construction identity","owner identity","hidden G3 topology","shell profile","ancestry","exact relation cardinality","exact branch multiplicity"],
         "construction_identity_role":"OPERATIONAL_EXACT_DEDUP_AND_BOUNDED_HOLDOUT_ONLY_NO_PUBLIC_SELECTOR","hidden_topology_read":False,"shell_profile_read":False,
         "source_snippet_hashes":{"abstract":hashlib.sha256(abstract_src.encode()).hexdigest(),"reserve":hashlib.sha256(reserve_src.encode()).hexdigest(),"compose":hashlib.sha256(compose_src.encode()).hexdigest()}}
    out["science_sha256"]=canonical_sha256(out); return out


def stable_science_payload(*, authority:Mapping[str,Any], regression:Mapping[str,Any], theorem:Mapping[str,Any], holdout:Mapping[str,Any], hidden:Mapping[str,Any]) -> dict[str,Any]:
    out={"schema_id":"IG_G4_S6_STABLE_SCIENCE_PAYLOAD_V1","g4_s6_spec_sha256":s6_spec()["science_sha256"],"authority_science_sha256":authority["science_sha256"],"regression_gate_science_sha256":regression["science_sha256"],"recursive_theorem_science_sha256":theorem["science_sha256"],"fresh_holdout_science_sha256":holdout["science_sha256"],"no_hidden_read_audit_science_sha256":hidden["science_sha256"],"observer":s6_spec()["observer"]["name"],"graduated_descriptor_candidate":"CAPS7_PLUS_TIER1_CLASS_BAG"}
    out["science_sha256"]=canonical_sha256(out); return out


def primary_result(*, authority:Mapping[str,Any], regression:Mapping[str,Any], theorem:Mapping[str,Any], holdout:Mapping[str,Any], hidden:Mapping[str,Any]) -> dict[str,Any]:
    passed=all(x.get("status")=="PASS" for x in (authority,regression,theorem,holdout,hidden)); stable=stable_science_payload(authority=authority,regression=regression,theorem=theorem,holdout=holdout,hidden=hidden)
    out={"schema_id":"IG_G4_S6_PRIMARY_RECURSIVE_CLOSURE_RESULT_V1","schema_version":"1.0.0","date":"2026-09-04","stage_ref":"G4:S6","status":"PASS" if passed else "REVIEW_REQUIRED",
         "classification":"G4_CAPS7_PLUS_TIER1_BAG_RECURSIVE_RELATION_CLOSURE_EARNED_GRADUATION_CANDIDATE_AWAITING_COLD_REPLAY" if passed else "G4_S6_RECURSIVE_CLOSURE_GATE_FAILED_G4_NOT_GRADUATED",
         "authority":dict(authority),"regression_gate":dict(regression),"recursive_descriptor_factorisation_theorem":dict(theorem),"fresh_recursive_holdout":dict(holdout),"no_hidden_selector_read_audit":dict(hidden),
         "stable_science_payload":stable,"stable_science_payload_sha256":stable["science_sha256"],"cold_replay_required_for_graduation":True,"g4_graduation_candidate":passed,"g4_graduated":False,"r0_unlocked":False,
         "public_descriptor_promoted":True,"promoted_descriptor":"CAPS7_PLUS_TIER1_CLASS_BAG","topology_promoted":False,"shell_profile_promoted":False,
         "graduation_scope_if_cold_replay_passes":"FROZEN_G4_CAPS7_PLUS_TIER1_CLASS_BAG_TRANSITION_OBSERVER_WITH_7_RESERVATION_ACTIONS_AND_31_RELATION_VALUED_TYPED_BINARY_OPERATORS_OVER_ALL_FINITE_COMPLETE_G4_RELATION_TERMS_BUILT_FROM_CERTIFIED_G3_UNITS",
         "nonclaims":["DESCRIPTOR_MINIMALITY_NOT_CLAIMED","RAW_EXACT_RELATION_EQUIVALENCE_NOT_CLAIMED","EXACT_RELATION_CARDINALITY_OR_BRANCH_MULTIPLICITY_EQUIVALENCE_NOT_CLAIMED","RAW_TOPOLOGY_ASSOCIATIVITY_NOT_CLAIMED","TOPOLOGY_ERASURE_NOT_CLAIMED","SHELL_PROFILE_ERASURE_NOT_CLAIMED","NO_PHYSICAL_GEOMETRY_DIMENSION_TIME_OR_SPACETIME_CLAIM"]}
    out["science_sha256"]=canonical_sha256({k:v for k,v in out.items() if k not in {"source_sha256","source_version","registry_sha256","execution_metadata","science_sha256"}}); return out


def compare_cold_replay(primary:Mapping[str,Any], cold:Mapping[str,Any]) -> dict[str,Any]:
    failures=[]
    for lab,r in (("PRIMARY",primary),("COLD",cold)):
        if r.get("schema_id")!="IG_G4_S6_PRIMARY_RECURSIVE_CLOSURE_RESULT_V1": failures.append(lab+"_SCHEMA")
        if r.get("status")!="PASS" or r.get("g4_graduation_candidate") is not True: failures.append(lab+"_STATUS")
        md=r.get("execution_metadata",{})
        if md.get("registered_experiment_id")!="G4:S6" or md.get("execution_backend_owned_by_decoder") is not True or md.get("stage_specific_external_science_runner") is not False: failures.append(lab+"_EXECUTION")
    exact=primary.get("stable_science_payload")==cold.get("stable_science_payload") and primary.get("stable_science_payload_sha256")==cold.get("stable_science_payload_sha256")
    checks={"stable_scientific_payload_exact_equal":exact,"science_sha_equal":primary.get("science_sha256")==cold.get("science_sha256"),"source_sha_equal":primary.get("source_sha256")==cold.get("source_sha256"),"registry_sha_equal":primary.get("registry_sha256")==cold.get("registry_sha256"),"classification_equal":primary.get("classification")==cold.get("classification")}
    failures.extend(k.upper() for k,v in checks.items() if not v)
    out={"schema_id":"IG_G4_S6_COLD_REPLAY_COMPARISON_V1","status":"PASS" if not failures else "FAIL","certification":"CERTIFIED_PASS" if not failures else "NOT_CERTIFIED","failures":failures,"experiment_id":"G4:S6","same_registered_decoder_native_experiment":True,**checks,
         "stable_scientific_payload_sha256":primary.get("stable_science_payload_sha256"),"primary_science_sha256":primary.get("science_sha256"),"cold_science_sha256":cold.get("science_sha256"),"primary_source_sha256":primary.get("source_sha256"),"cold_source_sha256":cold.get("source_sha256"),"primary_registry_sha256":primary.get("registry_sha256"),"cold_registry_sha256":cold.get("registry_sha256")}
    out["comparison_sha256"]=canonical_sha256(out); return out


def graduation_certificate(primary:Mapping[str,Any], replay:Mapping[str,Any]) -> dict[str,Any]:
    failures=[]
    if primary.get("status")!="PASS" or primary.get("g4_graduation_candidate") is not True: failures.append("PRIMARY")
    if replay.get("schema_id")!="IG_G4_S6_COLD_REPLAY_COMPARISON_V1" or replay.get("certification")!="CERTIFIED_PASS" or replay.get("status")!="PASS": failures.append("REPLAY")
    if not all(replay.get(k) is True for k in ("stable_scientific_payload_exact_equal","science_sha_equal","source_sha_equal","registry_sha_equal")): failures.append("IDENTITY")
    passed=not failures
    out={"schema_id":"IG_G4_GRADUATION_CERTIFICATE_V1","date":"2026-09-04","status":"PASS" if passed else "FAIL","classification":"G4_GRADUATED_CAPS7_PLUS_TIER1_BAG_RECURSIVE_RELATION_GRAMMAR_EARNED_R0_UNLOCKED" if passed else "G4_NOT_GRADUATED_R0_LOCKED","failures":failures,
         "g4_graduated":passed,"r0_unlocked":passed,"graduated_observer":s6_spec()["observer"]["name"],"graduated_descriptor":"CAPS7_PLUS_TIER1_CLASS_BAG",
         "graduated_grammar":{"reservation_actions":7,"binary_operator_count":31,"routing_semantics":"RELATION_VALUED_ALL_ELIGIBLE_OWNER_CHOICES_V1","recursive_scope":"ALL_FINITE_COMPLETE_G4_RELATION_TERMS_BUILT_FROM_CERTIFIED_G3_UNITS"},
         "primary_s6_science_sha256":primary.get("science_sha256"),"stable_s6_science_payload_sha256":primary.get("stable_science_payload_sha256"),"cold_replay_comparison_sha256":replay.get("comparison_sha256"),
         "public_descriptor_promoted":passed,"topology_promoted":False,"shell_profile_promoted":False,"authorizes":"G4:R0_POST_GRADUATION_STRUCTURAL_RECONNAISSANCE" if passed else None,
         "nonclaims":["DESCRIPTOR_MINIMALITY_NOT_CLAIMED","RAW_EXACT_RELATION_EQUIVALENCE_NOT_CLAIMED","EXACT_RELATION_CARDINALITY_OR_BRANCH_MULTIPLICITY_EQUIVALENCE_NOT_CLAIMED","TOPOLOGY_ERASURE_NOT_CLAIMED","SHELL_PROFILE_ERASURE_NOT_CLAIMED","NO_GEOMETRY_OR_PHYSICS_CLAIM"],
         "reopen_conditions":["G4 public observer changes","relation-valued G4 routing changes","certified G3 leaf semantics change","31 bridge-operator basis changes","Tier-1 class definition changes"]}
    out["science_sha256"]=canonical_sha256(out); return out


def certified_closeout(primary:Mapping[str,Any], cold:Mapping[str,Any], replay:Mapping[str,Any], graduation:Mapping[str,Any]) -> dict[str,Any]:
    failures=[]
    if primary.get("status")!="PASS" or cold.get("status")!="PASS": failures.append("SCIENCE_STATUS")
    if replay.get("certification")!="CERTIFIED_PASS": failures.append("REPLAY")
    if graduation.get("status")!="PASS" or graduation.get("g4_graduated") is not True: failures.append("GRADUATION")
    passed=not failures
    out={"schema_id":"IG_G4_S6_CERTIFIED_CLOSEOUT_V1","status":"CERTIFIED_PASS" if passed else "NOT_CERTIFIED","failures":failures,"experiment_id":"G4:S6","classification":graduation.get("classification") if passed else None,
         "science_sha256":primary.get("science_sha256"),"stable_science_payload_sha256":primary.get("stable_science_payload_sha256"),"source_sha256":primary.get("source_sha256"),"registry_sha256":primary.get("registry_sha256"),"comparison_sha256":replay.get("comparison_sha256"),"graduation_certificate_sha256":graduation.get("science_sha256"),
         "g4_graduated":passed,"r0_unlocked":passed,"public_descriptor_promoted":passed,"promoted_descriptor":"CAPS7_PLUS_TIER1_CLASS_BAG" if passed else None,"topology_promoted":False,"shell_profile_promoted":False,"next_authorized_stage":"G4:R0" if passed else None}
    out["closeout_sha256"]=canonical_sha256(out); return out
