from __future__ import annotations

"""Registered G3:S2 exact CAPS7 pair-observer quotient certification.

S2 is a deterministic quotient over the already-certified G3:S0 collision corpus and
registered G3:S1 pair-context evidence. It does not execute a new topology-reading
observer. It asks the Phase-0 question for S2: after S1 found no topology read, does
the *complete exact pair-context evidence* factor through inherited CAPS7 on the
frozen collision domain?
"""

from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import gzip, json

from .canon import canonical_sha256
from .uplift_g3_s0 import phase0_spec


class G3S2Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s2_spec() -> dict[str, Any]:
    obj=json.loads(_resource("G3_S2_CAPS7_PAIR_OBSERVER_QUOTIENT_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id")!="IG_G3_S2_CAPS7_PAIR_OBSERVER_QUOTIENT_SPEC_V1":
        raise G3S2Error("bad G3:S2 spec schema")
    payload={k:v for k,v in obj.items() if k!="science_sha256"}
    if canonical_sha256(payload)!=obj.get("science_sha256"):
        raise G3S2Error("G3:S2 spec hash mismatch")
    if obj.get("phase0_spec_sha256")!=phase0_spec().get("science_sha256"):
        raise G3S2Error("G3:S2/Phase0 binding mismatch")
    return obj


def verify_s2_authority(*, s0_result: Mapping[str,Any], s1_result: Mapping[str,Any]) -> dict[str,Any]:
    a=s2_spec()["authority"]
    failures=[]
    if s0_result.get("status")!="PASS": failures.append("S0_STATUS")
    if s0_result.get("science_sha256")!=a["g3_s0_science_sha256"]: failures.append("S0_SHA")
    if s0_result.get("classification")!=a["g3_s0_classification"]: failures.append("S0_CLASSIFICATION")
    if s1_result.get("status")!="PASS": failures.append("S1_STATUS")
    if s1_result.get("science_sha256")!=a["g3_s1_science_sha256"]: failures.append("S1_SHA")
    if s1_result.get("classification")!=a["g3_s1_classification"]: failures.append("S1_CLASSIFICATION")
    if s1_result.get("outcome")!=a["g3_s1_outcome"]: failures.append("S1_OUTCOME")
    if int(s1_result.get("context_basis",{}).get("task_count",-1))!=int(a["required_s1_task_count"]): failures.append("S1_TASK_COUNT")
    if bool(s1_result.get("g3_s2_unlocked")) is not bool(a["required_s1_s2_unlocked"]): failures.append("S2_UNLOCK")
    if bool(s1_result.get("topology_promoted")) is not bool(a["required_topology_promoted"]): failures.append("TOPOLOGY_PROMOTION")
    if bool(s1_result.get("g3_graduated")) is not bool(a["required_g3_graduated"]): failures.append("G3_GRADUATION")
    if failures:
        raise G3S2Error("G3:S2 authority verification failed: "+",".join(failures))
    out={
      "schema_id":"IG_G3_S2_AUTHORITY_V1","status":"PASS",
      "g3_s0_science_sha256":str(s0_result["science_sha256"]),
      "g3_s1_science_sha256":str(s1_result["science_sha256"]),
      "g3_phase0_science_sha256":str(phase0_spec()["science_sha256"]),
      "g3_s2_spec_sha256":str(s2_spec()["science_sha256"]),
    }
    out["science_sha256"]=canonical_sha256(out)
    return out


def load_s1_signature_index(path: str|Path) -> dict[str,Any]:
    p=Path(path)
    if not p.exists(): raise G3S2Error("G3:S1 signature index missing")
    with gzip.open(p,"rt",encoding="utf-8") as fh: obj=json.load(fh)
    if obj.get("schema_id")!="IG_G3_S1_PAIR_CONTEXT_SIGNATURE_INDEX_V1": raise G3S2Error("bad S1 signature index schema")
    if int(obj.get("task_count",-1))!=len(obj.get("rows",[])): raise G3S2Error("S1 signature index count mismatch")
    return obj


def certify_caps7_pair_observer_quotient(*, s0_result:Mapping[str,Any], s1_result:Mapping[str,Any], signature_index:Mapping[str,Any]) -> dict[str,Any]:
    spec=s2_spec(); auth=verify_s2_authority(s0_result=s0_result,s1_result=s1_result)
    challenge=list(s0_result.get("challenge_corpus",{}).get("records",[]))
    if len(challenge)!=6: raise G3S2Error(f"expected six S0 challenge carriers, got {len(challenge)}")
    refs=sorted(str(r["g2_carrier_ref"]) for r in challenge)
    if len(refs)!=len(set(refs)): raise G3S2Error("duplicate S0 challenge carrier ref")
    caps_by_ref={str(r["g2_carrier_ref"]):str(r["public_interface"]["science_sha256"]) for r in challenge}
    caps_classes=defaultdict(list)
    for ref,h in caps_by_ref.items(): caps_classes[h].append(ref)
    if len(caps_classes)!=1: raise G3S2Error("frozen S0 challenge domain is not one CAPS7 collision class")
    topology_by_ref={str(r["g2_carrier_ref"]):str(r["hidden_challenge_diagnostic"]["topology_canon"]) for r in challenge}
    if len(set(topology_by_ref.values()))<2: raise G3S2Error("S2 challenge domain lacks distinct hidden topology classes")

    rows=list(signature_index.get("rows",[]))
    expected=int(spec["input_evidence"]["expected_exact_pair_rows"])
    if len(rows)!=expected: raise G3S2Error(f"expected {expected} S1 rows, got {len(rows)}")
    seen=set(); grouped=defaultdict(list); per_target=defaultdict(list)
    for row in rows:
        tr=str(row.get("target_ref")); cr=str(row.get("context_ref"))
        if tr not in caps_by_ref or cr not in caps_by_ref: raise G3S2Error("S1 row references carrier outside frozen S0 domain")
        op=tuple(map(int,row.get("operator",[]))); ori=str(row.get("orientation"))
        if len(op)!=2 or ori not in {"TARGET_LEFT_CONTEXT_RIGHT","CONTEXT_LEFT_TARGET_RIGHT"}: raise G3S2Error("malformed S1 context key")
        exact_key=(tr,cr,op,ori)
        if exact_key in seen: raise G3S2Error("duplicate exact S1 context row")
        seen.add(exact_key)
        sig=row.get("operational_signature",{})
        if sig.get("direct_topology_read") is not False or sig.get("output_construction_identity_recorded") is not False:
            raise G3S2Error("S1 evidence contains forbidden direct topology/identity read")
        sh=str(sig.get("science_sha256"))
        if canonical_sha256({k:v for k,v in sig.items() if k!="science_sha256"})!=sh:
            raise G3S2Error("S1 operational signature hash mismatch")
        pubkey=(caps_by_ref[tr],caps_by_ref[cr],op,ori)
        grouped[pubkey].append((tr,cr,sh))
        per_target[tr].append({"context_ref":cr,"operator":list(op),"orientation":ori,"operational_signature_sha256":sh})

    public_key_count=len(grouped)
    expected_keys=int(spec["input_evidence"]["expected_public_context_keys"])
    reps_expected=int(spec["input_evidence"]["expected_representatives_per_public_context_key"])
    conflicts=[]; quotient_rows=[]
    for key in sorted(grouped,key=lambda k:(k[0],k[1],k[2],k[3])):
        vals=grouped[key]; hashes=sorted({x[2] for x in vals})
        row={
          "target_caps7_sha256":key[0],"context_caps7_sha256":key[1],
          "operator":list(key[2]),"orientation":key[3],
          "exact_representative_pair_count":len(vals),
          "observer_value_count":len(hashes),"observer_signature_sha256":hashes[0] if len(hashes)==1 else None,
        }
        quotient_rows.append(row)
        if len(vals)!=reps_expected or len(hashes)!=1:
            conflicts.append({**row,"observer_hashes":hashes[:8]})
    if public_key_count!=expected_keys: conflicts.append({"failure":"PUBLIC_CONTEXT_KEY_COUNT","observed":public_key_count,"expected":expected_keys})

    # Reproduce target behavior hashes exactly from the raw S1 evidence and compare to S1 authority.
    target_kernel=defaultdict(list); reconstructed=[]
    auth_summary={str(r["target_ref"]):str(r["behavior_signature_sha256"]) for r in s1_result.get("target_behavior_summaries",[])}
    for tr in refs:
        table=sorted(per_target[tr],key=lambda r:(r["context_ref"],r["orientation"],r["operator"]))
        bh=canonical_sha256(table)
        if auth_summary.get(tr)!=bh:
            raise G3S2Error(f"cannot reproduce certified S1 target behavior hash for {tr}")
        target_kernel[bh].append(tr)
        reconstructed.append({"target_ref":tr,"behavior_signature_sha256":bh,"caps7_sha256":caps_by_ref[tr]})

    caps_partition=sorted(sorted(v) for v in caps_classes.values())
    observer_partition=sorted(sorted(v) for v in target_kernel.values())
    partitions_equal=caps_partition==observer_partition
    if not partitions_equal: conflicts.append({"failure":"CAPS7_OBSERVER_KERNEL_PARTITION_MISMATCH","caps_partition":caps_partition,"observer_partition":observer_partition})

    passed=not conflicts
    classification=(
      "G3_CAPS7_PAIR_OBSERVER_QUOTIENT_CERTIFIED_NO_ADDED_READ_S3_UNLOCKED"
      if passed else "G3_CAPS7_PAIR_OBSERVER_QUOTIENT_NOT_CERTIFIED_S3_LOCKED"
    )
    result={
      "schema_id":"IG_G3_S2_CAPS7_PAIR_OBSERVER_QUOTIENT_RESULT_V1",
      "status":"PASS" if passed else "REVIEW_REQUIRED",
      "stage_ref":"G3:S2","classification":classification,
      "authority":auth,
      "frozen_domain":{"exact_carrier_count":len(refs),"caps7_class_count":len(caps_classes),"hidden_topology_class_count":len(set(topology_by_ref.values())),"s1_exact_pair_context_rows":len(rows)},
      "quotient_audit":{
        "public_pair_context_key_count":public_key_count,
        "expected_public_pair_context_key_count":expected_keys,
        "representatives_per_public_context_key_expected":reps_expected,
        "representative_conflict_count":len(conflicts),
        "all_public_pair_context_keys_single_valued":all(r["observer_value_count"]==1 and r["exact_representative_pair_count"]==reps_expected for r in quotient_rows),
        "caps7_partition_equals_operational_target_kernel":partitions_equal,
        "caps7_partition_class_count":len(caps_partition),
        "observer_kernel_class_count":len(observer_partition),
      },
      "minimal_added_read_at_tested_pair_observer":"NONE" if passed else None,
      "topology_promoted":False,
      "g3_s3_unlocked":passed,
      "g3_graduated":False,
      "conflicts":conflicts[:16],
      "quotient_table":quotient_rows,
      "reconstructed_target_behavior":reconstructed,
      "scope":"EXACT_ONLY_ON_FROZEN_G3_S0_COLLISION_DOMAIN_UNDER_COMPLETE_REGISTERED_G3_S1_PAIR_CONTEXT_OBSERVER",
      "nonclaims":list(spec["nonclaims"]),
    }
    result["science_sha256"]=canonical_sha256(result)
    return result
