from __future__ import annotations

"""Registered G3:S1 pair-context topology-read audit.

S1 does not read topology while executing contexts.  It applies the same frozen
whole-unit relation contexts to every member of the G3:S0 equal-CAPS7 challenge
corpus, records only branch-sensitive operational behavior allowed by the S1
specification, and uses the already-frozen S0 topology labels *afterward* to ask
whether operational behavior is congruent within and/or separated between the
S0 topology classes.
"""

from collections import Counter, defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence
import json

from .canon import canonical_sha256
from .g2_relation import compose_binary_relation, reserve_external_relation
from .uplift_r0 import _choose_growth_carrier_and_operator
from .uplift_g3_s0 import phase0_spec


class G3S1Error(RuntimeError):
    pass


def _resource(name: str) -> Path:
    return Path(str(files("infinity_grid").joinpath("resources/uplift").joinpath(name)))


def s1_spec() -> dict[str, Any]:
    obj = json.loads(_resource("G3_S1_PAIR_CONTEXT_READ_SPEC_V1.json").read_text(encoding="utf-8"))
    if obj.get("schema_id") != "IG_G3_S1_PAIR_CONTEXT_READ_SPEC_V1":
        raise G3S1Error("bad G3:S1 spec schema")
    payload = {k: v for k, v in obj.items() if k != "science_sha256"}
    if canonical_sha256(payload) != obj.get("science_sha256"):
        raise G3S1Error("G3:S1 spec hash mismatch")
    if obj.get("phase0_spec_sha256") != phase0_spec().get("science_sha256"):
        raise G3S1Error("G3:S1/Phase0 binding mismatch")
    return obj


def verify_s1_authority(s0_result: Mapping[str, Any]) -> dict[str, Any]:
    a = s1_spec()["s0_authority"]
    if s0_result.get("status") != "PASS":
        raise G3S1Error("G3:S0 authority is not PASS")
    if s0_result.get("classification") != a["classification"]:
        raise G3S1Error("G3:S0 classification authority mismatch")
    if s0_result.get("science_sha256") != a["science_sha256"]:
        raise G3S1Error("G3:S0 science hash authority mismatch")
    if bool(s0_result.get("g3_s1_unlocked")) is not bool(a["required_g3_s1_unlocked"]):
        raise G3S1Error("G3:S0 S1-unlock authority mismatch")
    if bool(s0_result.get("g3_graduated")) is not bool(a["required_g3_graduated"]):
        raise G3S1Error("G3:S0 graduation authority mismatch")
    out = {
        "schema_id": "IG_G3_S1_AUTHORITY_V1",
        "status": "PASS",
        "g3_s0_science_sha256": str(s0_result["science_sha256"]),
        "g3_phase0_science_sha256": str(phase0_spec()["science_sha256"]),
    }
    out["science_sha256"] = canonical_sha256(out)
    return out


def reproduce_s0_challenge_states(*, engine: Any, s0_result: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rebuild exactly the six S0 four-G1-unit challenge carriers.

    Exact construction refs are used only to bind/recover the frozen authority
    corpus.  They are never used as output descriptors or context selectors.
    """
    spec = s1_spec()
    auth = verify_s1_authority(s0_result)
    rows = list(s0_result.get("challenge_corpus", {}).get("records", []))
    expected_n = int(spec["challenge_domain"]["expected_exact_carrier_count"])
    if len(rows) != expected_n:
        raise G3S1Error(f"G3:S1 expected {expected_n} S0 challenge records, got {len(rows)}")
    refs = [str(r.get("g2_carrier_ref")) for r in rows]
    if len(refs) != len(set(refs)):
        raise G3S1Error("G3:S0 challenge refs are not unique")
    iface_hashes = {str(r.get("public_interface", {}).get("science_sha256")) for r in rows}
    if spec["challenge_domain"].get("required_single_caps7_class") and len(iface_hashes) != 1:
        raise G3S1Error("G3:S1 challenge corpus is not one CAPS7 class")
    hist = Counter(str(r.get("hidden_challenge_diagnostic", {}).get("topology_canon")) for r in rows)
    expected_hist = {str(k): int(v) for k, v in spec["challenge_domain"]["expected_topology_class_histogram"].items()}
    if dict(sorted(hist.items())) != dict(sorted(expected_hist.items())):
        raise G3S1Error(f"G3:S1 topology authority histogram mismatch: {dict(hist)}")

    carriers = engine.ensure_g1_r100_population()
    bridge_pairs = sorted({tuple(map(int, x)) for x in engine.session.bridge_pairs})
    if len(bridge_pairs) != 31:
        raise G3S1Error("G3:S1 requires the complete frozen 31-operator bridge basis")
    seed, op = _choose_growth_carrier_and_operator(carriers, bridge_pairs)
    a, b = map(int, op)
    states = [seed]
    for n in range(2, 5):
        nxt: dict[str, Any] = {}
        for st in states:
            outs = compose_binary_relation(
                engine.session.engine,
                100 + n,
                st,
                seed,
                a,
                b,
                lane="G2_R0_TREE_GROWTH",
                motif_id=f"G2:R0:TREE:{n}:{a}>{b}",
            )
            for out in outs:
                nxt.setdefault(str(out.construction_digest), out)
        if not nxt:
            raise G3S1Error(f"G3:S1 authority reproduction exhausted at n={n}")
        states = [nxt[k] for k in sorted(nxt)]
    by_ref = {str(st.construction_digest): st for st in states}
    if set(by_ref) != set(refs):
        missing = sorted(set(refs) - set(by_ref))
        extra = sorted(set(by_ref) - set(refs))
        raise G3S1Error(f"G3:S1 exact S0 corpus reproduction mismatch; missing={missing[:3]} extra={extra[:3]}")
    meta = {
        "schema_id": "IG_G3_S1_CHALLENGE_CORPUS_REPRODUCTION_V1",
        "status": "PASS",
        "authority": auth,
        "exact_carrier_count": len(by_ref),
        "topology_class_histogram": dict(sorted(hist.items())),
        "single_caps7_class": len(iface_hashes) == 1,
        "reproduction_growth_operator": [a, b],
        "reproduction_recipe": "EXACT_CERTIFIED_G3_S0_G2_R0_HOMOGENEOUS_GROWTH_RECIPE",
        "seed_g1_ref": str(seed.construction_digest),
        "operator_count": len(bridge_pairs),
        "context_target_count": len(by_ref),
    }
    meta["science_sha256"] = canonical_sha256(meta)
    return by_ref, meta


def _caps7(st: Any) -> list[int]:
    caps = [int(x) for x in st.total_caps]
    if len(caps) != 7 or any(x < 0 for x in caps):
        raise G3S1Error("malformed CAPS7 during G3:S1")
    return caps


def pair_context_signature(*, engine: Any, target: Any, context: Any, operator: Sequence[int], orientation: str) -> dict[str, Any]:
    """Strict operational signature with no direct topology or output-identity read."""
    if orientation not in {"TARGET_LEFT_CONTEXT_RIGHT", "CONTEXT_LEFT_TARGET_RIGHT"}:
        raise G3S1Error(f"bad G3:S1 orientation {orientation}")
    a, b = map(int, operator)
    if orientation == "TARGET_LEFT_CONTEXT_RIGHT":
        left, right = target, context
    else:
        left, right = context, target
    level = max(int(left.level), int(right.level)) + 1
    outputs = compose_binary_relation(
        engine,
        level,
        left,
        right,
        a,
        b,
        lane="G3_S1_PAIR_CONTEXT",
        motif_id=f"G3:S1:PAIR:{orientation}:{a}>{b}",
    )

    public_caps_set = sorted({tuple(_caps7(st)) for st in outputs})
    branch_payloads: dict[str, dict[str, Any]] = {}
    branch_hist = Counter()
    for st in outputs:
        caps = _caps7(st)
        rprof = []
        for t in range(7):
            rel = reserve_external_relation(st, t) if caps[t] > 0 else tuple()
            succ_caps = sorted({tuple(_caps7(s)) for s in rel})
            row = {
                "endpoint_type": t,
                "available": caps[t] > 0,
                "relation_cardinality": len(rel),
                "successor_caps7_set": [list(x) for x in succ_caps],
            }
            rprof.append(row)
        bp = {"output_caps7": caps, "reservation_profile": rprof}
        h = canonical_sha256(bp)
        branch_payloads.setdefault(h, bp)
        branch_hist[h] += 1

    sig = {
        "schema_id": "IG_G3_S1_OPERATIONAL_PAIR_CONTEXT_SIGNATURE_V1",
        "operator": [a, b],
        "orientation": orientation,
        "relation_output_cardinality": len(outputs),
        "public_output_caps7_set": [list(x) for x in public_caps_set],
        "branch_profile_histogram": [
            {"branch_profile_sha256": h, "multiplicity": int(branch_hist[h]), "profile": branch_payloads[h]}
            for h in sorted(branch_hist)
        ],
        "direct_topology_read": False,
        "output_construction_identity_recorded": False,
    }
    sig["science_sha256"] = canonical_sha256(sig)
    return sig


def finalize_s1_result(*, s0_result: Mapping[str, Any], reproduction: Mapping[str, Any], task_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spec = s1_spec()
    auth = verify_s1_authority(s0_result)
    challenge_rows = list(s0_result["challenge_corpus"]["records"])
    refs = sorted(str(r["g2_carrier_ref"]) for r in challenge_rows)
    topo_by_ref = {str(r["g2_carrier_ref"]): str(r["hidden_challenge_diagnostic"]["topology_canon"]) for r in challenge_rows}
    expected_tasks = len(refs) * len(refs) * 31 * 2
    if len(task_rows) != expected_tasks:
        raise G3S1Error(f"G3:S1 expected {expected_tasks} task rows, got {len(task_rows)}")

    # Build one canonical behavior table per target. Topology labels are consulted only now,
    # after every context task has returned its topology-blind operational signature.
    per_target: dict[str, list[dict[str, Any]]] = {r: [] for r in refs}
    seen = set()
    row_by_key: dict[tuple[str, str, str, int, int], Mapping[str, Any]] = {}
    for row in task_rows:
        tr = str(row["target_ref"]); cr = str(row["context_ref"]); ori = str(row["orientation"])
        a, b = map(int, row["operator"])
        key = (tr, cr, ori, a, b)
        if key in seen:
            raise G3S1Error("duplicate G3:S1 task key")
        seen.add(key); row_by_key[key] = row
        per_target[tr].append({
            "context_ref": cr,
            "orientation": ori,
            "operator": [a, b],
            "operational_signature_sha256": str(row["operational_signature"]["science_sha256"]),
        })
    target_summaries = []
    behavior_hash: dict[str, str] = {}
    for tr in refs:
        rows = sorted(per_target[tr], key=lambda x: (x["context_ref"], x["orientation"], x["operator"]))
        bh = canonical_sha256(rows)
        behavior_hash[tr] = bh
        target_summaries.append({"target_ref": tr, "frozen_topology_label": topo_by_ref[tr], "behavior_signature_sha256": bh, "context_row_count": len(rows)})

    groups: dict[str, list[str]] = defaultdict(list)
    for tr in refs:
        groups[topo_by_ref[tr]].append(tr)
    within = []
    within_fail = False
    class_hash: dict[str, str] = {}
    for topo in sorted(groups):
        hs = sorted({behavior_hash[r] for r in groups[topo]})
        ok = len(hs) == 1
        within_fail = within_fail or not ok
        within.append({"topology_label": topo, "carrier_count": len(groups[topo]), "behavior_signature_hashes": hs, "congruent": ok})
        if ok:
            class_hash[topo] = hs[0]

    separation_witness = None
    if not within_fail and len(set(class_hash.values())) > 1:
        topologies = sorted(class_hash)
        # Find first pair of topology classes and first frozen context that separates them.
        for i in range(len(topologies)):
            if separation_witness is not None: break
            for j in range(i + 1, len(topologies)):
                ta, tb = topologies[i], topologies[j]
                if class_hash[ta] == class_hash[tb]:
                    continue
                ra, rb = sorted(groups[ta])[0], sorted(groups[tb])[0]
                keys_a = sorted(k for k in row_by_key if k[0] == ra)
                for ka in keys_a:
                    _, ctx, ori, a, b = ka
                    kb = (rb, ctx, ori, a, b)
                    xa, xb = row_by_key[ka], row_by_key[kb]
                    ha = xa["operational_signature"]["science_sha256"]
                    hb = xb["operational_signature"]["science_sha256"]
                    if ha != hb:
                        separation_witness = {
                            "topology_A": ta,
                            "topology_B": tb,
                            "target_A_ref": ra,
                            "target_B_ref": rb,
                            "context_ref": ctx,
                            "orientation": ori,
                            "operator": [a, b],
                            "signature_A": xa["operational_signature"],
                            "signature_B": xb["operational_signature"],
                            "meaning": "Same frozen CAPS7 input class, same admitted whole-unit context and operator, but different topology-blind operational relation signatures.",
                        }
                        break
                if separation_witness is not None: break

    if within_fail:
        status = "REVIEW_REQUIRED"
        classification = "G3_S1_REVIEW_REQUIRED_NONTOPOLOGICAL_REALIZATION_SENSITIVITY_S2_LOCKED"
        outcome = "REVIEW_REQUIRED"
        s2 = False
    elif separation_witness is not None:
        status = "PASS"
        classification = "G3_TOPOLOGY_READ_EARNED_ON_REGISTERED_PAIR_CONTEXT_BASIS_S2_UNLOCKED"
        outcome = "TOPOLOGY_READ_EARNED"
        s2 = True
    else:
        status = "PASS"
        classification = "G3_TOPOLOGY_NOT_READ_ON_REGISTERED_PAIR_CONTEXT_BASIS_CAPS7_SURVIVES_S2_UNLOCKED"
        outcome = "TOPOLOGY_NOT_READ"
        s2 = True

    result = {
        "schema_id": "IG_G3_S1_PAIR_CONTEXT_READ_RESULT_V1",
        "status": status,
        "classification": classification,
        "outcome": outcome,
        "g3_started": True,
        "g3_graduated": False,
        "g3_s2_unlocked": s2,
        "topology_promoted": False,
        "authority": auth,
        "phase0_spec_sha256": phase0_spec()["science_sha256"],
        "s1_spec_sha256": spec["science_sha256"],
        "challenge_reproduction": dict(reproduction),
        "context_basis": {
            "target_count": len(refs),
            "context_partner_count": len(refs),
            "operator_count": 31,
            "orientation_count": 2,
            "task_count": len(task_rows),
            "all_targets_and_contexts_from_frozen_s0_corpus": True,
        },
        "within_topology_congruence": within,
        "target_behavior_summaries": target_summaries,
        "topology_class_behavior_hashes": class_hash if not within_fail else {},
        "separation_witness": separation_witness,
        "operational_observer": spec["operational_signature"],
        "next_authorized_stage": "G3:S2" if s2 else None,
        "nonclaims": list(spec["nonclaims"]),
    }
    result["science_sha256"] = canonical_sha256(result)
    return result
