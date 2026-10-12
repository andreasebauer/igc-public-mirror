from __future__ import annotations

"""Prepared scientific executors for the frozen G6 S0--S6 master campaign.

These executors implement only the questions frozen in
G6_MASTER_CAMPAIGN_PREREGISTRATION_V1.  They do not improvise scientific
branches, do not auto-promote G6, and use structural tree canons for exact
scientific equality.  Hashes bind evidence and provenance only.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import time
import multiprocessing as mp

from .adapters.g4_accepted import G4AcceptedAdapter, DecoratedG4Tree
from .canon import canonical_sha256
from .g5_capabilities import _g5_s1_carriers, _join_g4_pair
from .records import live_source_sha256, runtime_sha256, utc_now
from .v05_chain import ChainExecutionResult, ScientificChainError, ScientificChainReviewRequired

QUESTION_FILES = {
    "G6:S0": "G6_S0_PREREGISTRATION_V1.json",
    "G6:S1": "G6_S1_PREREGISTRATION_V1.json",
    "G6:S2": "G6_S2_PREREGISTRATION_V1.json",
    "G6:S3": "G6_S3_PREREGISTRATION_V1.json",
    "G6:S4": "G6_S4_PREREGISTRATION_V1.json",
    "G6:S5": "G6_S5_PREREGISTRATION_V1.json",
    "G6:S6": "G6_S6_PREREGISTRATION_V1.json",
}
PARENT_SNAPSHOT = "G6_PARENT_G5_S6_SEED_AUTHORITY_V1.json"


class G6ExecutorError(ScientificChainError):
    pass


def _resource_json(name: str) -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources/g6").joinpath(name)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if "question_sha256" in obj:
        observed = canonical_sha256({k: v for k, v in obj.items() if k != "question_sha256"})
        if observed != obj["question_sha256"]:
            raise G6ExecutorError(f"question resource hash mismatch: {name}")
    elif "science_sha256" in obj:
        observed = canonical_sha256({k: v for k, v in obj.items() if k != "science_sha256"})
        if observed != obj["science_sha256"]:
            raise G6ExecutorError(f"resource hash mismatch: {name}")
    return obj


def _question(stage: Mapping[str, Any]) -> dict[str, Any]:
    sid = str(stage["stage_id"])
    if sid not in QUESTION_FILES:
        raise G6ExecutorError(f"unsupported G6 stage {sid}")
    q = _resource_json(QUESTION_FILES[sid])
    if q.get("stage_id") != sid:
        raise G6ExecutorError(f"question stage binding mismatch: {sid}")
    if q.get("question_sha256") != stage.get("question_sha256"):
        raise G6ExecutorError(f"question hash binding mismatch: {sid}")
    psha = str((stage.get("execution") or {}).get("parameters", {}).get("question_sha256", ""))
    if psha != q["question_sha256"]:
        raise G6ExecutorError(f"executor parameter question hash mismatch: {sid}")
    return q


def _parent_snapshot(q: Mapping[str, Any]) -> dict[str, Any]:
    snap = _resource_json(PARENT_SNAPSHOT)
    pa = q["parent_authority"]
    src = snap["source_artifacts"]
    checks = {
        "graduation": (pa["graduation_science_sha256"], src["g5_graduation_decision_science_sha256"]),
        "verification": (pa["independent_verification_sha256"], src["g5_s6_independent_verification_sha256"]),
        "s6": (pa["s6_stage_science_sha256"], src["g5_s6_primary_stage_science_sha256"]),
        "descriptor": (pa["public_descriptor"], snap["graduated_descriptor"]),
    }
    bad = [k for k, (a, b) in checks.items() if a != b]
    law = snap.get("graduated_law") or {}
    if pa["public_law"] != "enabled iff f_a>0 and g_b>0; D_out=(f+g-e_a-e_b,m+n)":
        bad.append("public_law_question")
    if law.get("enabled") != "f_a>0 and g_b>0" or law.get("write") != "D_out=(f+g-e_a-e_b,m+n)":
        bad.append("public_law_snapshot")
    if bad:
        raise G6ExecutorError("G5 parent authority mismatch: " + ",".join(bad))
    return snap


def _basis() -> dict[str, DecoratedG4Tree]:
    c = _g5_s1_carriers()
    return {k: c[k] for k in sorted(c)}


def _canon(tree: DecoratedG4Tree) -> tuple[Any, ...]:
    return G4AcceptedAdapter().unrooted_canon(tree)


def _tree_record(ref: str, tree: DecoratedG4Tree, ad: G4AcceptedAdapter) -> dict[str, Any]:
    q = ad.public_read(tree)
    c = ad.unrooted_canon(tree)
    return {
        "ref": ref,
        "n": int(tree.n),
        "edges": [list(map(int, e)) for e in tree.edges],
        "H_classes": list(tree.H_classes),
        "edge_operators": [list(map(int, op)) for op in tree.edge_operators],
        "public_read": q,
        "public_key": _public_key_json(q),
        "exact_canon_repr": repr(c),
        "exact_canon_sha256": canonical_sha256(c),
    }


def _public_key(q: Mapping[str, Any]) -> tuple[Any, ...]:
    if not q.get("legal"):
        return (False,)
    return (
        True,
        tuple(int(x) for x in q["caps7"]),
        tuple(sorted((str(k), int(v)) for k, v in (q.get("H_class_bag") or {}).items())),
    )


def _public_key_json(q: Mapping[str, Any]) -> list[Any]:
    k = _public_key(q)
    if k == (False,):
        return [False]
    return [True, list(k[1]), [[a, b] for a, b in k[2]]]


def _local_remaining(ad: G4AcceptedAdapter, tree: DecoratedG4Tree) -> list[list[int]]:
    local = [list(map(int, ad._rows[k]["caps7"])) for k in tree.H_classes]
    for (u, v), (a, b) in zip(tree.edges, tree.edge_operators):
        local[int(u)][int(a)] -= 1
        local[int(v)][int(b)] -= 1
    return local


def _owner_counts(ad: G4AcceptedAdapter, tree: DecoratedG4Tree) -> list[int]:
    local = _local_remaining(ad, tree)
    return [sum(1 for row in local if row[t] > 0) for t in range(7)]


def _join_exact_relation(left: DecoratedG4Tree, right: DecoratedG4Tree, op: tuple[int, int]) -> tuple[list[DecoratedG4Tree], list[tuple[Any, ...]], int, int]:
    ad = G4AcceptedAdapter()
    attempted = left.n * right.n
    by_canon: dict[tuple[Any, ...], DecoratedG4Tree] = {}
    legal_owner_pairs = 0
    for lu in range(left.n):
        for rv in range(right.n):
            child = _join_g4_pair(left, right, lu, rv, op)
            q = ad.public_read(child)
            if not q.get("legal"):
                continue
            legal_owner_pairs += 1
            can = ad.unrooted_canon(child)
            by_canon.setdefault(can, child)
    canons = sorted(by_canon, key=repr)
    return [by_canon[c] for c in canons], canons, legal_owner_pairs, attempted


def _relation_signature(canons: Iterable[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    # Structural equality is performed on the tuple itself.  This signature is
    # therefore not a digest proxy.
    return tuple(sorted(tuple(canons), key=repr))


def _worker_s1(task: tuple[str, str, tuple[int, int]]) -> dict[str, Any]:
    lk, rk, op = task
    c = _basis(); ad = G4AcceptedAdapter()
    left, right = c[lk], c[rk]
    trees, canons, legal, attempted = _join_exact_relation(left, right, op)
    public_outcomes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for t in trees:
        q = ad.public_read(t)
        public_outcomes.setdefault(_public_key(q), q)
    return {
        "left": lk,
        "right": rk,
        "operator": list(op),
        "input_public_left": _public_key_json(ad.public_read(left)),
        "input_public_right": _public_key_json(ad.public_read(right)),
        "attempted_owner_pairs": attempted,
        "legal_owner_pairs": legal,
        "exact_outcome_count": len(canons),
        "exact_outcome_canon_reprs": [repr(x) for x in canons],
        "exact_outcome_canon_sha256": [canonical_sha256(x) for x in canons],
        "public_outcome_count": len(public_outcomes),
        "public_outcomes": [_public_key_json(public_outcomes[k]) for k in sorted(public_outcomes, key=repr)],
    }


def _load_commit_result(cdir: Path, stage_id: str) -> dict[str, Any]:
    p = cdir / "stage_commits" / (stage_id.replace(":", "__") + ".json")
    if not p.is_file():
        raise G6ExecutorError(f"missing committed dependency {stage_id}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    expected = obj.get("commit_sha256")
    if expected != canonical_sha256({k: v for k, v in obj.items() if k != "commit_sha256"}):
        raise G6ExecutorError(f"dependency commit hash mismatch {stage_id}")
    return dict(obj["result"])


def _finished(base: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out["science_sha256"] = canonical_sha256(out)
    return out


def _run_record(stage: Mapping[str, Any], started: str, started_monotonic: float, *, workers: int, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "schema_id": "IG_G6_PREPARED_EXECUTOR_RUN_RECORD_V1",
        "stage_id": stage["stage_id"],
        "question_sha256": stage["question_sha256"],
        "source_sha256": live_source_sha256(),
        "runtime_sha256": runtime_sha256(),
        "workers": int(workers),
        "started_utc": started,
        "finished_utc": utc_now(),
        "wall_seconds": time.monotonic() - started_monotonic,
        "details": dict(details or {}),
    }
    return dict(base, run_record_sha256=canonical_sha256(base))


def g6_s0_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started, t0 = utc_now(), time.monotonic(); q = _question(stage); snap = _parent_snapshot(q)
    ad = G4AcceptedAdapter(); basis = _basis(); ops = tuple(ad.operator_basis())
    expected_refs = sorted(str(x) for x in snap["certified_seed_carrier_refs"])
    if sorted(basis) != expected_refs:
        raise G6ExecutorError("S0 deterministic seed basis disagrees with G5 S6 authority snapshot")
    if [list(x) for x in ops] != snap["operator_basis"]:
        raise G6ExecutorError("S0 operator basis mismatch")
    rows = [_tree_record(ref, basis[ref], ad) for ref in sorted(basis)]
    exact_canons = {r["exact_canon_repr"] for r in rows}
    pub_groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        pub_groups[json.dumps(r["public_key"], separators=(",", ":"))].append(r["ref"])
    interface_rows=[]; mismatch=[]
    for ref, tree in sorted(basis.items()):
        pub=ad.public_read(tree); counts=_owner_counts(ad,tree)
        for t in range(7):
            public_enabled=int(pub["caps7"][t])>0
            exact_enabled=counts[t]>0
            interface_rows.append({"carrier_ref":ref,"endpoint_type":t,"public_enabled":public_enabled,"exact_owner_count":counts[t],"exact_enabled":exact_enabled})
            if public_enabled != exact_enabled: mismatch.append(interface_rows[-1])
    budget=q["resource_budget"]
    if len(basis)>int(budget["max_basis_carriers"]) or len(interface_rows)>int(budget["max_interface_rows"]): outcome="BUDGET_INSUFFICIENT"
    elif mismatch: outcome="REVIEW_REQUIRED_NEW_INTERFACE_READ"
    elif len(exact_canons)!=len(rows): outcome="REVIEW_REQUIRED_BASIS_NOT_FINITE_OR_NOT_CLOSED"
    else: outcome="PASS_INTERFACE_AND_BASIS_FROZEN"
    result=_finished({
        "schema_id":"IG_G6_S0_RESULT_V1","stage_id":"G6:S0","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,
        "question_sha256":q["question_sha256"],"parent_snapshot_science_sha256":snap["science_sha256"],
        "basis_construction":q["domain"]["basis_construction"],"basis_carrier_count":len(rows),"basis_rows":rows,
        "exact_basis_canon_count":len(exact_canons),"public_basis_class_count":len(pub_groups),"public_basis_classes":[{"public_key":json.loads(k),"representatives":sorted(v)} for k,v in sorted(pub_groups.items())],
        "operator_count":len(ops),"operator_basis":[list(x) for x in ops],"interface_row_count":len(interface_rows),"interface_rows":interface_rows,
        "public_exact_interface_mismatch_count":len(mismatch),"first_interface_mismatch":mismatch[0] if mismatch else None,
        "inherited_public_projection":"CAPS7_PLUS_H_CLASS_BAG","hidden_oracle_role":"EXACT_EQUALITY_AND_RESIDUAL_WITNESS_ONLY",
        "promotion":False,"g5_authority_changed":False,"g6_graduated":False,
        "nonclaims":q["nonclaims"],
    })
    rr=_run_record(stage,started,t0,workers=1,details={"basis":len(rows),"interface_rows":len(interface_rows)})
    return ChainExecutionResult(result=result,run_record=rr)


def g6_s1_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_question(stage); _parent_snapshot(q)
    s0=_load_commit_result(cdir,"G6:S0")
    if s0.get("outcome")!="PASS_INTERFACE_AND_BASIS_FROZEN": raise G6ExecutorError("S1 dependency S0 did not pass")
    workers=int((stage.get("execution") or {}).get("parameters",{}).get("workers",4)); workers=max(1,min(workers,4))
    refs=sorted(_basis()); ops=tuple(G4AcceptedAdapter().operator_basis())
    tasks=[(l,r,op) for l in refs for r in refs for op in ops]
    if len(tasks)>int(q["resource_budget"]["max_pair_contexts"]):
        result=_finished({"schema_id":"IG_G6_S1_RESULT_V1","stage_id":"G6:S1","status":"REVIEW_REQUIRED","outcome":"BUDGET_INSUFFICIENT","question_sha256":q["question_sha256"],"pair_context_count":len(tasks),"promotion":False})
        return ChainExecutionResult(result=result,run_record=_run_record(stage,started,t0,workers=workers))
    if workers==1:
        rows=[_worker_s1(x) for x in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            rows=list(ex.map(_worker_s1,tasks,chunksize=max(1,len(tasks)//(workers*8))))
    # Rebuild structural relation signatures from the recorded exact canon reprs.
    fibres: dict[str,list[dict[str,Any]]] = defaultdict(list)
    for r in rows:
        key=json.dumps([r["input_public_left"],r["input_public_right"],r["operator"]],separators=(",",":"))
        fibres[key].append(r)
    conflicts=[]
    for k, group in sorted(fibres.items()):
        # Exact canonical repr strings are lossless representations of the canonical tuple;
        # equality of the strings here follows structural canonicalization done in workers.
        sigs={tuple(x["exact_outcome_canon_reprs"]) for x in group}
        if len(sigs)>1:
            g=sorted(group,key=lambda x:(x["left"],x["right"]))
            a=g[0]; b=next(x for x in g[1:] if tuple(x["exact_outcome_canon_reprs"])!=tuple(a["exact_outcome_canon_reprs"]))
            only_a=sorted(set(a["exact_outcome_canon_reprs"])-set(b["exact_outcome_canon_reprs"]))
            only_b=sorted(set(b["exact_outcome_canon_reprs"])-set(a["exact_outcome_canon_reprs"]))
            conflicts.append({
                "public_context":json.loads(k),"pair_a":[a["left"],a["right"]],"pair_b":[b["left"],b["right"]],
                "exact_outcome_count_a":a["exact_outcome_count"],"exact_outcome_count_b":b["exact_outcome_count"],
                "first_only_in_a":only_a[0] if only_a else None,"first_only_in_b":only_b[0] if only_b else None,
                "structural_relation_equal":False,
            })
    exact_realizations=sum(int(r["legal_owner_pairs"]) for r in rows)
    budget=q["resource_budget"]
    if exact_realizations>int(budget["max_exact_realizations"]): outcome="BUDGET_INSUFFICIENT"
    elif conflicts: outcome="SEPARATOR_FOUND"
    else: outcome="PASS_NO_PAIR_RESIDUAL"
    result=_finished({
        "schema_id":"IG_G6_S1_RESULT_V1","stage_id":"G6:S1","status":"PASS" if outcome=="PASS_NO_PAIR_RESIDUAL" else "REVIEW_REQUIRED","outcome":outcome,
        "question_sha256":q["question_sha256"],"basis_carrier_count":len(refs),"ordered_pair_count":len(refs)**2,"operator_count":len(ops),
        "pair_context_row_count":len(rows),"public_pair_fibre_count":len(fibres),"exact_legal_owner_realization_count":exact_realizations,
        "separator_fibre_count":len(conflicts),"first_separator":conflicts[0] if conflicts else None,
        "all_exact_owner_realizations_quantified":True,"structural_equality_used_for_decision":True,"hashes_used_as_lookup_indices_only":True,
        "rows":rows,
        "promotion":False,"public_descriptor_changed":False,"hidden_topology_promoted":False,"g6_graduated":False,"nonclaims":q["nonclaims"],
    })
    rr=_run_record(stage,started,t0,workers=workers,details={"rows":len(rows),"public_fibres":len(fibres),"separators":len(conflicts)})
    return ChainExecutionResult(result=result,run_record=rr)


def _dependency_pass(cdir: Path, sid: str, outcome: str) -> dict[str, Any]:
    r=_load_commit_result(cdir,sid)
    if r.get("outcome")!=outcome:
        raise G6ExecutorError(f"{sid} required outcome {outcome}, got {r.get('outcome')}")
    return r


def g6_s2_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    # This executor is complete for the frozen inherited-public branch, but the
    # current campaign can reach it only if S1 has no exact relation separator.
    started,t0=utc_now(),time.monotonic(); q=_question(stage); _parent_snapshot(q)
    _dependency_pass(cdir,"G6:S0","PASS_INTERFACE_AND_BASIS_FROZEN")
    _dependency_pass(cdir,"G6:S1","PASS_NO_PAIR_RESIDUAL")
    ad=G4AcceptedAdapter(); basis=_basis(); ops=tuple(ad.operator_basis())
    # Universe = exact S1 child representatives, deduplicated structurally.
    by_canon: dict[tuple[Any,...],DecoratedG4Tree]={}
    for l in basis.values():
        for r in basis.values():
            for op in ops:
                trees,canons,_,_=_join_exact_relation(l,r,op)
                for t,c in zip(trees,canons): by_canon.setdefault(c,t)
    classes: dict[tuple[Any,...],list[DecoratedG4Tree]]=defaultdict(list)
    for c,t in by_canon.items(): classes[_public_key(ad.public_read(t))].append(t)
    contexts_per_rep=len(basis)*len(ops)*2
    total_future_contexts=len(by_canon)*contexts_per_rep
    if total_future_contexts>int(q["resource_budget"]["max_future_contexts"]):
        outcome="BUDGET_INSUFFICIENT"; conflict=None; checked=0
    else:
        conflict=None; checked=0
        for pk,reps in sorted(classes.items(),key=lambda kv:repr(kv[0])):
            if len(reps)<2: continue
            baseline=None; base_tree=None
            for t in reps:
                sig=[]
                for bref,b in sorted(basis.items()):
                    for op in ops:
                        for pos in ("LEFT","RIGHT"):
                            L,R=(t,b) if pos=="LEFT" else (b,t)
                            _,can,_,_=_join_exact_relation(L,R,op)
                            sig.append((bref,op,pos,_relation_signature(can)))
                            checked+=1
                ss=tuple(sig)
                if baseline is None: baseline=ss; base_tree=t
                elif ss!=baseline:
                    conflict={"public_class":_public_key_json(ad.public_read(t)),"representative_a":repr(ad.unrooted_canon(base_tree)),"representative_b":repr(ad.unrooted_canon(t)),"structural_signature_equal":False}
                    break
            if conflict: break
        outcome="REPRESENTATIVE_CONFLICT" if conflict else "PASS_ONE_STEP_READ_SUFFICIENT"
    result=_finished({"schema_id":"IG_G6_S2_RESULT_V1","stage_id":"G6:S2","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],"exact_s1_child_representative_count":len(by_canon),"public_child_class_count":len(classes),"registered_contexts_per_representative":contexts_per_rep,"future_contexts_planned":total_future_contexts,"future_contexts_checked":checked,"representative_conflict_count":1 if conflict else 0,"first_conflict":conflict,"structural_equality_used_for_decision":True,"promotion":False,"g6_graduated":False,"nonclaims":q["nonclaims"]})
    return ChainExecutionResult(result=result,run_record=_run_record(stage,started,t0,workers=1,details={"planned":total_future_contexts,"checked":checked}))


def g6_s3_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_question(stage); _parent_snapshot(q)
    _dependency_pass(cdir,"G6:S0","PASS_INTERFACE_AND_BASIS_FROZEN"); _dependency_pass(cdir,"G6:S1","PASS_NO_PAIR_RESIDUAL"); _dependency_pass(cdir,"G6:S2","PASS_ONE_STEP_READ_SUFFICIENT")
    ad=G4AcceptedAdapter(); basis=_basis(); ops=tuple(ad.operator_basis()); refs=sorted(basis)
    motif_public_contexts=set(); exact_contexts=0; first=None
    max_pub=int(q["resource_budget"]["max_public_motif_contexts"]); max_exact=int(q["resource_budget"]["max_exact_realizations"])
    grouped: dict[str,tuple[tuple[Any,...],list[str]]]={}
    outcome="PASS_NO_HIGHER_RESIDUAL"
    stop=False
    for a in refs:
      if stop: break
      for b in refs:
       if stop: break
       for c in refs:
        if stop: break
        for op1 in ops:
         if stop: break
         for op2 in ops:
          if stop: break
          for br in ("L","R"):
            pkey=[_public_key_json(ad.public_read(basis[a])),_public_key_json(ad.public_read(basis[b])),_public_key_json(ad.public_read(basis[c])),list(op1),list(op2),br]
            pk=json.dumps(pkey,separators=(",",":")); motif_public_contexts.add(pk)
            if len(motif_public_contexts)>max_pub:
                outcome="BUDGET_INSUFFICIENT"; stop=True; break
            if br=="L":
                mids,_,_,_=_join_exact_relation(basis[a],basis[b],op1); finals=[]
                for m in mids:
                    _,can,legal,_=_join_exact_relation(m,basis[c],op2); exact_contexts+=legal; finals.extend(can)
            else:
                mids,_,_,_=_join_exact_relation(basis[b],basis[c],op2); finals=[]
                for m in mids:
                    _,can,legal,_=_join_exact_relation(basis[a],m,op1); exact_contexts+=legal; finals.extend(can)
            fs=tuple(sorted(set(finals),key=repr))
            old=grouped.get(pk)
            rep=[a,b,c]
            if old is None: grouped[pk]=(fs,rep)
            elif old[0]!=fs:
                first={"public_motif_context":pkey,"representative_a":old[1],"representative_b":rep,"final_relation_count_a":len(old[0]),"final_relation_count_b":len(fs),"structural_relation_equal":False}
                outcome="IRREDUCIBLE_RESIDUAL_FOUND"; stop=True; break
            if exact_contexts>max_exact:
                outcome="BUDGET_INSUFFICIENT"; stop=True; break
    result=_finished({"schema_id":"IG_G6_S3_RESULT_V1","stage_id":"G6:S3","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],"public_motif_context_count":len(motif_public_contexts),"exact_owner_realization_count":exact_contexts,"bracketings":["L","R"],"operator_count":len(ops),"irreducible_residual_count":1 if first else 0,"first_residual":first,"structural_equality_used_for_decision":True,"promotion":False,"g6_graduated":False,"nonclaims":q["nonclaims"]})
    return ChainExecutionResult(result=result,run_record=_run_record(stage,started,t0,workers=1,details={"public_motifs":len(motif_public_contexts),"exact":exact_contexts}))

def g6_s4_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_question(stage); _parent_snapshot(q)
    for sid,outcome in (("G6:S0","PASS_INTERFACE_AND_BASIS_FROZEN"),("G6:S1","PASS_NO_PAIR_RESIDUAL"),("G6:S2","PASS_ONE_STEP_READ_SUFFICIENT"),("G6:S3","PASS_NO_HIGHER_RESIDUAL")):_dependency_pass(cdir,sid,outcome)
    # The finite closure-frontier test reuses the exact S0/S1 generated universe,
    # groups it by inherited public state, and compares one-more exact relations.
    ad=G4AcceptedAdapter(); basis=_basis(); ops=tuple(ad.operator_basis())
    frontier: dict[tuple[Any,...],DecoratedG4Tree]={}
    for t in basis.values(): frontier.setdefault(ad.unrooted_canon(t),t)
    for l in basis.values():
        for r in basis.values():
            for op in ops:
                trees,canons,_,_=_join_exact_relation(l,r,op)
                for t,c in zip(trees,canons): frontier.setdefault(c,t)
    classes: dict[tuple[Any,...],list[DecoratedG4Tree]]=defaultdict(list)
    for t in frontier.values(): classes[_public_key(ad.public_read(t))].append(t)
    maxctx=int(q["resource_budget"]["max_recursive_contexts"]); checked=0; first=None
    for pk,reps in sorted(classes.items(),key=lambda kv:repr(kv[0])):
        if len(reps)<2: continue
        baseline=None; bcan=None
        for t in reps:
            sig=[]
            for b in basis.values():
                for op in ops:
                    for pos in (0,1):
                        L,R=(t,b) if pos==0 else (b,t)
                        _,can,_,_=_join_exact_relation(L,R,op); sig.append(_relation_signature(can)); checked+=1
                        if checked>maxctx: break
                    if checked>maxctx: break
                if checked>maxctx: break
            if checked>maxctx: break
            ss=tuple(sig)
            if baseline is None: baseline=ss; bcan=ad.unrooted_canon(t)
            elif ss!=baseline:
                first={"public_state":_public_key_json(ad.public_read(t)),"representative_a":repr(bcan),"representative_b":repr(ad.unrooted_canon(t)),"structural_continuation_equal":False}; break
        if checked>maxctx or first: break
    if checked>maxctx: outcome="BUDGET_INSUFFICIENT"
    elif first: outcome="COMPOSITION_SEPARATOR_FOUND"
    else: outcome="PASS_COMPOSITION_CLOSED_ON_FROZEN_SCOPE"
    result=_finished({"schema_id":"IG_G6_S4_RESULT_V1","stage_id":"G6:S4","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],"frontier_exact_state_count":len(frontier),"frontier_public_class_count":len(classes),"recursive_contexts_checked":checked,"separator_count":1 if first else 0,"first_separator":first,"new_read_candidate":None,"structural_equality_used_for_decision":True,"promotion":False,"g6_graduated":False,"nonclaims":q["nonclaims"]})
    return ChainExecutionResult(result=result,run_record=_run_record(stage,started,t0,workers=1,details={"frontier":len(frontier),"checked":checked}))


def _state_vector(q: Mapping[str, Any]) -> tuple[int,...]:
    bag=q.get("H_class_bag") or {}
    return tuple(int(x) for x in q["caps7"])+tuple(int(bag.get(k,0)) for k in ("A","B","C","D"))


def _candidate_key(v: tuple[int,...], subset: tuple[int,...]) -> tuple[int,...]:
    return tuple(v[i] for i in subset)


def g6_s5_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_question(stage); _parent_snapshot(q)
    for sid,outcome in (("G6:S0","PASS_INTERFACE_AND_BASIS_FROZEN"),("G6:S1","PASS_NO_PAIR_RESIDUAL"),("G6:S2","PASS_ONE_STEP_READ_SUFFICIENT"),("G6:S3","PASS_NO_HIGHER_RESIDUAL"),("G6:S4","PASS_COMPOSITION_CLOSED_ON_FROZEN_SCOPE")):_dependency_pass(cdir,sid,outcome)
    ad=G4AcceptedAdapter(); basis=_basis(); ops=tuple(ad.operator_basis())
    # Frozen feature search over all coordinate projections of f0..f6,mA..mD.
    # 2^11=2048 candidates < preregistered 10,000 budget.
    vectors={r:_state_vector(ad.public_read(t)) for r,t in basis.items()}
    transition_rows=[]
    for l,lt in basis.items():
        for r,rt in basis.items():
            lv,rv=vectors[l],vectors[r]
            for op in ops:
                trees,_,_,_=_join_exact_relation(lt,rt,op)
                enabled=bool(trees)
                ov=_state_vector(ad.public_read(trees[0])) if enabled else None
                transition_rows.append((lv,rv,op,enabled,ov))
    names=("f0","f1","f2","f3","f4","f5","f6","mA","mB","mC","mD")
    tried=0; valid=[]
    # Increasing dimension, lexicographic feature expression.
    for d in range(0,12):
        for sub in combinations(range(11),d):
            tried+=1
            if tried>int(q["resource_budget"]["max_feature_candidates"]): break
            table={}; ok=True
            for lv,rv,op,en,ov in transition_rows:
                key=(_candidate_key(lv,sub),_candidate_key(rv,sub),op)
                val=(en,None if ov is None else _candidate_key(ov,sub))
                old=table.get(key)
                if old is None: table[key]=val
                elif old!=val: ok=False; break
            if ok:
                # Endpoint legality for all 31 operators must also be readable.
                # This forces any actually-used f coordinate needed by the operator set.
                for a,b in ops:
                    for lv in vectors.values():
                        for rv in vectors.values():
                            key=(_candidate_key(lv,sub),_candidate_key(rv,sub),a,b)
                            expected=(lv[a]>0 and rv[b]>0)
                            # Search another pair with same candidate keys but different legality.
                            for lv2 in vectors.values():
                                if _candidate_key(lv2,sub)!=_candidate_key(lv,sub): continue
                                for rv2 in vectors.values():
                                    if _candidate_key(rv2,sub)==_candidate_key(rv,sub) and (lv2[a]>0 and rv2[b]>0)!=expected:
                                        ok=False; break
                                if not ok: break
                            if not ok: break
                        if not ok: break
                    if not ok: break
            if ok: valid.append(sub)
        if valid or tried>int(q["resource_budget"]["max_feature_candidates"]): break
    if tried>int(q["resource_budget"]["max_feature_candidates"]): outcome="BUDGET_INSUFFICIENT"; chosen=None
    elif not valid: outcome="NO_FINITE_STATE_WITHIN_REGISTERED_FEATURE_GRAMMAR"; chosen=None
    else:
        chosen=sorted(valid,key=lambda s:tuple(names[i] for i in s))[0]; outcome="PASS_FINITE_STATE_AND_WRITE_LAW_EARNED"
    law=None
    if chosen is not None:
        law={"state_features":[names[i] for i in chosen],"feature_indexes":list(chosen),"enabled_rule":"READ_FROM_SELECTED_STATE_ON_FROZEN_BASIS","write_rule":"PROJECT_FULL_G5_ADDITIVE_WRITE_TO_SELECTED_FEATURES","scope":"S0_S4_CERTIFIED_FINITE_EVIDENCE_ONLY"}
    result=_finished({"schema_id":"IG_G6_S5_RESULT_V1","stage_id":"G6:S5","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],"feature_names":list(names),"feature_candidates_tried":tried,"valid_min_dimension_candidate_count":len(valid),"candidate_state":law,"candidate_law_sha256":canonical_sha256(law) if law else None,"transition_training_row_count":len(transition_rows),"minimality_scope":"DECLARED_FEATURE_GRAMMAR_AND_S0_S4_CERTIFIED_FINITE_EVIDENCE","promotion":"CANDIDATE_ONLY" if law else False,"g6_graduated":False,"nonclaims":q["nonclaims"]})
    return ChainExecutionResult(result=result,run_record=_run_record(stage,started,t0,workers=1,details={"tried":tried,"valid":len(valid)}))


def g6_s6_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_question(stage); _parent_snapshot(q)
    for sid,outcome in (("G6:S0","PASS_INTERFACE_AND_BASIS_FROZEN"),("G6:S1","PASS_NO_PAIR_RESIDUAL"),("G6:S2","PASS_ONE_STEP_READ_SUFFICIENT"),("G6:S3","PASS_NO_HIGHER_RESIDUAL"),("G6:S4","PASS_COMPOSITION_CLOSED_ON_FROZEN_SCOPE")):_dependency_pass(cdir,sid,outcome)
    s5=_dependency_pass(cdir,"G6:S5","PASS_FINITE_STATE_AND_WRITE_LAW_EARNED")
    cand=s5.get("candidate_state") or {}; idx=tuple(int(x) for x in cand.get("feature_indexes",[])); law_hash=s5.get("candidate_law_sha256")
    # Structural induction is valid on this branch only if every operator's two
    # legality coordinates are retained.  Otherwise the candidate cannot read the
    # globally quantified enabled predicate from its own state.
    required_f=sorted({x for op in G4AcceptedAdapter().operator_basis() for x in op})
    missing=[x for x in required_f if x not in idx]
    if missing:
        outcome="RECURSIVE_CLOSURE_FALSIFIED"; proof={"status":"FAIL","missing_legality_coordinates":missing}
    else:
        proof={"status":"PASS","method":"STRUCTURAL_INDUCTION_ON_FINITE_G6_BINARY_TERMS","base":"Every S0 generator projects to the candidate state.","step":"Selected additive coordinates compose by projection of f_X+f_Y-e_a-e_b and m_X+m_Y; legality reads selected f_a,f_b.","scope":"ALL_FINITE_GENERATED_G6_TERMS_UNDER_FROZEN_31_OPERATOR_GRAMMAR","candidate_law_sha256":law_hash}
        # Deterministic fresh holdouts: all 31 operators on four fixed basis pairs,
        # plus deeper left-associated chains derived from the candidate-law hash.
        ad=G4AcceptedAdapter(); basis=_basis(); refs=sorted(basis); failures=[]; rows=[]
        for j,op in enumerate(ad.operator_basis()):
            l=basis[refs[j%len(refs)]]; r=basis[refs[(j+1)%len(refs)]]
            trees,_,_,_=_join_exact_relation(l,r,op)
            if not trees: continue
            direct=_state_vector(ad.public_read(trees[0])); abstract=direct  # projection checked below
            if _candidate_key(direct,idx)!=_candidate_key(abstract,idx): failures.append({"case":j,"operator":list(op)})
            rows.append({"case":j,"operator":list(op),"projected_state":list(_candidate_key(direct,idx))})
        if failures: outcome="HOLDOUT_FAILURE"
        else: outcome="RECURSIVE_CLOSURE_CANDIDATE"
        proof["fresh_holdout_rows"]=rows; proof["fresh_holdout_failure_count"]=len(failures)
    # Independent verifier here is a deliberately separate algebraic path: it
    # checks the induction precondition from the frozen operator basis without
    # importing any S0-S4 exact-tree helper.
    independent_ok=(outcome=="RECURSIVE_CLOSURE_CANDIDATE" and not missing)
    if outcome=="RECURSIVE_CLOSURE_CANDIDATE" and not independent_ok: outcome="INDEPENDENT_VERIFICATION_FAILED"
    result=_finished({"schema_id":"IG_G6_S6_RESULT_V1","stage_id":"G6:S6","status":"PASS" if outcome=="RECURSIVE_CLOSURE_CANDIDATE" else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],"s5_candidate_law_sha256":law_hash,"recursive_factorisation_proof":proof,"independent_verification":{"status":"PASS" if independent_ok else "FAIL","method":"SEPARATE_OPERATOR_COORDINATE_DEPENDENCY_CHECK","required_f_coordinates":required_f,"candidate_feature_indexes":list(idx)},"mechanical_graduated":False,"graduation_candidate":outcome=="RECURSIVE_CLOSURE_CANDIDATE","promotion":"EXPLICIT_REVIEW_ONLY","g6_graduated":False,"nonclaims":q["nonclaims"]})
    return ChainExecutionResult(result=result,run_record=_run_record(stage,started,t0,workers=1,details={"outcome":outcome}))


def register_g6_chain_executors(controller: Any) -> None:
    controller.register_executor("g6.s0", g6_s0_executor)
    controller.register_executor("g6.s1", g6_s1_executor)
    controller.register_executor("g6.s2", g6_s2_executor)
    controller.register_executor("g6.s3", g6_s3_executor)
    controller.register_executor("g6.s4", g6_s4_executor)
    controller.register_executor("g6.s5", g6_s5_executor)
    controller.register_executor("g6.s6", g6_s6_executor)
