from __future__ import annotations

"""Controlled G6:S1 descriptor/read repair ladder.

Scientific discipline:
- Authority is the frozen official G6:S0/G6:S1 commits.
- Candidate ladder is frozen before execution.
- No public promotion occurs here.
- Structural equality decides science; digests are provenance/indexing only.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Mapping
import json, multiprocessing as mp, time

from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from .canon import canonical_sha256
from .g5_capabilities import _g5_s1_carriers, _join_g4_pair
from .g6_stage_executors import _join_exact_relation, _local_remaining, _public_key, _public_key_json
from .records import live_source_sha256, runtime_sha256, utc_now
from .uplift_g5_r3 import independent_rooted_canon, independent_unrooted_canon
from .v05_chain import ChainExecutionResult, ScientificChainError

QUESTION_FILE = "G6_S1_REPAIR_PREREGISTRATION_V1.json"

class G6S1RepairError(ScientificChainError):
    pass


def _resource_question(stage: Mapping[str, Any]) -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources/g6").joinpath(QUESTION_FILE)
    q = json.loads(p.read_text(encoding="utf-8"))
    observed = canonical_sha256({k:v for k,v in q.items() if k != "question_sha256"})
    if observed != q.get("question_sha256"):
        raise G6S1RepairError("G6:S1R question resource hash mismatch")
    if stage.get("stage_id") != "G6:S1R" or q.get("stage_id") != "G6:S1R":
        raise G6S1RepairError("G6:S1R stage binding mismatch")
    if stage.get("question_sha256") != q["question_sha256"]:
        raise G6S1RepairError("G6:S1R question hash mismatch")
    params=(stage.get("execution") or {}).get("parameters",{})
    psha = str(params.get("question_sha256", ""))
    if psha != q["question_sha256"]:
        raise G6S1RepairError("G6:S1R executor question hash mismatch")
    declared_source=str(params.get("working_source_sha256", ""))
    observed_source=live_source_sha256()
    if declared_source != observed_source:
        raise G6S1RepairError(f"G6:S1R source binding mismatch: declared {declared_source}, observed {observed_source}")
    return q


def _load_authority_commit(q: Mapping[str, Any], stage: Mapping[str, Any], sid: str) -> dict[str, Any]:
    # Exact official S0/S1 commits are embedded byte-for-byte as portable authority
    # resources.  This avoids scientific dependence on an absolute filesystem path.
    name=sid.replace(":","__")+".json"
    p=files("infinity_grid").joinpath("resources/g6/authority").joinpath(name)
    d=json.loads(p.read_text(encoding="utf-8"))
    expected=d.get("commit_sha256")
    observed=canonical_sha256({k:v for k,v in d.items() if k!="commit_sha256"})
    if expected != observed:
        raise G6S1RepairError(f"authority commit integrity failure {sid}")
    qa=q["authority"][sid]
    checks={
      "commit_sha256": d.get("commit_sha256"),
      "result_sha256": d.get("result_sha256"),
      "question_sha256": d.get("question_sha256"),
      "science_sha256": (d.get("result") or {}).get("science_sha256"),
      "outcome": d.get("outcome"),
    }
    bad=[k for k,v in checks.items() if v != qa.get(k)]
    if bad:
        raise G6S1RepairError(f"authority mismatch {sid}: {','.join(bad)}")
    return d


def _basis() -> dict[str, DecoratedG4Tree]:
    c=_g5_s1_carriers(); return {k:c[k] for k in sorted(c)}


def _tree_to_record(t: DecoratedG4Tree) -> dict[str, Any]:
    return {"n":int(t.n),"edges":[list(map(int,e)) for e in t.edges],"H_classes":list(t.H_classes),"edge_operators":[list(map(int,o)) for o in t.edge_operators]}


def _tree_from_record(r: Mapping[str, Any]) -> DecoratedG4Tree:
    return DecoratedG4Tree(int(r["n"]),tuple(tuple(map(int,e)) for e in r["edges"]),tuple(map(str,r["H_classes"])),tuple(tuple(map(int,o)) for o in r["edge_operators"]))


def _universe_from_s1() -> dict[tuple[Any,...], DecoratedG4Tree]:
    ad=G4AcceptedAdapter(); basis=_basis(); ops=tuple(ad.operator_basis()); out={}
    for l in basis.values():
      for r in basis.values():
        for op in ops:
          trees,canons,_,_=_join_exact_relation(l,r,op)
          for t,c in zip(trees,canons): out.setdefault(c,t)
    return out


def _rho01(t: DecoratedG4Tree) -> int:
    probe=_basis()["D2_PATH"]
    _,can,_,_=_join_exact_relation(t,probe,(0,1))
    return len(can)


def _typed_owner_orbits(t: DecoratedG4Tree) -> tuple[int,...]:
    ad=G4AcceptedAdapter(); local=_local_remaining(ad,t); vals=[]
    for typ in range(7):
        can={ad.rooted_canon(t,u) for u,row in enumerate(local) if row[typ]>0}
        vals.append(len(can))
    return tuple(vals)


def _rooted_owner_response_bag(t: DecoratedG4Tree) -> tuple[Any,...]:
    ad=G4AcceptedAdapter(); local=_local_remaining(ad,t); out=[]
    for typ in range(7):
        cnt=Counter(repr(ad.rooted_canon(t,u)) for u,row in enumerate(local) if row[typ]>0)
        out.append(tuple(sorted(cnt.items())))
    return tuple(out)


def _read_candidate(t: DecoratedG4Tree, cid: str) -> Any:
    if cid=="C1_RHO01": return _rho01(t)
    if cid=="C2_TYPED_OWNER_ORBITS": return _typed_owner_orbits(t)
    if cid=="C3_ROOTED_OWNER_RESPONSE_BAG": return _rooted_owner_response_bag(t)
    raise G6S1RepairError(f"unknown candidate {cid}")


def _public_plus_read_key(t: DecoratedG4Tree, cid: str) -> tuple[Any,...]:
    ad=G4AcceptedAdapter(); return (_public_key(ad.public_read(t)), _read_candidate(t,cid))


def _basis_replay(s1_result: Mapping[str,Any], cid: str) -> dict[str,Any]:
    basis=_basis(); fibres=defaultdict(list)
    for r in s1_result["rows"]:
        lk,rk=str(r["left"]),str(r["right"]); op=tuple(map(int,r["operator"]))
        key=(_public_plus_read_key(basis[lk],cid),_public_plus_read_key(basis[rk],cid),op)
        fibres[key].append(r)
    conflicts=[]
    for k,g in fibres.items():
        sigs={tuple(x["exact_outcome_canon_reprs"]) for x in g}
        if len(sigs)>1:
            gg=sorted(g,key=lambda x:(x["left"],x["right"]))
            a=gg[0]; b=next(x for x in gg[1:] if tuple(x["exact_outcome_canon_reprs"])!=tuple(a["exact_outcome_canon_reprs"]))
            conflicts.append({"pair_a":[a["left"],a["right"]],"pair_b":[b["left"],b["right"]],"operator":list(op)})
            break
    return {"refined_pair_fibre_count":len(fibres),"separator_count":len(conflicts),"first_separator":conflicts[0] if conflicts else None}


def _continuation_relation_sig(t: DecoratedG4Tree, b: DecoratedG4Tree, op: tuple[int,int], pos: str) -> tuple[Any,...]:
    L,R=(t,b) if pos=="LEFT" else (b,t)
    _,can,_,_=_join_exact_relation(L,R,op)
    return tuple(can)


def _compare_group_task(task: tuple[str,list[dict[str,Any]]]) -> dict[str,Any]:
    cid, records = task
    ad=G4AcceptedAdapter(); basis=_basis(); ops=tuple(ad.operator_basis())
    trees=[_tree_from_record(r) for r in records]
    if len(trees)<2: return {"conflict":None,"contexts_checked":0}
    base=trees[0]; checked=0
    for other in trees[1:]:
      for bref,b in sorted(basis.items()):
        for op in ops:
          for pos in ("LEFT","RIGHT"):
            sa=_continuation_relation_sig(base,b,op,pos); sb=_continuation_relation_sig(other,b,op,pos); checked+=2
            if sa!=sb:
                return {"conflict":{"candidate":cid,"public_key":_public_key_json(ad.public_read(base)),"read_value_repr":repr(_read_candidate(base,cid)),"representative_a":repr(ad.unrooted_canon(base)),"representative_b":repr(ad.unrooted_canon(other)),"probe_ref":bref,"operator":list(op),"position":pos,"outcome_count_a":len(sa),"outcome_count_b":len(sb)},"contexts_checked":checked}
    return {"conflict":None,"contexts_checked":checked}


def _candidate_expanded_test(universe: Mapping[tuple[Any,...],DecoratedG4Tree], cid: str, workers:int, max_contexts:int) -> dict[str,Any]:
    ad=G4AcceptedAdapter(); groups=defaultdict(list); index=[]
    for can,t in sorted(universe.items(),key=lambda kv:repr(kv[0])):
        rv=_read_candidate(t,cid); key=(_public_key(ad.public_read(t)),rv)
        rec=_tree_to_record(t); groups[key].append(rec)
        index.append({"exact_canon_sha256":canonical_sha256(can),"public_key":_public_key_json(ad.public_read(t)),"read_value_repr":repr(rv)})
    multi=[(cid,v) for _,v in sorted(groups.items(),key=lambda kv:repr(kv[0])) if len(v)>1]
    planned=sum((len(v)-1)*len(_basis())*len(G4AcceptedAdapter().operator_basis())*2*2 for _,v in multi)
    if planned>max_contexts:
        return {"status":"BUDGET_INSUFFICIENT","universe_count":len(universe),"refined_class_count":len(groups),"multi_class_count":len(multi),"continuation_contexts_planned":planned,"continuation_contexts_checked":0,"first_conflict":None,"index_sha256":canonical_sha256(index)}
    checked=0; conflict=None
    if workers<=1:
        results=map(_compare_group_task,multi)
    else:
        ex=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"))
        results=ex.map(_compare_group_task,multi,chunksize=1)
    try:
        for r in results:
            checked+=int(r["contexts_checked"])
            if r["conflict"] is not None:
                conflict=r["conflict"]; break
    finally:
        if workers>1: ex.shutdown(cancel_futures=True)
    return {"status":"CONFLICT" if conflict else "PASS","universe_count":len(universe),"refined_class_count":len(groups),"multi_class_count":len(multi),"continuation_contexts_planned":planned,"continuation_contexts_checked":checked,"first_conflict":conflict,"index_sha256":canonical_sha256(index)}


def _manual_join(left:DecoratedG4Tree,right:DecoratedG4Tree,lu:int,rv:int,op:tuple[int,int]) -> DecoratedG4Tree:
    off=left.n
    return DecoratedG4Tree(left.n+right.n, tuple(left.edges)+tuple((a+off,b+off) for a,b in right.edges)+((lu,rv+off),), tuple(left.H_classes)+tuple(right.H_classes), tuple(left.edge_operators)+tuple(right.edge_operators)+(op,))


def _independent_relation_count(left:DecoratedG4Tree,right:DecoratedG4Tree,op:tuple[int,int]) -> int:
    ad=G4AcceptedAdapter(); localL=_local_remaining(ad,left); localR=_local_remaining(ad,right); can=set()
    vkeysL=[ad._vertex_key(x) for x in left.H_classes]; vkeysR=[ad._vertex_key(x) for x in right.H_classes]
    for lu in range(left.n):
      if localL[lu][op[0]]<=0: continue
      for rv in range(right.n):
        if localR[rv][op[1]]<=0: continue
        ch=_manual_join(left,right,lu,rv,op)
        vc=vkeysL+vkeysR
        cc=independent_unrooted_canon(ch.n,ch.edges,vc,ch.edge_operators)
        can.add(cc)
    return len(can)


def _independent_read(t:DecoratedG4Tree,cid:str) -> Any:
    ad=G4AcceptedAdapter(); local=_local_remaining(ad,t)
    if cid=="C1_RHO01": return _independent_relation_count(t,_basis()["D2_PATH"],(0,1))
    if cid=="C2_TYPED_OWNER_ORBITS":
        vk=[ad._vertex_key(x) for x in t.H_classes]; vals=[]
        for typ in range(7):
            can={independent_rooted_canon(t.n,t.edges,vk,t.edge_operators,u) for u,row in enumerate(local) if row[typ]>0}
            vals.append(len(can))
        return tuple(vals)
    if cid=="C3_ROOTED_OWNER_RESPONSE_BAG":
        vk=[ad._vertex_key(x) for x in t.H_classes]; out=[]
        for typ in range(7):
            cnt=Counter(repr(independent_rooted_canon(t.n,t.edges,vk,t.edge_operators,u)) for u,row in enumerate(local) if row[typ]>0)
            out.append(tuple(sorted(cnt.items())))
        return tuple(out)
    raise G6S1RepairError(cid)


def _independent_verify(universe:Mapping[tuple[Any,...],DecoratedG4Tree],cid:str,sample_n:int) -> dict[str,Any]:
    ordered=sorted(universe.items(), key=lambda kv:canonical_sha256(kv[0]))
    sample=ordered[:min(sample_n,len(ordered))]; fail=[]
    for can,t in sample:
        a=_read_candidate(t,cid); b=_independent_read(t,cid)
        if a!=b:
            fail.append({"exact_canon_sha256":canonical_sha256(can),"primary_read_repr":repr(a),"independent_read_repr":repr(b)}); break
    return {"status":"PASS" if not fail else "FAIL","candidate":cid,"sample_count":len(sample),"first_failure":fail[0] if fail else None}


def _run_record(stage:Mapping[str,Any],started:str,t0:float,workers:int,details:Mapping[str,Any]) -> dict[str,Any]:
    base={"schema_id":"IG_G6_S1_REPAIR_RUN_RECORD_V1","stage_id":"G6:S1R","question_sha256":stage["question_sha256"],"source_sha256":live_source_sha256(),"runtime_sha256":runtime_sha256(),"workers":workers,"started_utc":started,"finished_utc":utc_now(),"wall_seconds":time.monotonic()-t0,"details":dict(details)}
    return dict(base,run_record_sha256=canonical_sha256(base))


def g6_s1_repair_executor(stage:dict[str,Any], cdir:Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_resource_question(stage)
    s0=_load_authority_commit(q,stage,"G6:S0"); s1c=_load_authority_commit(q,stage,"G6:S1"); s1=s1c["result"]
    for key, observed in (("expected_s1_pair_context_rows",s1.get("pair_context_row_count")),("expected_s1_separator_fibres",s1.get("separator_fibre_count")),("expected_s1_exact_legal_owner_realizations",s1.get("exact_legal_owner_realization_count"))):
        if int(observed) != int(q["authority"][key]): raise G6S1RepairError(f"official S1 count mismatch: {key}")
    # Bind Candidate 1 to the descriptor-review values before testing the expanded universe.
    basis=_basis(); observed_rho={k:_rho01(t) for k,t in basis.items()}
    if observed_rho != q["authority"]["basis_rho01_values"]: raise G6S1RepairError("rho01 basis values disagree with frozen descriptor review")
    workers=max(1,min(int((stage.get("execution") or {}).get("parameters",{}).get("workers",4)),4))
    # Regenerate exact S1 child universe and bind it to the official S1 commit.
    universe=_universe_from_s1()
    official_canons=set()
    for r in s1["rows"]: official_canons.update(r["exact_outcome_canon_reprs"])
    regenerated={repr(c) for c in universe}
    if regenerated!=official_canons:
        raise G6S1RepairError("regenerated S1 child universe does not structurally match official S1 commit")
    if len(universe)!=int(q["authority"]["expected_exact_s1_child_universe_count"]):
        raise G6S1RepairError("S1 child universe count mismatch")
    # Freeze an evidence handle to the full structural universe.
    evidence_dir=cdir/"evidence"/"G6__S1R"; evidence_dir.mkdir(parents=True,exist_ok=True)
    universe_index=[{"exact_canon_repr":repr(c),"tree":_tree_to_record(t)} for c,t in sorted(universe.items(),key=lambda kv:repr(kv[0]))]
    up=evidence_dir/"S1_CHILD_UNIVERSE.json"
    if not up.exists(): up.write_text(json.dumps(universe_index,sort_keys=True,separators=(",",":")),encoding="utf-8")
    universe_sha=canonical_sha256(universe_index)
    if canonical_sha256(json.loads(up.read_text(encoding="utf-8")))!=universe_sha:
        raise G6S1RepairError("durable universe checkpoint mismatch")

    attempts=[]; selected=None; expanded=None; basis_replay=None; indep=None
    for cand in q["candidate_ladder"]:
        cid=cand["candidate_id"]
        basis_replay=_basis_replay(s1,cid)
        expanded=_candidate_expanded_test(universe,cid,workers,int(q["resource_budget"]["max_continuation_relation_evaluations"]))
        att={"candidate_id":cid,"basis_replay":basis_replay,"expanded_test":expanded}
        attempts.append(att)
        if expanded["status"]=="BUDGET_INSUFFICIENT":
            outcome="BUDGET_INSUFFICIENT"; break
        if basis_replay["separator_count"]==0 and expanded["status"]=="PASS":
            indep=_independent_verify(universe,cid,int(q["independent_verification"]["read_sample_size"]))
            att["independent_verification"]=indep
            if indep["status"]!="PASS":
                outcome="INDEPENDENT_VERIFICATION_FAILED"; break
            selected=cid
            outcome={
              "C1_RHO01":"PASS_RHO01_EXPANDED_UNIVERSE",
              "C2_TYPED_OWNER_ORBITS":"PASS_TYPED_OWNER_ORBITS_EXPANDED_UNIVERSE",
              "C3_ROOTED_OWNER_RESPONSE_BAG":"PASS_ROOTED_OWNER_RESPONSE_BAG_EXPANDED_UNIVERSE",
            }[cid]
            break
    else:
        outcome="NO_REGISTERED_REPAIR_SUFFICIENT"
    if selected is None and 'outcome' not in locals(): outcome="NO_REGISTERED_REPAIR_SUFFICIENT"

    result_base={
      "schema_id":"IG_G6_S1_REPAIR_RESULT_V1","stage_id":"G6:S1R","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],
      "authority":{"s0_commit_sha256":s0["commit_sha256"],"s1_commit_sha256":s1c["commit_sha256"],"s1_science_sha256":s1.get("science_sha256"),"descriptor_review_audit_sha256":q["authority"]["descriptor_review_audit_sha256"]},
      "exact_s1_child_universe_count":len(universe),"exact_s1_child_universe_sha256":universe_sha,
      "candidate_ladder_attempts":attempts,"selected_candidate":selected,
      "promotion":False,"public_descriptor_promoted":False,"hidden_topology_promoted":False,"g6_graduated":False,
      "next_authorized":"G6:S2_REPAIRED_REGISTRATION_DESIGN" if selected else "G6:S1_DESCRIPTOR_READ_REVIEW",
      "nonclaims":q["nonclaims"],
    }
    result=dict(result_base,science_sha256=canonical_sha256(result_base))
    rr=_run_record(stage,started,t0,workers,{"universe":len(universe),"selected":selected,"outcome":outcome})
    return ChainExecutionResult(result=result,run_record=rr)


def register_g6_s1_repair_executor(controller:Any) -> None:
    controller.register_executor("g6.s1-repair",g6_s1_repair_executor)
