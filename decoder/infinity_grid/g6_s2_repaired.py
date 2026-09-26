from __future__ import annotations

"""Repaired G6:S2 behavior-kernel audit.

Scientific discipline:
- Parent authority is official G6:S0/G6:S1 plus certified G6:S1R.
- C3_ROOTED_OWNER_RESPONSE_BAG is used only as the certified exact-state upper bound.
- No public/topology promotion occurs here.
- Full structural canonical tuples decide every scientific equality.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Mapping
import json, multiprocessing as mp, time

from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from .canon import canonical_sha256
from .g6_stage_executors import _join_exact_relation, _local_remaining, _relation_signature
from .g6_s1_repair import _basis, _tree_from_record, _tree_to_record, _universe_from_s1
from .records import live_source_sha256, runtime_sha256, utc_now
from .uplift_g5_r3 import independent_unrooted_canon
from .v05_chain import ChainExecutionResult, ScientificChainError

QUESTION_FILE = "G6_S2_REPAIRED_PREREGISTRATION_V1.json"

class G6S2RepairedError(ScientificChainError):
    pass


def _resource_question(stage: Mapping[str, Any]) -> dict[str, Any]:
    p=files("infinity_grid").joinpath("resources/g6").joinpath(QUESTION_FILE)
    q=json.loads(p.read_text(encoding="utf-8"))
    observed=canonical_sha256({k:v for k,v in q.items() if k!="question_sha256"})
    if observed!=q.get("question_sha256"):
        raise G6S2RepairedError("G6:S2R question resource hash mismatch")
    if stage.get("stage_id")!="G6:S2R" or q.get("stage_id")!="G6:S2R":
        raise G6S2RepairedError("G6:S2R stage binding mismatch")
    if stage.get("question_sha256")!=q["question_sha256"]:
        raise G6S2RepairedError("G6:S2R question hash mismatch")
    params=(stage.get("execution") or {}).get("parameters",{})
    if str(params.get("question_sha256",""))!=q["question_sha256"]:
        raise G6S2RepairedError("G6:S2R executor question hash mismatch")
    declared_source=str(params.get("working_source_sha256",""))
    observed_source=live_source_sha256()
    if declared_source!=observed_source:
        raise G6S2RepairedError(f"G6:S2R source binding mismatch: declared {declared_source}, observed {observed_source}")
    return q


def _load_authority_commit(q: Mapping[str,Any], sid: str) -> dict[str,Any]:
    name=sid.replace(":","__")+".json"
    p=files("infinity_grid").joinpath("resources/g6/authority").joinpath(name)
    d=json.loads(p.read_text(encoding="utf-8"))
    expected=d.get("commit_sha256")
    observed=canonical_sha256({k:v for k,v in d.items() if k!="commit_sha256"})
    if expected!=observed:
        raise G6S2RepairedError(f"authority commit integrity failure {sid}")
    qa=q["authority"][sid]
    checks={
      "commit_sha256":d.get("commit_sha256"),
      "science_sha256":(d.get("result") or {}).get("science_sha256"),
      "outcome":d.get("outcome"),
    }
    for k,v in checks.items():
        if v!=qa.get(k):
            raise G6S2RepairedError(f"authority mismatch {sid}/{k}: {v!r} != {qa.get(k)!r}")
    if sid=="G6:S1R":
        r=d["result"]
        for k in ("selected_candidate","exact_s1_child_universe_count","exact_s1_child_universe_sha256"):
            if r.get(k)!=qa.get(k):
                raise G6S2RepairedError(f"authority mismatch {sid}/{k}")
        # Pull the certified C3 partition counts from the selected attempt.
        sel=None
        for att in r.get("candidate_ladder_attempts",[]):
            if att.get("candidate_id")==qa["selected_candidate"]:
                sel=att; break
        if not sel:
            raise G6S2RepairedError("certified C3 attempt missing from S1R authority")
        ex=sel.get("expanded_test") or {}
        if int(ex.get("refined_class_count",-1))!=int(qa["candidate_refined_class_count"]):
            raise G6S2RepairedError("S1R C3 refined class count mismatch")
        if int(ex.get("multi_class_count",-1))!=int(qa["candidate_multi_class_count"]):
            raise G6S2RepairedError("S1R C3 multi class count mismatch")
    return d


def _universe_index(universe: Mapping[tuple[Any,...],DecoratedG4Tree]) -> list[dict[str,Any]]:
    return [{"exact_canon_repr":repr(c),"tree":_tree_to_record(t)} for c,t in sorted(universe.items(),key=lambda kv:repr(kv[0]))]


def _all_contexts() -> tuple[list[tuple[str,tuple[int,int],str]], list[tuple[str,tuple[int,int],str]], list[tuple[str,tuple[int,int],str]]]:
    basis=_basis(); ops=tuple(G4AcceptedAdapter().operator_basis())
    l1=[("D2_PATH",(0,0),"LEFT")]
    l2=[("D2_PATH",tuple(op),pos) for op in ops for pos in ("LEFT","RIGHT")]
    l3=[(bref,tuple(op),pos) for bref in sorted(basis) for op in ops for pos in ("LEFT","RIGHT")]
    if l1[0] not in l2 or any(x not in l3 for x in l2):
        raise G6S2RepairedError("context ladder containment failure")
    if len(l1)!=1 or len(l2)!=62 or len(l3)!=248:
        raise G6S2RepairedError("context ladder size mismatch")
    return l1,l2,l3


def _new_contexts(level:int) -> list[tuple[str,tuple[int,int],str]]:
    l1,l2,l3=_all_contexts()
    if level==1: return l1
    if level==2: return [x for x in l2 if x not in set(l1)]
    if level==3: return [x for x in l3 if x not in set(l2)]
    raise ValueError(level)


def _primary_sig_for_context(t:DecoratedG4Tree, ctx:tuple[str,tuple[int,int],str]) -> tuple[tuple[Any,...],...]:
    bref,op,pos=ctx; b=_basis()[bref]
    L,R=(t,b) if pos=="LEFT" else (b,t)
    _,can,_,_=_join_exact_relation(L,R,op)
    return _relation_signature(can)


def _eval_state_task(task:tuple[str,dict[str,Any],tuple[tuple[str,tuple[int,int],str],...]]) -> tuple[str,tuple[Any,...]]:
    sid,trec,contexts=task; t=_tree_from_record(trec)
    sig=tuple((bref,op,pos,_primary_sig_for_context(t,(bref,op,pos))) for bref,op,pos in contexts)
    return sid,sig


def _manual_join(left:DecoratedG4Tree,right:DecoratedG4Tree,lu:int,rv:int,op:tuple[int,int]) -> DecoratedG4Tree:
    off=left.n
    return DecoratedG4Tree(left.n+right.n, tuple(left.edges)+tuple((a+off,b+off) for a,b in right.edges)+((lu,rv+off),), tuple(left.H_classes)+tuple(right.H_classes), tuple(left.edge_operators)+tuple(right.edge_operators)+(op,))


def _independent_sig_for_context(t:DecoratedG4Tree, ctx:tuple[str,tuple[int,int],str]) -> tuple[Any,...]:
    bref,op,pos=ctx; b=_basis()[bref]; L,R=(t,b) if pos=="LEFT" else (b,t)
    ad=G4AcceptedAdapter(); localL=_local_remaining(ad,L); localR=_local_remaining(ad,R)
    vkeysL=[ad._vertex_key(x) for x in L.H_classes]; vkeysR=[ad._vertex_key(x) for x in R.H_classes]
    can=set()
    for lu in range(L.n):
      if localL[lu][op[0]]<=0: continue
      for rv in range(R.n):
        if localR[rv][op[1]]<=0: continue
        ch=_manual_join(L,R,lu,rv,op)
        cc=independent_unrooted_canon(ch.n,ch.edges,vkeysL+vkeysR,ch.edge_operators)
        can.add(cc)
    return tuple(sorted(can,key=repr))


def _context_json(ctx:tuple[str,tuple[int,int],str]) -> dict[str,Any]:
    return {"probe_ref":ctx[0],"operator":list(ctx[1]),"position":ctx[2]}


def _partition_summary(groups:list[list[str]], level_name:str, contexts_total:int, contexts_added:int, states_evaluated:int, evals:int) -> dict[str,Any]:
    multi=[g for g in groups if len(g)>1]
    first=None
    if multi:
        g=sorted(multi,key=lambda x:(-len(x),x))[0]
        first={"class_size":len(g),"member_exact_canon_sha256":g[:2]}
    return {
      "level":level_name,"contexts_total":contexts_total,"contexts_added":contexts_added,
      "class_count":len(groups),"multi_class_count":len(multi),"max_class_size":max((len(g) for g in groups),default=0),
      "states_evaluated":states_evaluated,"relation_evaluations":evals,"first_surviving_multi_class":first,
    }


def _refine_once(
    groups:list[list[str]],
    records:Mapping[str,dict[str,Any]],
    contexts:list[tuple[str,tuple[int,int],str]],
    workers:int,
) -> tuple[list[list[str]],dict[str,tuple[Any,...]],int]:
    target=sorted({sid for g in groups if len(g)>1 for sid in g})
    if not target:
        return groups,{},0
    tasks=[(sid,records[sid],tuple(contexts)) for sid in target]
    results:dict[str,tuple[Any,...]]={}
    if workers<=1:
        it=map(_eval_state_task,tasks)
        for sid,sig in it: results[sid]=sig
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as ex:
            for sid,sig in ex.map(_eval_state_task,tasks,chunksize=4): results[sid]=sig
    new_groups=[]
    for g in groups:
        if len(g)<=1:
            new_groups.append(g); continue
        buckets:dict[tuple[Any,...],list[str]]=defaultdict(list)
        for sid in g:
            buckets[results[sid]].append(sid)  # actual structural tuple key decides equality
        new_groups.extend(sorted((sorted(v) for v in buckets.values()),key=lambda x:x[0]))
    return sorted(new_groups,key=lambda x:x[0]),results,len(target)*len(contexts)


def _independent_verify(records:Mapping[str,dict[str,Any]], contexts:list[tuple[str,tuple[int,int],str]], sample_n:int, max_evals:int) -> dict[str,Any]:
    ordered=sorted(records,key=str)
    # ids are exact-canon SHA-256 strings, so lexicographic ordering implements frozen sample rule.
    sample=ordered[:min(sample_n,len(ordered))]
    planned=len(sample)*len(contexts)
    if planned>max_evals:
        return {"status":"BUDGET_INSUFFICIENT","sample_count":len(sample),"contexts_per_state":len(contexts),"relation_evaluations_planned":planned,"first_failure":None}
    checked=0; fail=None
    for sid in sample:
        t=_tree_from_record(records[sid])
        for ctx in contexts:
            a=_primary_sig_for_context(t,ctx); b=_independent_sig_for_context(t,ctx); checked+=1
            if a!=b:
                fail={"exact_canon_sha256":sid,"context":_context_json(ctx),"primary_outcome_count":len(a),"independent_outcome_count":len(b)}
                break
        if fail: break
    return {"status":"PASS" if fail is None else "FAIL","sample_count":len(sample),"contexts_per_state":len(contexts),"relation_evaluations_planned":planned,"relation_evaluations_checked":checked,"first_failure":fail}


def _run_record(stage:Mapping[str,Any],started:str,t0:float,workers:int,details:Mapping[str,Any]) -> dict[str,Any]:
    base={"schema_id":"IG_G6_S2_REPAIRED_RUN_RECORD_V1","stage_id":"G6:S2R","question_sha256":stage["question_sha256"],"source_sha256":live_source_sha256(),"runtime_sha256":runtime_sha256(),"workers":workers,"started_utc":started,"finished_utc":utc_now(),"wall_seconds":time.monotonic()-t0,"details":dict(details)}
    return dict(base,run_record_sha256=canonical_sha256(base))


def g6_s2_repaired_executor(stage:dict[str,Any], cdir:Path) -> ChainExecutionResult:
    started,t0=utc_now(),time.monotonic(); q=_resource_question(stage)
    s0=_load_authority_commit(q,"G6:S0"); s1=_load_authority_commit(q,"G6:S1"); s1r=_load_authority_commit(q,"G6:S1R")
    qa=q["authority"]["G6:S1R"]
    workers=max(1,min(int((stage.get("execution") or {}).get("parameters",{}).get("workers",4)),4))

    # Regenerate and structurally bind the complete certified S1 universe.
    universe=_universe_from_s1(); uidx=_universe_index(universe); usha=canonical_sha256(uidx)
    if len(universe)!=int(qa["exact_s1_child_universe_count"]) or usha!=qa["exact_s1_child_universe_sha256"]:
        raise G6S2RepairedError("regenerated S1 universe differs from certified S1R universe")
    if qa["selected_candidate"]!="C3_ROOTED_OWNER_RESPONSE_BAG" or int(qa["candidate_refined_class_count"])!=len(universe) or int(qa["candidate_multi_class_count"])!=0:
        raise G6S2RepairedError("S1R C3 authority does not certify a 4520-state discrete upper-bound input")

    evidence_dir=cdir/"evidence"/"G6__S2R"; evidence_dir.mkdir(parents=True,exist_ok=True)
    index_path=evidence_dir/"S1_CHILD_UNIVERSE_INDEX.json"
    if not index_path.exists(): index_path.write_text(json.dumps(uidx,sort_keys=True,separators=(",",":")),encoding="utf-8")
    if canonical_sha256(json.loads(index_path.read_text(encoding="utf-8")))!=usha:
        raise G6S2RepairedError("durable S2R universe index mismatch")

    records={canonical_sha256(c):_tree_to_record(t) for c,t in universe.items()}
    if len(records)!=len(universe):
        raise G6S2RepairedError("unexpected exact-canon SHA collision in evidence index")
    groups=[sorted(records)]
    levels=[]; total_evals=0; used_contexts=[]; selected_level=None
    max_primary=int(q["resource_budget"]["max_primary_relation_evaluations"])
    level_defs=[("L1_FIRST_CONTEXT",1),("L2_SINGLE_PROBE_FULL_OPERATOR",2),("L3_FULL_ONE_STEP_OBSERVER",3)]
    outcome=None
    for level_name,level_no in level_defs:
        add=_new_contexts(level_no)
        unresolved_states=sum(len(g) for g in groups if len(g)>1)
        planned=unresolved_states*len(add)
        if total_evals+planned>max_primary:
            outcome="BUDGET_INSUFFICIENT"; break
        groups,_,evals=_refine_once(groups,records,add,workers)
        total_evals+=evals; used_contexts.extend(add)
        summary=_partition_summary(groups,level_name,len(used_contexts),len(add),unresolved_states,evals)
        levels.append(summary)
        if summary["multi_class_count"]==0:
            selected_level=level_name
            outcome={1:"PASS_DISCRETE_KERNEL_L1",2:"PASS_DISCRETE_KERNEL_L2",3:"PASS_DISCRETE_KERNEL_L3"}[level_no]
            break
    if outcome is None:
        # L3 completed and retains nontrivial behavior classes.
        outcome="PASS_NONTRIVIAL_FULL_ONE_STEP_QUOTIENT"
        selected_level="L3_FULL_ONE_STEP_OBSERVER"

    quotient_path=None; quotient_sha=None
    if outcome=="PASS_NONTRIVIAL_FULL_ONE_STEP_QUOTIENT":
        quotient=[{"class_id":i,"size":len(g),"member_exact_canon_sha256":g} for i,g in enumerate(groups)]
        quotient_path=evidence_dir/"FULL_ONE_STEP_BEHAVIOR_QUOTIENT.json"
        quotient_path.write_text(json.dumps(quotient,sort_keys=True,separators=(",",":")),encoding="utf-8")
        quotient_sha=canonical_sha256(quotient)

    indep=None
    if outcome.startswith("PASS_"):
        indep=_independent_verify(records,used_contexts,int(q["independent_verification"]["sample_size"]),int(q["resource_budget"]["max_independent_relation_evaluations"]))
        if indep["status"]!="PASS":
            outcome="INDEPENDENT_VERIFICATION_FAILED" if indep["status"]=="FAIL" else "BUDGET_INSUFFICIENT"

    ctx_record=[_context_json(c) for c in used_contexts]
    context_path=evidence_dir/"USED_CONTEXTS.json"; context_path.write_text(json.dumps(ctx_record,sort_keys=True,separators=(",",":")),encoding="utf-8")
    levels_path=evidence_dir/"PARTITION_LEVELS.json"; levels_path.write_text(json.dumps(levels,sort_keys=True,separators=(",",":")),encoding="utf-8")

    result_base={
      "schema_id":"IG_G6_S2_REPAIRED_RESULT_V1","stage_id":"G6:S2R","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],
      "authority":{"s0_commit_sha256":s0["commit_sha256"],"s1_commit_sha256":s1["commit_sha256"],"s1r_commit_sha256":s1r["commit_sha256"],"s1r_science_sha256":s1r["result"]["science_sha256"],"selected_candidate":qa["selected_candidate"]},
      "candidate_input_role":q["candidate_input"]["role"],"candidate_promoted":False,
      "exact_s1_child_universe_count":len(universe),"exact_s1_child_universe_sha256":usha,
      "registered_contexts_used":len(used_contexts),"used_contexts_sha256":canonical_sha256(ctx_record),
      "partition_levels":levels,"selected_level":selected_level,
      "final_behavior_class_count":len(groups),"final_multi_class_count":sum(1 for g in groups if len(g)>1),"final_max_class_size":max((len(g) for g in groups),default=0),
      "full_behavior_quotient_sha256":quotient_sha,
      "primary_relation_evaluations":total_evals,"independent_verification":indep,
      "structural_equality_used_for_decision":True,"hashes_used_for_index_and_sample_only":True,
      "monotonicity_rule_applied":bool(outcome in {"PASS_DISCRETE_KERNEL_L1","PASS_DISCRETE_KERNEL_L2"}),
      "promotion":False,"public_descriptor_promoted":False,"hidden_topology_promoted":False,"g6_graduated":False,
      "next_authorized":"G6:S3_REPAIRED_REGISTRATION_DESIGN" if outcome.startswith("PASS_") else "G6:S2_REPAIRED_REVIEW",
      "nonclaims":q["nonclaims"],
    }
    result=dict(result_base,science_sha256=canonical_sha256(result_base))
    rr=_run_record(stage,started,t0,workers,{"universe":len(universe),"outcome":outcome,"selected_level":selected_level,"contexts_used":len(used_contexts),"primary_relation_evaluations":total_evals})
    return ChainExecutionResult(result=result,run_record=rr)


def register_g6_s2_repaired_executor(controller:Any) -> None:
    controller.register_executor("g6.s2-repaired",g6_s2_repaired_executor)
