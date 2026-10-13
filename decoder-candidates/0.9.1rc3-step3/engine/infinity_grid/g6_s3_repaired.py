from __future__ import annotations

"""Repaired G6:S3 higher-context challenge.

Frozen scientific role:
- challenge the scoped G6:S2R single-context separation result on genuinely
  higher three-atom G6 carriers;
- no public/topology promotion and no G6 graduation;
- structural canonical representations decide all scientific equality;
- generation and observer work is durably sharded for exact resume.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping
import json, multiprocessing as mp, time

from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from .canon import canonical_sha256
from .g6_stage_executors import _join_exact_relation
from .g6_s1_repair import _basis, _tree_from_record, _tree_to_record
from .g6_s2_repaired import (
    _all_contexts, _independent_sig_for_context, _primary_sig_for_context,
    _universe_from_s1, _universe_index,
)
from .records import live_source_sha256, runtime_sha256, utc_now
from .v05_chain import ChainExecutionResult, ScientificChainError

QUESTION_FILE = "G6_S3_REPAIRED_PREREGISTRATION_V1.json"

class G6S3RepairedError(ScientificChainError):
    pass


def _question(stage: Mapping[str, Any]) -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources/g6").joinpath(QUESTION_FILE)
    q = json.loads(p.read_text(encoding="utf-8"))
    observed = canonical_sha256({k: v for k, v in q.items() if k != "question_sha256"})
    if observed != q.get("question_sha256"):
        raise G6S3RepairedError("G6:S3R question resource hash mismatch")
    if stage.get("stage_id") != "G6:S3R" or q.get("stage_id") != "G6:S3R":
        raise G6S3RepairedError("G6:S3R stage binding mismatch")
    if stage.get("question_sha256") != q["question_sha256"]:
        raise G6S3RepairedError("G6:S3R question hash mismatch")
    params = (stage.get("execution") or {}).get("parameters", {})
    if str(params.get("question_sha256", "")) != q["question_sha256"]:
        raise G6S3RepairedError("G6:S3R executor question hash mismatch")
    declared = str(params.get("working_source_sha256", ""))
    observed_source = live_source_sha256()
    if declared != observed_source:
        raise G6S3RepairedError(f"G6:S3R source binding mismatch: declared {declared}, observed {observed_source}")
    return q


def _load_authority(q: Mapping[str, Any], sid: str) -> dict[str, Any]:
    p = files("infinity_grid").joinpath("resources/g6/authority").joinpath(sid.replace(":", "__") + ".json")
    d = json.loads(p.read_text(encoding="utf-8"))
    expected = d.get("commit_sha256")
    if expected != canonical_sha256({k: v for k, v in d.items() if k != "commit_sha256"}):
        raise G6S3RepairedError(f"authority commit integrity failure {sid}")
    qa = q["authority"][sid]
    checks = {
        "commit_sha256": d.get("commit_sha256"),
        "science_sha256": (d.get("result") or {}).get("science_sha256"),
        "outcome": d.get("outcome"),
    }
    for k, v in checks.items():
        if v != qa.get(k):
            raise G6S3RepairedError(f"authority mismatch {sid}/{k}: {v!r} != {qa.get(k)!r}")
    return d


def _shard_write(path: Path, base: dict[str, Any]) -> None:
    obj = dict(base)
    obj["science_sha256"] = canonical_sha256(obj)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def _shard_read(path: Path, schema_id: str) -> dict[str, Any]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("schema_id") != schema_id:
        raise G6S3RepairedError(f"bad shard schema {path}")
    expected = d.get("science_sha256")
    observed = canonical_sha256({k: v for k, v in d.items() if k != "science_sha256"})
    if expected != observed:
        raise G6S3RepairedError(f"shard integrity failure {path}")
    return d


def _state_record(t: DecoratedG4Tree, canon: tuple[Any, ...]) -> dict[str, Any]:
    sid = canonical_sha256(canon)
    return {"state_id": sid, "exact_canon_repr": repr(canon), "tree": _tree_to_record(t)}


def _axis_a_task(triple: tuple[str, str, str]) -> dict[str, Any]:
    basis = _basis(); op = (0, 0); a, b, c = triple
    records: dict[str, dict[str, Any]] = {}
    contexts = []
    # LEFT_ASSOC: (A B) C
    mids, _, _, _ = _join_exact_relation(basis[a], basis[b], op)
    finals = []; legal = 0
    for m in mids:
        trees, canons, ll, _ = _join_exact_relation(m, basis[c], op)
        legal += ll
        finals.extend(zip(trees, canons))
    for t, can in finals:
        r = _state_record(t, can); records.setdefault(r["state_id"], r)
    contexts.append({"bracketing": "LEFT_ASSOC", "exact_outcome_count": len({repr(c) for _, c in finals}), "legal_owner_realizations": legal})
    # RIGHT_ASSOC: A (B C)
    mids, _, _, _ = _join_exact_relation(basis[b], basis[c], op)
    finals = []; legal = 0
    for m in mids:
        trees, canons, ll, _ = _join_exact_relation(basis[a], m, op)
        legal += ll
        finals.extend(zip(trees, canons))
    for t, can in finals:
        r = _state_record(t, can); records.setdefault(r["state_id"], r)
    contexts.append({"bracketing": "RIGHT_ASSOC", "exact_outcome_count": len({repr(c) for _, c in finals}), "legal_owner_realizations": legal})
    return {"triple": list(triple), "contexts": contexts, "records": [records[k] for k in sorted(records)]}


def _axis_b_contexts() -> tuple[tuple[str, tuple[int, int], str], ...]:
    return (
        ("D2_PATH", (0, 0), "LEFT"),
        ("D2_PATH", (0, 1), "LEFT"),
        ("D2_PATH", (1, 0), "RIGHT"),
        ("D4_PATH", (2, 4), "LEFT"),
    )


def _axis_b_task(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    parent_id, trec = task; t = _tree_from_record(trec); basis = _basis()
    records: dict[str, dict[str, Any]] = {}; contexts = []
    for bref, op, pos in _axis_b_contexts():
        L, R = (t, basis[bref]) if pos == "LEFT" else (basis[bref], t)
        trees, canons, legal, _ = _join_exact_relation(L, R, op)
        for tr, can in zip(trees, canons):
            r = _state_record(tr, can); records.setdefault(r["state_id"], r)
        contexts.append({"probe_ref": bref, "operator": list(op), "position": pos, "exact_outcome_count": len(canons), "legal_owner_realizations": legal})
    return {"parent_state_id": parent_id, "contexts": contexts, "records": [records[k] for k in sorted(records)]}


def _observer_chunk(task: tuple[list[tuple[str, dict[str, Any]]], tuple[tuple[str, tuple[int, int], str], ...]]) -> list[dict[str, Any]]:
    rows, contexts = task; out = []
    for sid, trec in rows:
        t = _tree_from_record(trec)
        sig = []
        for ctx in contexts:
            rs = _primary_sig_for_context(t, ctx)
            sig.append({"context": [ctx[0], list(ctx[1]), ctx[2]], "outcomes": [repr(x) for x in rs]})
        out.append({"state_id": sid, "signature": sig})
    return out


def _fresh_universe_from_shards(gen_root: Path, s1_ids: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}; canon_by_id: dict[str, str] = {}
    axis_counts = {"A_shards": 0, "B_shards": 0, "A_records_raw": 0, "B_records_raw": 0}
    for axis in ("A", "B"):
        d = gen_root / axis
        schema = f"IG_G6_S3R_GENERATION_{axis}_SHARD_V1"
        for p in sorted(d.glob("*.json")):
            sh = _shard_read(p, schema); axis_counts[f"{axis}_shards"] += 1
            axis_counts[f"{axis}_records_raw"] += len(sh["records"])
            for r in sh["records"]:
                sid = r["state_id"]; cr = r["exact_canon_repr"]
                if sid in canon_by_id and canon_by_id[sid] != cr:
                    raise G6S3RepairedError("unexpected SHA collision in higher generation")
                canon_by_id[sid] = cr; states.setdefault(sid, r["tree"])
    overlap = sorted(set(states) & s1_ids)
    return states, dict(axis_counts, fresh_state_count=len(states), overlap_with_s1_count=len(overlap), first_overlap=overlap[0] if overlap else None)


def _generate_panel(q: Mapping[str, Any], cdir: Path, workers: int, s1_records: Mapping[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    root = cdir / "evidence" / "G6__S3R" / "generation"; (root / "A").mkdir(parents=True, exist_ok=True); (root / "B").mkdir(parents=True, exist_ok=True)
    refs = sorted(_basis()); triples = [(a, b, c) for a in refs for b in refs for c in refs]
    if len(triples) * 2 + int(q["higher_panel"]["axis_B_adversarial_h4"]["relation_context_count"]) > int(q["resource_budget"]["max_generation_relation_contexts"]):
        return {}, {"budget_failure": "max_generation_relation_contexts"}
    # A: one durable shard per ordered triple.
    missing = [(i, tr) for i, tr in enumerate(triples) if not (root / "A" / f"{i:03d}.json").exists()]
    if missing:
        tasks = [tr for _, tr in missing]
        if workers <= 1:
            results = map(_axis_a_task, tasks)
        else:
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")); results = ex.map(_axis_a_task, tasks, chunksize=1)
        try:
            for (i, tr), res in zip(missing, results):
                _shard_write(root / "A" / f"{i:03d}.json", {"schema_id": "IG_G6_S3R_GENERATION_A_SHARD_V1", "task_index": i, **res})
        finally:
            if workers > 1: ex.shutdown(wait=True)
    # B: deterministic lexicographically largest 128 S1 exact-canon SHA states.
    n = int(q["higher_panel"]["axis_B_adversarial_h4"]["sample_size"])
    selected = sorted(s1_records, reverse=True)[:n]
    missing_b = [(i, sid) for i, sid in enumerate(selected) if not (root / "B" / f"{i:03d}.json").exists()]
    if missing_b:
        tasks = [(sid, s1_records[sid]) for _, sid in missing_b]
        if workers <= 1:
            results = map(_axis_b_task, tasks)
        else:
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")); results = ex.map(_axis_b_task, tasks, chunksize=1)
        try:
            for (i, sid), res in zip(missing_b, results):
                _shard_write(root / "B" / f"{i:03d}.json", {"schema_id": "IG_G6_S3R_GENERATION_B_SHARD_V1", "task_index": i, **res})
        finally:
            if workers > 1: ex.shutdown(wait=True)
    states, summary = _fresh_universe_from_shards(root, set(s1_records))
    summary.update({"axis_A_expected_shards": len(triples), "axis_B_expected_shards": n, "generation_relation_context_count": len(triples) * 2 + n * len(_axis_b_contexts()), "axis_B_selected_parent_ids_sha256": canonical_sha256(selected)})
    return states, summary


def _new_contexts(level: int) -> list[tuple[str, tuple[int, int], str]]:
    l1, l2, l3 = _all_contexts()
    if level == 1: return l1
    if level == 2: return [x for x in l2 if x not in set(l1)]
    if level == 3: return [x for x in l3 if x not in set(l2)]
    raise ValueError(level)


def _refine_with_checkpoint(groups: list[list[str]], records: Mapping[str, dict[str, Any]], contexts: list[tuple[str, tuple[int, int], str]], level_name: str, cdir: Path, workers: int) -> tuple[list[list[str]], int]:
    target = sorted({sid for g in groups if len(g) > 1 for sid in g})
    if not target: return groups, 0
    root = cdir / "evidence" / "G6__S3R" / "observer" / level_name; root.mkdir(parents=True, exist_ok=True)
    chunk_size = 24
    chunks = [target[i:i+chunk_size] for i in range(0, len(target), chunk_size)]
    missing = []
    for i, ids in enumerate(chunks):
        p = root / f"{i:05d}.json"
        if not p.exists():
            missing.append((i, ids))
    if missing:
        tasks = [([(sid, records[sid]) for sid in ids], tuple(contexts)) for _, ids in missing]
        if workers <= 1:
            results = map(_observer_chunk, tasks)
        else:
            ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")); results = ex.map(_observer_chunk, tasks, chunksize=1)
        try:
            for (i, ids), rows in zip(missing, results):
                _shard_write(root / f"{i:05d}.json", {"schema_id": "IG_G6_S3R_OBSERVER_SHARD_V1", "level": level_name, "chunk_index": i, "state_ids": ids, "rows": rows})
        finally:
            if workers > 1: ex.shutdown(wait=True)
    signatures: dict[str, str] = {}
    for p in sorted(root.glob("*.json")):
        sh = _shard_read(p, "IG_G6_S3R_OBSERVER_SHARD_V1")
        if sh.get("level") != level_name: raise G6S3RepairedError("observer level shard mismatch")
        for row in sh["rows"]:
            signatures[row["state_id"]] = json.dumps(row["signature"], sort_keys=True, separators=(",", ":"))
    if set(signatures) != set(target):
        raise G6S3RepairedError(f"observer shard coverage mismatch at {level_name}")
    new_groups = []
    for g in groups:
        if len(g) <= 1: new_groups.append(g); continue
        buckets: dict[str, list[str]] = defaultdict(list)
        for sid in g: buckets[signatures[sid]].append(sid)
        new_groups.extend(sorted((sorted(v) for v in buckets.values()), key=lambda x: x[0]))
    return sorted(new_groups, key=lambda x: x[0]), len(target) * len(contexts)


def _independent_verify(records: Mapping[str, dict[str, Any]], contexts: list[tuple[str, tuple[int, int], str]], sample_n: int, max_evals: int) -> dict[str, Any]:
    sample = sorted(records)[:min(sample_n, len(records))]
    planned = len(sample) * len(contexts)
    if planned > max_evals:
        return {"status": "BUDGET_INSUFFICIENT", "sample_count": len(sample), "contexts_per_state": len(contexts), "relation_evaluations_planned": planned, "first_failure": None}
    checked = 0; fail = None
    for sid in sample:
        t = _tree_from_record(records[sid])
        for ctx in contexts:
            a = _primary_sig_for_context(t, ctx); b = _independent_sig_for_context(t, ctx); checked += 1
            if a != b:
                fail = {"state_id": sid, "context": {"probe_ref": ctx[0], "operator": list(ctx[1]), "position": ctx[2]}, "primary_outcomes": len(a), "independent_outcomes": len(b)}
                break
        if fail: break
    return {"status": "PASS" if fail is None else "FAIL", "sample_count": len(sample), "contexts_per_state": len(contexts), "relation_evaluations_planned": planned, "relation_evaluations_checked": checked, "first_failure": fail}


def _run_record(stage: Mapping[str, Any], started: str, t0: float, workers: int, details: Mapping[str, Any]) -> dict[str, Any]:
    base = {"schema_id": "IG_G6_S3_REPAIRED_RUN_RECORD_V1", "stage_id": "G6:S3R", "question_sha256": stage["question_sha256"], "source_sha256": live_source_sha256(), "runtime_sha256": runtime_sha256(), "workers": workers, "started_utc": started, "finished_utc": utc_now(), "wall_seconds": time.monotonic() - t0, "details": dict(details)}
    return dict(base, run_record_sha256=canonical_sha256(base))


def g6_s3_repaired_executor(stage: dict[str, Any], cdir: Path) -> ChainExecutionResult:
    started, t0 = utc_now(), time.monotonic(); q = _question(stage)
    auth = {sid: _load_authority(q, sid) for sid in ("G6:S0", "G6:S1", "G6:S1R", "G6:S2R")}
    workers = max(1, min(int((stage.get("execution") or {}).get("parameters", {}).get("workers", 4)), 4))
    # Bind exact S1 universe to certified authority.
    s1 = _universe_from_s1(); s1_idx = _universe_index(s1); s1_sha = canonical_sha256(s1_idx)
    qa = q["authority"]["G6:S2R"]
    if len(s1) != int(qa["exact_s1_child_universe_count"]) or s1_sha != qa["exact_s1_child_universe_sha256"]:
        raise G6S3RepairedError("regenerated S1 universe mismatch")
    s1_records = {canonical_sha256(c): _tree_to_record(t) for c, t in s1.items()}
    if len(s1_records) != len(s1): raise G6S3RepairedError("unexpected S1 state-id collision")
    # Generate fresh higher panel with durable shards.
    fresh, gen = _generate_panel(q, cdir, workers, s1_records)
    if gen.get("budget_failure"):
        outcome = "GENERATION_BUDGET_INSUFFICIENT"; levels=[]; indep=None; used=[]
    elif gen["overlap_with_s1_count"]:
        raise G6S3RepairedError("fresh higher panel overlaps certified S1 universe")
    elif len(fresh) > int(q["resource_budget"]["max_higher_exact_states"]):
        outcome = "GENERATION_BUDGET_INSUFFICIENT"; levels=[]; indep=None; used=[]
    else:
        groups = [sorted(fresh)]; levels=[]; used=[]; total=0; outcome=None
        defs=[("L1_S2R_CONTEXT",1,"PASS_HIGHER_DISCRETE_KERNEL_L1"),("L2_SINGLE_PROBE_FULL_OPERATOR",2,"PASS_HIGHER_DISCRETE_KERNEL_L2"),("L3_FULL_ONE_STEP_OBSERVER",3,"PASS_HIGHER_DISCRETE_KERNEL_L3")]
        for lname, lno, pass_out in defs:
            add = _new_contexts(lno); unresolved = sum(len(g) for g in groups if len(g)>1); planned=unresolved*len(add)
            if total + planned > int(q["resource_budget"]["max_primary_observer_relation_evaluations"]):
                outcome="OBSERVER_BUDGET_INSUFFICIENT"; break
            groups, ev = _refine_with_checkpoint(groups, fresh, add, lname, cdir, workers); total += ev; used.extend(add)
            multi=[g for g in groups if len(g)>1]
            witness=None
            if multi:
                g=sorted(multi,key=lambda x:(-len(x),x))[0]; witness={"class_size":len(g),"state_a":g[0],"state_b":g[1]}
            levels.append({"level":lname,"contexts_total":len(used),"contexts_added":len(add),"states_evaluated":unresolved,"relation_evaluations":ev,"class_count":len(groups),"multi_class_count":len(multi),"max_class_size":max((len(g) for g in groups),default=0),"first_surviving_collision":witness})
            if not multi:
                outcome=pass_out; break
        if outcome is None: outcome="PASS_HIGHER_NONTRIVIAL_ONE_STEP_QUOTIENT"
        indep=None
        if outcome.startswith("PASS_"):
            indep=_independent_verify(fresh,used,int(q["independent_verification"]["sample_size"]),int(q["resource_budget"]["max_independent_relation_evaluations"]))
            if indep["status"]!="PASS": outcome="INDEPENDENT_VERIFICATION_FAILED" if indep["status"]=="FAIL" else "OBSERVER_BUDGET_INSUFFICIENT"
    result_base={
      "schema_id":"IG_G6_S3_REPAIRED_RESULT_V1","stage_id":"G6:S3R","status":"PASS" if outcome.startswith("PASS_") else "REVIEW_REQUIRED","outcome":outcome,"question_sha256":q["question_sha256"],
      "authority":{sid:{"commit_sha256":auth[sid]["commit_sha256"],"science_sha256":auth[sid]["result"]["science_sha256"],"outcome":auth[sid]["outcome"]} for sid in auth},
      "s1_universe_count":len(s1),"s1_universe_sha256":s1_sha,"higher_panel_generation":gen,"fresh_higher_state_count":len(fresh),
      "observer_levels":levels,"registered_contexts_used":len(used) if 'used' in locals() else 0,"primary_observer_relation_evaluations":sum(x.get("relation_evaluations",0) for x in levels),"independent_verification":indep,
      "structural_equality_used_for_decision":True,"hashes_used_for_binding_sampling_and_state_ids_only":True,"promotion":False,"public_descriptor_promoted":False,"hidden_topology_promoted":False,"g6_graduated":False,
      "next_authorized":"G6:S4_REPAIRED_REGISTRATION_DESIGN" if outcome.startswith("PASS_") else "G6:S3_REPAIRED_REVIEW","nonclaims":q["nonclaims"],
    }
    result=dict(result_base,science_sha256=canonical_sha256(result_base))
    rr=_run_record(stage,started,t0,workers,{"outcome":outcome,"fresh_states":len(fresh),"generation_contexts":gen.get("generation_relation_context_count"),"observer_evals":result["primary_observer_relation_evaluations"]})
    return ChainExecutionResult(result=result,run_record=rr)


def register_g6_s3_repaired_executor(controller: Any) -> None:
    controller.register_executor("g6.s3-repaired", g6_s3_repaired_executor)
