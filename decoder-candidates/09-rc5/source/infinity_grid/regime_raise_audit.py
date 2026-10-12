from __future__ import annotations

import json, os, shutil, statistics, tempfile
from collections import defaultdict
from importlib.resources import files
from pathlib import Path
from typing import Any

from . import regime_scanner as rs


def load_raise_audit_spec() -> dict:
    p = files("infinity_grid").joinpath("resources/decoder/O12_O13_GROUP_RAISE_AUDIT_SPEC_v1.json")
    return json.loads(p.read_text(encoding="utf-8"))


def _recipe_id(s: Any) -> str:
    return f"{getattr(s,'lane','')}|{getattr(s,'motif_id','')}"


def _candidate_pool(engine, prev, level, pairs, motifs, scanner_spec):
    prev=sorted(prev,key=lambda s:s.construction_digest)
    totals=[sum(s.total_caps) for s in prev]
    med=statistics.median(totals)
    center=min(prev,key=lambda s:(abs(sum(s.total_caps)-med),s.construction_digest))
    diverse=prev
    cand=[]; failures=0
    for mi,m in enumerate(motifs):
        n=int(m["n"]); edges=[tuple(e) for e in m["edges"]]
        s=rs._build_lift(engine,level,[center]*n,edges,pairs,"HOM",f"HOM:{n}:{mi}",schedule_seed=mi%len(pairs))
        if s:cand.append(s)
        else:failures+=1
        if n in (4,5) or (n==6 and mi%6==0):
            owners=[diverse[j%len(diverse)] for j in range(n)]
            s=rs._build_lift(engine,level,owners,edges,pairs,"HET",f"HET:{n}:{mi}",schedule_seed=(mi*3+1)%len(pairs))
            if s:cand.append(s)
            else:failures+=1
        if n==4 and len(diverse)>=2:
            owners=[diverse[0],diverse[1],diverse[0],diverse[1]]
            s=rs._build_lift(engine,level,owners,edges,pairs,"MIX",f"MIX:4:{mi}",schedule_seed=(mi*5+2)%len(pairs))
            if s:cand.append(s)
            else:failures+=1
    twin=[]
    if center.total_caps[0]>=6:
        A=rs._build_lift(engine,level,[center]*6,list(rs.GA),pairs,"TWIN","TWIN:A",force_pair=(0,0))
        B=rs._build_lift(engine,level,[center]*6,list(rs.GB),pairs,"TWIN","TWIN:B",force_pair=(0,0))
        if A and B: cand.extend([A,B]); twin=[A,B]
    cap=int(scanner_spec["panel"]["candidate_cap"])
    if len(cand)>cap:
        cand=sorted(cand,key=lambda s:s.construction_digest)[:cap]
        for t in twin:
            if all(x.construction_digest!=t.construction_digest for x in cand): cand[-1]=t
    # Recipe IDs should be unique in the frozen generator.
    ids=[_recipe_id(s) for s in cand]
    if len(ids)!=len(set(ids)):
        dup=sorted(k for k,v in __import__('collections').Counter(ids).items() if v>1)
        raise RuntimeError(f"duplicate recipe IDs at O{level}: {dup[:5]}")
    default=rs._farthest_select(cand,int(scanner_spec["panel"]["beam"]),must_include=twin)
    backbone=rs._build_backbone(engine,level,center,pairs)
    return cand,default,backbone,{"candidates":len(cand),"build_failures":failures,"twin_constructed":len(twin)==2}


def _recipe_signature(s, bridge_pairs):
    a=rs._action_aggregate(s,bridge_pairs)
    g=rs._graph_basic(len(s.owner_caps),s.top_pairs)
    f=rs._factor_fiber(s)
    return {
        "branching": {
            "legal_action_labels":a["legal_action_labels"],
            "action_orbits":a["action_orbits"],
            "type_pair_support":a["type_pair_support"],
            "owner_pair_support":a["owner_pair_support"],
            "service_classes":a["service_classes"],
        },
        "symmetry": {
            "automorphism_size":a["automorphism_size"],
            "owner_orbits":a["owner_orbits"],
        },
        "lineage": {
            "factor_fiber_size":len(f),
            "bridge_fraction":g["bridges"]/max(1,len(s.top_pairs)),
            "cycle_rank":g["beta"],
        },
        "topology_services": {
            "degree":g["degree"],"diameter":g["diameter"],"radius":g["radius"],
            "articulations":g["articulations"],"bridges":g["bridges"],"triangles":g["triangles"],"beta":g["beta"],
            "service_signature_sha256":a["service_signature_sha256"],
        },
        "endpoint_support_by_owner":[tuple(int(x>0) for x in c) for c in s.owner_caps],
    }


def _stratified_ids(pool, k=24):
    groups=defaultdict(list)
    for s in pool:
        n=len(s.owner_caps)
        groups[(s.lane,n)].append(_recipe_id(s))
    for v in groups.values(): v.sort()
    keys=sorted(groups)
    out=[]; i=0
    while len(out)<k:
        progressed=False
        for key in keys:
            vals=groups[key]
            if i<len(vals):
                out.append(vals[i]); progressed=True
                if len(out)>=k: break
        if not progressed: break
        i+=1
    return out


def _select_by_ids(pool, ids):
    mp={_recipe_id(s):s for s in pool}
    missing=[x for x in ids if x not in mp]
    if missing: raise RuntimeError(f"matched recipe IDs missing: {missing[:5]}")
    return [mp[x] for x in ids]


def _lane_hash(scan, lane):
    # Compare the frozen organizational identity, not Counter-magnitude diagnostics.
    # This is especially important for branching: raw branching includes total action-copy
    # magnitude, while the regime scanner explicitly excludes that from normalized novelty.
    return rs._sha(scan["normalized_signature"][lane])


def run_o12_o13_raise_audit(phase8_seed:Path, official_result:Path, output:Path, keep_work:bool=False) -> dict:
    spec=load_raise_audit_spec(); scanner_spec=rs.load_regime_scanner_spec(); motifs=rs.load_motif_library()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    official=json.loads(Path(official_result).read_text(encoding="utf-8"))
    if official.get("science_sha256")!=spec["candidate_under_audit"]["source_science_sha256"]:
        raise RuntimeError("official candidate science hash mismatch")
    work=Path(tempfile.mkdtemp(prefix="ig_o12_raise_",dir=str(output)))
    try:
        phase8,o7root=rs._extract_seed(Path(phase8_seed),work)
        os.environ["OSCOUT_DATA_ROOT"]=str(o7root.resolve())
        engine=rs._load_module("ig_raise_o7_engine",o7root/"02_CODE"/"o7_live_engine.py"); engine.O6=engine.import_o6()
        parent_map=engine.load_parent_records(); _,bpairs=engine.O6.load_rules(); bridge_pairs=sorted(tuple(map(int,x)) for x in bpairs)
        records=json.loads((phase8/"graduation_compact"/"07_INPUT_SNAPSHOTS"/"O7_IMMUTABLE_SURVIVORS.json").read_text())["records"]
        base=[]
        for r in records:
            ctx=engine._profile_row_context(r,parent_map); edges=tuple(tuple(x) for x in r["edges"])
            base.append(rs.O7State(engine,ctx,edges,(0,0,0,0,0,0,0),r["state_digest"],r["lane"]))
        prev=rs._farthest_select(base,int(scanner_spec["panel"]["beam"]))
        pools={}; defaults={7:prev}; backbones={}; default_scans={}; scan7,_=rs._scan_level(prev,7,bridge_pairs,rs.GRAMMAR_EXPECTED); default_scans[7]=scan7
        matched_ids=None; strat_ids=None
        matched_scans={"LEX24":{},"STRAT24":{},"FULL_POOL":{}}
        recipe_sigs={}
        panel_membership={}
        build_meta={}
        for level in range(8,14):
            pool,default,backbone,bm=_candidate_pool(engine,prev,level,bridge_pairs,motifs,scanner_spec)
            pools[level]=pool; defaults[level]=default; backbones[level]=backbone; build_meta[level]=bm
            if level==8:
                all_ids=sorted(_recipe_id(s) for s in pool)
                matched_ids=all_ids[:24]
                strat_ids=_stratified_ids(pool,24)
            lex=_select_by_ids(pool,matched_ids); strat=_select_by_ids(pool,strat_ids)
            for name,states in [("LEX24",lex),("STRAT24",strat),("FULL_POOL",pool)]:
                sc,_=rs._scan_level(states,level,bridge_pairs,rs.GRAMMAR_EXPECTED); matched_scans[name][level]=sc
            dscan,_=rs._scan_level(default,level,bridge_pairs,rs.GRAMMAR_EXPECTED); default_scans[level]=dscan
            panel_membership[level]=sorted(_recipe_id(s) for s in default)
            if level in (11,12,13):
                recipe_sigs[level]={_recipe_id(s):_recipe_signature(s,bridge_pairs) for s in pool}
            prev=default

        # Gate A1: exact reproduction of official normalized signatures O7-O13.
        reproduction={}
        for level in range(7,14):
            got=default_scans[level]["normalized_signature_sha256"]
            exp=official["level_summaries"][str(level)]["normalized_signature_sha256"]
            reproduction[str(level)]={"expected":exp,"observed":got,"pass":got==exp}
        A1=all(x["pass"] for x in reproduction.values())

        # A2: full endpoint support on every audit candidate O11-O13.
        support_fail=[]
        for level in (11,12,13):
            for s in pools[level]:
                for oi,c in enumerate(s.owner_caps):
                    mask=tuple(int(x>0) for x in c)
                    if mask!=(1,1,1,1,1,1,1): support_fail.append([level,_recipe_id(s),oi,list(mask)])
        A2=not support_fail

        # A3: per-recipe exact normalized signature invariance across O11/O12/O13.
        common=set(recipe_sigs[11]) & set(recipe_sigs[12]) & set(recipe_sigs[13])
        recipe_mismatch=[]; lane_mismatch_counts=defaultdict(int)
        for rid in sorted(common):
            for lane in ("branching","symmetry","lineage","topology_services"):
                h=[rs._sha(recipe_sigs[L][rid][lane]) for L in (11,12,13)]
                if len(set(h))!=1:
                    recipe_mismatch.append({"recipe_id":rid,"lane":lane,"hashes":h}); lane_mismatch_counts[lane]+=1
        A3=(len(common)==len(pools[11])==len(pools[12])==len(pools[13]) and not recipe_mismatch)

        # A4: frozen matched panels and full pool should be lane-stable O11-O13.
        matched_panel_results={}
        A4=True
        for name,ls in matched_scans.items():
            lane_status={}
            for lane in ("branching","symmetry","lineage","topology_services"):
                hs=[_lane_hash(ls[L],lane) for L in (11,12,13)]
                lane_status[lane]={"hashes":hs,"stable":len(set(hs))==1}
                A4 &= lane_status[lane]["stable"]
            matched_panel_results[name]={"lanes":lane_status,"normalized_signature_sha256":[ls[L]["normalized_signature_sha256"] for L in (11,12,13)],"states":[ls[L]["states"] for L in (11,12,13)]}

        # A5: default panel membership transition and exact accounting of aggregate changes.
        def jacc(a,b):
            A=set(a);B=set(b);return len(A&B)/max(1,len(A|B))
        membership={
            "O11_O12_jaccard":jacc(panel_membership[11],panel_membership[12]),
            "O12_O13_jaccard":jacc(panel_membership[12],panel_membership[13]),
            "O11_only":sorted(set(panel_membership[11])-set(panel_membership[12])),
            "O12_only_vs_O11":sorted(set(panel_membership[12])-set(panel_membership[11])),
            "O12_only_vs_O13":sorted(set(panel_membership[12])-set(panel_membership[13])),
            "O13_only_vs_O12":sorted(set(panel_membership[13])-set(panel_membership[12])),
            "O12_equals_O13_recipe_set":set(panel_membership[12])==set(panel_membership[13]),
        }
        default_lane_hashes={lane:[_lane_hash(default_scans[L],lane) for L in (11,12,13)] for lane in ("branching","symmetry","lineage","topology_services")}
        changed_11_12=[lane for lane,hs in default_lane_hashes.items() if hs[0]!=hs[1]]
        persisted_12_13=[lane for lane,hs in default_lane_hashes.items() if hs[1]==hs[2]]
        reported=set(spec["candidate_under_audit"]["reported_families"])
        A5=(reported.issubset(set(changed_11_12)) and reported.issubset(set(persisted_12_13)) and membership["O11_O12_jaccard"]<1.0 and A3 and A4)

        # A6: candidate survives only if depth-dependent matched/per-recipe change exists in >=3 lanes.
        depth_dependent_lanes=[lane for lane,count in lane_mismatch_counts.items() if count>0]
        # Matched panel instability would also count as depth dependence.
        for name,r in matched_panel_results.items():
            for lane,v in r["lanes"].items():
                if not v["stable"] and lane not in depth_dependent_lanes: depth_dependent_lanes.append(lane)
        survives=len(set(depth_dependent_lanes))>=3
        A6=not survives
        if A1 and A2 and A3 and A4 and A5 and A6:
            classification=spec["outcomes"]["REJECT"]
            interpretation=("The O12/O13 structural-shock signal is not depth-intrinsic in the frozen scanner observer. "
                            "Every common exact construction recipe has byte-identical normalized branching, symmetry, lineage and topology-service signatures at O11/O12/O13; "
                            "two independent frozen recipe-matched panels and the complete 193-recipe pool are likewise stable. "
                            "The official default farthest-point beam changes recipe membership between O11 and O12 and then stabilizes, which exactly explains the aggregate medians/histograms that triggered the raise candidate.")
        elif survives:
            classification=spec["outcomes"]["EARN"]
            interpretation="At least three normalized organizational lanes retain depth-dependent changes under exact per-recipe or frozen matched-panel comparison; the candidate survives targeted audit."
        else:
            classification=spec["outcomes"]["UNRESOLVED"]
            interpretation="The targeted gates did not support a clean earn or rejection."

        result={
            "schema":"IG_O12_O13_GROUP_RAISE_AUDIT_RESULT_V1","date":"2026-08-30","status":"PASS" if classification!=spec["outcomes"]["UNRESOLVED"] else "UNRESOLVED",
            "classification":classification,"spec_sha256":rs._sha(spec),"source_candidate_science_sha256":official["science_sha256"],
            "gates":{
                "A1_reproduce_candidate":{"pass":A1,"levels":reproduction},
                "A2_full_support":{"pass":A2,"failures":support_fail[:20],"failure_count":len(support_fail)},
                "A3_recipe_invariance":{"pass":A3,"common_recipes":len(common),"pool_sizes":{str(L):len(pools[L]) for L in (11,12,13)},"mismatches":recipe_mismatch[:50],"mismatch_count":len(recipe_mismatch)},
                "A4_matched_panels":{"pass":A4,"panels":matched_panel_results,"LEX24_recipe_ids":matched_ids,"STRAT24_recipe_ids":strat_ids},
                "A5_selection_causality":{"pass":A5,"membership":membership,"default_lane_hashes":default_lane_hashes,"changed_O11_to_O12":changed_11_12,"persisted_O12_to_O13":persisted_12_13},
                "A6_raise_criterion":{"pass":A6,"depth_dependent_lanes":sorted(set(depth_dependent_lanes)),"candidate_survives":survives}
            },
            "exact_mechanism":{
                "statement":"Under full seven-type endpoint support and the frozen top-level motif/type schedule, the audited normalized branching/symmetry/lineage/topology-service observables factor through the recipe's typed top graph and support pattern; Counter magnitude and nested depth are excluded by the frozen observer.",
                "implementation_dependencies":["_action_aggregate uses owner support booleans plus typed top edges for action-orbit/service organization","_factor_fiber uses top edges plus support colors","_graph_basic uses only the top graph","service identity excludes Counter magnitude"],
                "verified_hypotheses":{"full_support_O11_O13":A2,"fixed_common_recipe_set":len(common)==193,"zero_per_recipe_mismatch":len(recipe_mismatch)==0}
            },
            "build_meta":{str(L):build_meta[L] for L in range(8,14)},
            "default_panel_recipe_membership":{str(L):panel_membership[L] for L in range(8,14)},
            "scientific_interpretation":interpretation,
            "nonclaims":spec["nonclaims"],
            "next":"PATCH_SCANNER_STRUCTURAL_SHOCK_PROMOTION_TO_REQUIRE_MATCHED_LONGITUDINAL_EVIDENCE; then resume O-regime scouting from O13 rather than promoting O12."
        }
        result["science_sha256"]=rs._sha({k:v for k,v in result.items() if k!="science_sha256"})
        (output/"O12_O13_GROUP_RAISE_AUDIT_RESULT.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        (output/"O12_O13_GROUP_RAISE_AUDIT_PANEL_MEMBERSHIP.json").write_text(json.dumps({"schema":"IG_O12_O13_AUDIT_PANEL_MEMBERSHIP_V1","rows":result["default_panel_recipe_membership"]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return result
    finally:
        if keep_work:
            (output/"WORKDIR.txt").write_text(str(work)+"\n")
        else:
            shutil.rmtree(work,ignore_errors=True)
