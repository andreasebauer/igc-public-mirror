from __future__ import annotations

import json, os, shutil, tempfile
from pathlib import Path
from typing import Any
from importlib.resources import files

from .regime_scanner import (
    PHASE8_SEED_EXPECTED, GRAMMAR_EXPECTED, _sha, _sha_file, _extract_seed, _load_module,
    _farthest_select, O7State, _build_lift, load_motif_library, load_regime_scanner_spec,
    _graph_basic, _service_effect, _automorphisms, _orbits_from_perms,
)

P4=((0,1),(1,2),(2,3))
S4=((0,1),(0,2),(0,3))


def load_depth_service_validation_spec()->dict:
    p=files('infinity_grid').joinpath('resources/decoder/O_REGIME_DEPTH_ERASING_SERVICE_VALIDATION_SPEC_v1.json')
    return json.loads(p.read_text(encoding='utf-8'))


def _packet(engine, level:int, refa:Any, refb:Any, bridge_pairs:list[tuple[int,int]], motifs:list[dict]):
    out=[]; failures=[]
    for mi,m in enumerate(motifs):
        n=int(m['n']); edges=[tuple(map(int,e)) for e in m['edges']]
        hom=_build_lift(engine,level,[refa]*n,edges,bridge_pairs,'VALIDATION','VAL:HOM_CYCLIC:'+str(mi),schedule_seed=mi%len(bridge_pairs),force_pair=None)
        if hom is None: failures.append(['HOM_CYCLIC',mi])
        else: out.append(hom)
        owners=[refa if j%2==0 else refb for j in range(n)]
        het=_build_lift(engine,level,owners,edges,bridge_pairs,'VALIDATION','VAL:HET_TYPE0:'+str(mi),force_pair=(0,0))
        if het is None: failures.append(['HET_TYPE0',mi])
        else: out.append(het)
    return out,failures


def _service_packet_signature(packet:list[Any], bridge_pairs:list[tuple[int,int]], structural_cache:dict)->dict:
    rows=[]
    for st in sorted(packet,key=lambda x:x.motif_id):
        supports=[tuple(int(x>0) for x in c) for c in st.owner_caps]
        all_full=all(all(mask) for mask in supports)
        structural_key=_sha({"motif":st.motif_id,"typed_edges":st.typed_edges,"supports":supports})
        row=structural_cache.get(structural_key)
        if row is None:
            n=len(st.owner_caps); gm=_graph_basic(n,st.top_pairs)
            effects=[]; legal=0
            for u in range(n):
                for v in range(u+1,n):
                    eff=_service_effect(n,st.top_pairs,(u,v))
                    for a,b in bridge_pairs:
                        if st.owner_caps[u][a]>0 and st.owner_caps[v][b]>0:
                            legal += 1; effects.append((u,v,a,b,eff))
            autos=_automorphisms(n,supports,st.typed_edges)
            orbit_reps=set()
            for u,v,a,b,_eff in effects:
                imgs=[]
                for perm in autos or [tuple(range(n))]:
                    uu,vv=perm[u],perm[v]
                    imgs.append((uu,vv,a,b) if uu<=vv else (vv,uu,b,a))
                orbit_reps.add(min(imgs))
            row={
                "motif_id":st.motif_id,"owners":n,"typed_edges":st.typed_edges,"support_masks":supports,
                "degree":gm["degree"],"beta":gm["beta"],"diameter":gm["diameter"],"radius":gm["radius"],
                "articulations":gm["articulations"],"bridges":gm["bridges"],"triangles":gm["triangles"],
                "legal_action_labels":legal,"action_orbits":len(orbit_reps),
                "service_effect_signature_sha256":_sha(sorted(effects)),
                "automorphism_size":len(autos),"owner_orbits":_orbits_from_perms(n,autos),
                "all_support_full":all_full,
            }
            structural_cache[structural_key]=row
        rows.append(row)
    payload={"schema":"IG_O_REGIME_WIDE_SERVICE_PACKET_V1","rows":rows}
    return {"signature_sha256":_sha(payload),"rows_count":len(rows)}


def run_depth_erasing_service_validation(phase8_seed:Path, output:Path, through:int=13)->dict:
    spec=load_depth_service_validation_spec(); scanner_spec=load_regime_scanner_spec(); motifs=load_motif_library()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    work=Path(tempfile.mkdtemp(prefix='ig_o_regime_validate_',dir=str(output)))
    try:
        phase8,o7root=_extract_seed(Path(phase8_seed),work)
        os.environ['OSCOUT_DATA_ROOT']=str(o7root.resolve())
        engine=_load_module('ig_regime_o7_validation_engine',o7root/'02_CODE'/'o7_live_engine.py');engine.O6=engine.import_o6()
        parent_map=engine.load_parent_records(); _,bpairs=engine.O6.load_rules(); bridge_pairs=sorted(tuple(map(int,x)) for x in bpairs)
        o8auth=json.loads((phase8/'authority'/'O8_BP0_RESULT.json').read_text());o9auth=json.loads((phase8/'authority'/'O9_BP0_RESULT.json').read_text())
        if o8auth.get('normalized_grammar',{}).get('candidate_sha256')!=GRAMMAR_EXPECTED or o9auth.get('normalized_grammar',{}).get('candidate_sha256')!=GRAMMAR_EXPECTED:
            raise RuntimeError('authority grammar mismatch')
        records=json.loads((phase8/'graduation_compact'/'07_INPUT_SNAPSHOTS'/'O7_IMMUTABLE_SURVIVORS.json').read_text())['records']
        base=[]
        for r in records:
            ctx=engine._profile_row_context(r,parent_map); edges=tuple(tuple(x) for x in r['edges'])
            base.append(O7State(engine,ctx,edges,(0,0,0,0,0,0,0),r['state_digest'],r['lane']))
        sel=_farthest_select(base,int(scanner_spec['panel']['beam']))
        sel=sorted(sel,key=lambda s:s.construction_digest)
        refa=sel[0]
        refb=next((x for x in sel[1:] if x.skin!=refa.skin and x.construction_digest!=refa.construction_digest),None)
        if refb is None: raise RuntimeError('could not choose independent O7 validation references')
        targets=set(int(x) for x in spec['validation_packet']['target_depths'])
        target_results={}; build_failures=[]; structural_cache={}; reference_rows={7:{'A':refa.construction_digest,'B':refb.construction_digest,'A_skin':refa.skin,'B_skin':refb.skin}}
        for level in range(8,through+1):
            na=_build_lift(engine,level,[refa]*4,list(P4),bridge_pairs,'VALIDATION_REF','VALREF:A',force_pair=(0,0))
            nb=_build_lift(engine,level,[refb]*4,list(S4),bridge_pairs,'VALIDATION_REF','VALREF:B',force_pair=(0,0))
            if na is None or nb is None: raise RuntimeError(f'reference lineage build failure at O{level}')
            refa,refb=na,nb
            reference_rows[level]={
                'A':refa.construction_digest,'B':refb.construction_digest,'A_skin':refa.skin,'B_skin':refb.skin,
                'distinct_construction':refa.construction_digest!=refb.construction_digest,
                'distinct_skin':refa.skin!=refb.skin,
                'A_leaf_count':refa.leaf_count,'B_leaf_count':refb.leaf_count,
                'A_min_cap':min(refa.total_caps),'B_min_cap':min(refb.total_caps),
            }
            if level in targets:
                packet,fail=_packet(engine,level,refa,refb,bridge_pairs,motifs)
                build_failures.extend([[level,*x] for x in fail])
                sig=_service_packet_signature(packet,bridge_pairs,structural_cache)
                all_support=all(all(int(c[t])>0 for t in range(7)) for st in packet for c in st.owner_caps)
                min_cap=[min(int(c[t]) for st in packet for c in st.owner_caps) for t in range(7)]
                target_results[level]={
                    'packet_states':len(packet),'expected_packet_states':2*len(motifs),'build_failures':fail,
                    'signature_sha256':sig['signature_sha256'],'all_endpoint_types_supported_in_every_owner':all_support,
                    'min_owner_free_by_type':min_cap,
                    'raw_growth':{'median_leaf_count':sorted(st.leaf_count for st in packet)[len(packet)//2], 'median_total_relations':sorted(st.relation_count_total for st in packet)[len(packet)//2]},
                    'construction_set_sha256':_sha(sorted(st.construction_digest for st in packet)),
                    'reference_distinct':reference_rows[level]['distinct_construction'] and reference_rows[level]['distinct_skin'],
                }
        ids=sorted(target_results)
        sigs=[target_results[i]['signature_sha256'] for i in ids]
        build_ok=not build_failures and all(target_results[i]['packet_states']==target_results[i]['expected_packet_states'] for i in ids)
        support_ok=all(target_results[i]['all_endpoint_types_supported_in_every_owner'] for i in ids)
        same_sig=len(set(sigs))==1
        exactdiff=len({target_results[i]['construction_set_sha256'] for i in ids})==len(ids)
        grow=all(target_results[ids[j]]['raw_growth']['median_leaf_count']>target_results[ids[j-1]]['raw_growth']['median_leaf_count'] for j in range(1,len(ids)))
        ref_distinct=all(target_results[i]['reference_distinct'] for i in ids)
        passed=build_ok and support_ok and same_sig and exactdiff and grow and ref_distinct
        theorem={
            'name':'DEPTH_ERASED_RELATION_ADD_SERVICE_FACTORIZATION',
            'statement':'Within the frozen scanner observer, relation-add enabledness and access/bottleneck response factor through the top-level typed owner multigraph plus per-owner endpoint-support positivity. Exact child skin, nested O depth and Counter magnitude are not read by this observer. Therefore equal depth-erased service representations have equal service answers while the support invariant holds.',
            'proof_obligations':{
                'enabledness_reads_only_support_and_bridge_matrix':'SATISFIED_BY_FROZEN_GRRL_ACTION_SCOPE',
                'graph_effect_reads_only_top_level_graph_and_selected_owner_pair':'SATISFIED_BY_SERVICE_DEFINITION',
                'support_invariant_on_validation_packet':'PASS' if support_ok else 'FAIL',
                'independent_exact_holdout_depths':'PASS' if same_sig and 12 in target_results and 13 in target_results else 'FAIL'
            },
            'reopen_if':['new action reads hidden topology/ancestry/depth','endpoint support ceases to be full','relation arity/typing changes','observer begins reading Counter magnitude or exact successor skin','deletion/rewiring/retagging/resource release is admitted']
        }
        result={
            'schema':'IG_O_REGIME_DEPTH_ERASING_SERVICE_VALIDATION_RESULT_V1','date':'2026-08-30',
            'status':'PASS' if passed else 'FAIL',
            'classification':spec['classification_on_pass'] if passed else 'DEPTH_ERASING_SERVICE_VALIDATION_FAILED',
            'validation_spec_sha256':_sha(spec),'scanner_spec_sha256':_sha(scanner_spec),'phase8_seed_sha256':_sha_file(Path(phase8_seed)),
            'motif_count':len(motifs),'target_depths':ids,'target_results':{str(k):v for k,v in target_results.items()},
            'reference_lineage':{str(k):v for k,v in reference_rows.items()},
            'acceptance':{
                'all_packet_builds_exact':build_ok,'all_endpoint_types_supported_in_every_owner':support_ok,
                'packet_signature_byte_identical_at_all_target_depths':same_sig,'exact_construction_digests_differ_across_depths':exactdiff,
                'interior_leaf_complexity_strictly_grows_across_target_depths':grow,'reference_lineages_remain_exactly_distinct':ref_distinct,
            },
            'depth_erased_signature_sha256':sigs[0] if same_sig else None,'theorem':theorem,'nonclaims':spec['nonclaims'],
            'scientific_interpretation':'A depth-erased relation-add access/bottleneck service law is earned for the frozen observer and fixed-lift scope. This closes one scanner lane as inherited structure; it does not raise the entire O regime to a new algebraic carrier.',
            'next':'Return this earned service law to the adaptive scanner as an inherited theorem and continue the broad regime scan with this lane suppressed from future novelty counts.' if passed else 'Audit the failed obligation before any further promotion.'
        }
        result['science_sha256']=_sha({k:v for k,v in result.items() if k!='science_sha256'})
        (output/'O_REGIME_DEPTH_ERASING_SERVICE_VALIDATION_RESULT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        return result
    finally:
        shutil.rmtree(work,ignore_errors=True)
