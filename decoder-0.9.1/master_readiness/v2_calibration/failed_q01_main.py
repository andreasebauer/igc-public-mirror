"""Prospective finite V2 calibration candidates. Never install authority."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import resource
import statistics
import time

from infinity_grid import regime_scanner as rs, maturation_parallel as mp
from infinity_grid import materialized_discovery as md, fixed_grammar_transport as ft
from infinity_grid import semantic_sentinel as ss, theorem_registry as tr
from infinity_grid.canon import canonical_sha256 as digest
from oracle import summary, owner_maps


def write(name,value):
    Path(name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')

def journal(event,**fields):
    row={'event':event,**fields}
    with Path('PROGRESS.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps(row,sort_keys=True),flush=True)

def plain(value):return json.loads(json.dumps(value,sort_keys=True))

def seal(value):return {**value,'science_sha256':digest(value)}

def probe_state(caps,edges,label):
    caps=tuple(tuple(c) for c in caps);edges=tuple(tuple(e) for e in edges)
    return ft._RecipeState('PROBE',label,tuple((u,v) for u,v,a,b in edges),edges,caps,
        digest([label,caps,edges]),digest(['dummy_skin',label]))


def compare_routes(states,level,pairs,grammar):
    analyses={}
    for state in states:
        assert state.construction_digest not in analyses
        analyses[state.construction_digest]=mp._selected_state_worker({'state':mp.state_wire(state),'bridge_pairs':pairs})
    parallel=mp.scan_level_from_analysis(states,level,pairs,grammar,analyses)
    serial=rs._scan_level(states,level,pairs,grammar)
    if serial!=parallel:
        write('FIRST_ROUTE_MISMATCH.json',{'level':level,'serial':serial,'reducer':parallel})
        raise AssertionError('FULL_SCANNER_ROUTE_MISMATCH')
    assert serial[0]['diagnostic_schema']==rs.ACTION_DIAGNOSTIC_SCHEMA_V2
    assert serial[0]['normalized_signature']['schema_id']==rs.ORGANIZATIONAL_SCHEMA_V2
    return serial[0],analyses


def candidate_descriptors(engine,previous,level,pairs,motifs,spec):
    # Pure engine workers under this saved SCRIPT, without the shared-task service
    # (which correctly requires its own registered native stage caller).
    prev=sorted(previous,key=rs.execution_order_key)
    med=statistics.median(sum(s.total_caps) for s in prev)
    center=min(prev,key=lambda s:(abs(sum(s.total_caps)-med),rs.execution_order_key(s)))
    recipes=mp.enumerate_candidate_recipes(motifs,center_supports_twin=center.total_caps[0]>=6,pairs_count=len(pairs))
    mp.install_candidate_context(engine=engine,prev=prev,level=level,pairs=pairs,motifs=motifs,center=center)
    try:results=[mp._candidate_worker({'recipe':recipe}) for recipe in recipes]
    finally:mp.clear_candidate_context()
    rows=[row for row in results if row['success']]
    cap=int(spec['panel']['candidate_cap'])
    assert len(rows)<=cap, 'FROZEN_193_RECIPE_DOMAIN_EXCEEDS_CAP'
    selected=mp._farthest_select_descriptors(rows,int(spec['panel']['beam']),must_include_motif_ids={'TWIN:A','TWIN:B'})
    assert {row['motif_id'] for row in selected}>={'TWIN:A','TWIN:B'}
    return rows,selected,center,{'recipes':len(recipes),'successful':len(rows),'build_failures':len(results)-len(rows)}


def main():
    started=time.monotonic();cpu_started=time.process_time()
    inputs=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for name,h in params['input_hashes'].items():assert hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()==h
    protocol=json.loads(Path(inputs['protocol']).read_text())
    source=Path(__file__).resolve().parents[1]
    bindings=json.loads(Path(inputs['resource_bindings']).read_text())
    for row in bindings:
        raw=(source/row['path']).read_bytes()
        assert len(raw)==row['size_bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
    old_status=json.loads(Path(inputs['qualified_semantic_status']).read_text())
    semantic=ss.verify_native_semantics(raise_on_change=False)
    theorem=tr.verify_all_theorems(raise_on_stale=False)
    assert semantic['current_snapshot']==old_status['semantic']['current_snapshot']
    assert semantic['status']=='REOPEN_REQUIRED' and theorem['status']=='STALE_REOPEN_REQUIRED'
    assert len(theorem['stale_theorems'])==4
    source_binding={'engine_source_sha256':protocol['engine_source_sha256'],
        'registered_semantics_science_sha256':semantic['current_snapshot']['science_sha256'],
        'resource_bindings':bindings,'protocol_sha256':params['input_hashes']['protocol']}
    spec=rs.load_regime_scanner_spec();motifs=rs.load_motif_library()
    stream_spec=ft.load_streaming_regime_spec();seed=ft.load_streaming_o7_seed()
    pairs=[tuple(pair) for pair in seed['bridge_pairs']]
    grammar=stream_spec['authority']['normalized_grammar_sha256']
    assert grammar==rs.GRAMMAR_EXPECTED
    recipes=mp.enumerate_candidate_recipes(motifs,center_supports_twin=True,pairs_count=len(pairs))
    assert len(recipes)==protocol['expected_recipe_count']
    pool=[];catalogue=[];independent=[];org_by_recipe={}
    for index,recipe in enumerate(recipes):
        label=recipe['lane']+'|'+recipe['motif_id']
        forced=None if recipe['force_pair'] is None else tuple(recipe['force_pair'])
        state=ft._typed_recipe_state(recipe['lane'],recipe['motif_id'],recipe['n'],
            [tuple(e) for e in recipe['edges']],pairs,recipe['schedule_seed'],force_pair=forced)
        pool.append(state)
        catalogue.append({'recipe_id':label,'owners':recipe['n'],'edges':len(recipe['edges']),
            'typed_edges':[list(e) for e in state.typed_edges]})
        caps=tuple(state.owner_caps);edges=tuple(state.typed_edges)
        expected=summary(caps,edges,tuple(pairs),endpoint_backend='LEFT_ENDPOINT_MULTIPLICITY')
        actual=rs._action_aggregate_v2(state,pairs)
        assert actual==expected, 'INDEPENDENT_RECIPE_SUMMARY:'+label
        actual_group=sorted(tuple(p) for p in rs._automorphisms(len(caps),[tuple(int(x>0) for x in c) for c in caps],list(edges)))
        assert actual_group==list(owner_maps(caps,edges)), 'INDEPENDENT_OWNER_GROUP:'+label
        small=tuple(tuple((owner+t)%4 for t in range(7)) for owner in range(len(caps)))
        explicit=summary(small,edges,tuple(pairs),endpoint_backend='EXPANDED')
        compressed=summary(small,edges,tuple(pairs),endpoint_backend='LEFT_ENDPOINT_MULTIPLICITY')
        assert explicit==compressed==rs._action_aggregate_v2(probe_state(small,edges,'small:'+label),pairs)
        organization=rs._state_organizational_signature_v2(state,actual)
        org_by_recipe[label]=organization
        n=len(caps);perm=list(reversed(range(n)));rcaps=[None]*n;redges=[]
        for old,new in enumerate(perm):rcaps[new]=caps[old]
        for u,v,a,b in edges:
            u,v=perm[u],perm[v]
            redges.append((u,v,a,b) if u<v else (v,u,b,a))
        relabel=probe_state(rcaps,sorted(redges),'relabel:'+label)
        raction=rs._action_aggregate_v2(relabel,pairs)
        assert raction==actual and rs._state_organizational_signature_v2(relabel,raction)==organization
        scaled=probe_state([[3*x for x in c] for c in caps],edges,'scaled:'+label)
        saction=rs._action_aggregate_v2(scaled,pairs)
        assert saction['total_action_copies']==9*actual['total_action_copies']
        assert all(saction[k]==v for k,v in actual.items() if k!='total_action_copies')
        assert rs._state_organizational_signature_v2(scaled,saction)==organization
        independent.append({'recipe_id':label,'input_sha256':digest([caps,edges,pairs]),
            'independent_summary':expected,'independent_group_sha256':digest(actual_group),
            'summary_equal':True,'exact_group_equal':True,'endpoint_backends_equal_on_small_control':True,
            'relabel_equal':True,'scaled_copies_factor':9,'scaled_organization_equal':True,
            'organizational_sha256':digest(organization)})
        if (index+1)%25==0:journal('recipe_independent_checks',completed=index+1,total=len(recipes))
    assert digest(catalogue)==protocol['expected_recipe_catalogue_sha256']
    write('INDEPENDENT_RECIPE_VERIFICATION.json',{'status':'PASS','recipe_count':len(pool),
        'independent_large_checks':len(independent),'independent_backend_controls':len(independent),
        'relabel_controls':len(independent),'magnitude_controls':len(independent),'recipes':independent})
    static_scan,_=compare_routes(pool,11,pairs,grammar);route_matches=1
    lanes=('branching','symmetry','lineage','topology_services')
    lane_payloads={lane:static_scan['normalized_signature'][lane] for lane in lanes}
    lane_hashes={lane:digest(value) for lane,value in lane_payloads.items()}
    baseline=seal({'schema_id':'IG_STREAMING_RECIPE_V2_OBSERVER_BASELINE_CANDIDATE_V1',
        'status':protocol['candidate_status'],'diagnostic_schema':rs.ACTION_DIAGNOSTIC_SCHEMA_V2,
        'organizational_schema':rs.ORGANIZATIONAL_SCHEMA_V2,'source_binding':source_binding,
        'recipe_count':len(pool),'recipe_catalogue_sha256':digest(catalogue),'recipe_catalogue':catalogue,
        'lane_hashes':lane_hashes,'normalized_lane_payloads':lane_payloads,'full_recipe_scan':plain(static_scan),
        'resource_population_materialized':False,'authority_adopted':False})
    write('STREAMING_RECIPE_BASELINE_CANDIDATE_V3.json',baseline)
    journal('static_baseline_computed',recipe_count=len(pool),lane_hashes=lane_hashes)
    # Raw construction is an experimental finite source qualification, not a
    # production discovery session or an inherited theorem/calibration consumer.
    with md._DISCOVERY_SESSION_LOCK:
        runtime=md._get_process_o7_runtime(spec)
    assert pairs==runtime['bridge_pairs']
    states=list(runtime['base_states']);assert len(states)==protocol['selected_per_level']
    assert all(s.execution_policy=='NATIVE_V3' for s in states)
    old_calibration=json.loads((source/'infinity_grid'/md.TRUST_REPAIR_CALIBRATION_RESOURCE).read_text())
    level_rows=[];full_rows=[];cal_levels={};root_states=[];root_index=[]
    previous=None
    for level in protocol['calibration_levels']:
        full_states=None;build_meta={'source':'EXACT_EMBEDDED_O7_COMPACT_INPUT','selected':len(states)}
        if level>7:
            descriptors,selected,center,build_meta=candidate_descriptors(runtime['engine'],states,level,pairs,motifs,spec)
            desired=descriptors if level in protocol['full_pool_comparison_levels'] else selected
            rebuilt=mp.rebuild_selected_states(desired,engine=runtime['engine'],prev=states,level=level,
                pairs=pairs,motifs=motifs,center=center)
            by_digest={s.construction_digest:s for s in rebuilt};assert len(by_digest)==len(rebuilt)
            states=[by_digest[row['construction_digest']] for row in selected]
            assert len(states)==protocol['selected_per_level']
            if level in protocol['full_pool_comparison_levels']:
                assert len(descriptors)==len(recipes) and build_meta['build_failures']==0
                full_states=rebuilt
        scan,analyses=compare_routes(states,level,pairs,grammar);route_matches+=1
        assert all(s.execution_policy=='NATIVE_V3' for s in states)
        genealogy=rs._classify_genealogy(previous,scan);previous=scan
        hashes={'normalized_signature_sha256':scan['normalized_signature_sha256'],'grammar_sha256':scan['grammar_sha256']}
        for lane in ['diversity','branching','symmetry','lineage','topology_services']:hashes[lane+'_sha256']=digest(scan[lane])
        cal_levels[str(level)]=hashes
        historical=old_calibration['levels'][str(level)]
        level_rows.append({'level':level,'state_count':len(states),'full_routes_equal':True,'build_meta':build_meta,
            'construction_digests':[s.construction_digest for s in states],'hashes':hashes,'genealogy':genealogy,
            'historical_v1_equal_fields':{k:v==historical[k] for k,v in hashes.items()},'scan':plain(scan)})
        for state in states:
            root_index.append({'root_index':len(root_states),'domain':'selected','level':level,
                'construction_digest':state.construction_digest,'public_state_sha256':digest(mp.state_wire(state))})
            root_states.append(state)
        if full_states is not None:
            actual_scan,actual_analyses=compare_routes(full_states,level,pairs,grammar);route_matches+=1
            actual_lanes={lane:actual_scan['normalized_signature'][lane] for lane in lanes}
            assert actual_lanes==lane_payloads, 'FULL_RECIPE_LANES_MISMATCH:O'+str(level)
            matches=[]
            for state in full_states:
                label=state.lane+'|'+state.motif_id
                row=actual_analyses[state.construction_digest]
                assert row['organizational_sha256']==digest(org_by_recipe[label]), 'CONCRETE_DUMMY_ORGANIZATION:'+label
                matches.append({'recipe_id':label,'construction_digest':state.construction_digest,
                    'organizational_sha256':row['organizational_sha256'],'organization_equal':True})
                root_index.append({'root_index':len(root_states),'domain':'full_pool','level':level,
                    'recipe_id':label,'construction_digest':state.construction_digest,'public_state_sha256':digest(mp.state_wire(state))})
                root_states.append(state)
            assert len(matches)==len(recipes) and len({row['recipe_id'] for row in matches})==len(recipes)
            full_rows.append({'level':level,'recipe_count':len(matches),'all_recipe_organizations_equal':True,
                'all_four_lane_payloads_equal':True,'full_routes_equal':True,'lane_hashes':{k:digest(v) for k,v in actual_lanes.items()},
                'recipes':matches,'scan':plain(actual_scan)})
        journal('materialized_level_completed',level=level,selected=len(states),full_pool=0 if full_states is None else len(full_states))
    assert len(root_states)==protocol['expected_state_root_occurrences']
    dag=mp.state_dag_wire(root_states)
    write('QUALIFIED_PANELS_STATE_DAG_V2.json',dag)
    write('STATE_ROOT_INDEX.json',{'schema_id':'IG_V2_CALIBRATION_ROOT_INDEX_V1','root_occurrences':len(root_states),
        'unique_constructions':len(set(dag['roots'])),'roots':root_index,'no_carrier_certification_adopted':True})
    _,restored=mp._states_from_dag(dag)
    assert len(restored)==len(root_states)
    assert all(mp.state_wire(a)==mp.state_wire(b) for a,b in zip(restored,root_states))
    write('STATE_DAG_ROUNDTRIP.json',{'status':'PASS','root_occurrences':len(root_states),
        'unique_constructions':len(set(dag['roots'])),'nodes':len(dag['nodes']),'public_state_equality':True,
        'scope':'Same-process exact native DAG roundtrip, not independent-host cold qualification.'})
    calibration=seal({'schema_id':'IG_O7_O13_V2_CALIBRATION_CANDIDATE_V1','status':protocol['candidate_status'],
        'diagnostic_schema':rs.ACTION_DIAGNOSTIC_SCHEMA_V2,'organizational_schema':rs.ORGANIZATIONAL_SCHEMA_V2,
        'source_binding':source_binding,'execution_policy':'NATIVE_V3','levels':cal_levels,
        'calibration_installed':False,'theorems_adopted':False,'scope':protocol['construction_scope']})
    write('MATERIALIZED_CALIBRATION_CANDIDATE_V2.json',calibration)
    write('MATERIALIZED_LEVEL_AUDIT.json',{'status':'PASS','levels':level_rows,'level_count':len(level_rows),
        'selected_states':sum(row['state_count'] for row in level_rows),'new_observations_not_v1_retagging':True})
    write('FULL_RECIPE_DEPTH_MATCHES.json',{'status':'PASS','depth_count':len(full_rows),
        'recipe_pairs':sum(row['recipe_count'] for row in full_rows),'depths':full_rows})
    after=ss.verify_native_semantics(raise_on_change=False);after_theorem=tr.verify_all_theorems(raise_on_stale=False)
    assert after==semantic and after_theorem==theorem
    for row in bindings:assert hashlib.sha256((source/row['path']).read_bytes()).hexdigest()==row['sha256']
    write('SEMANTIC_AND_THEOREM_STATUS.json',{'status':'PASS','semantic':after,'theorems':after_theorem,
        'registered_semantics_unchanged':True,'frozen_resource_bytes_unchanged':True,'authority_adopted':False})
    assert route_matches==protocol['expected_full_packet_route_matches']
    result={'schema_id':'IG_STEP3_V2_CALIBRATION_QUALIFICATION_RESULT_V1','status':'PASS',
        'scientific_outcome':protocol['scientific_outcome'],'recipe_count':len(recipes),
        'independent_large_recipe_checks':len(independent),'independent_backend_controls':len(independent),
        'relabel_controls':len(independent),'magnitude_controls':len(independent),
        'selected_calibration_levels':len(level_rows),'selected_states':sum(row['state_count'] for row in level_rows),
        'full_recipe_depths':len(full_rows),'paired_recipe_organization_matches':sum(row['recipe_count'] for row in full_rows),
        'full_packet_route_matches':route_matches,'state_root_occurrences':len(root_states),'dag_roundtrip_equal':True,
        'registered_semantics_unchanged':True,'frozen_resource_bytes_unchanged':True,
        'calibration_installed':False,'baseline_adopted':False,'theorems_adopted':False,'current_R100_qualified':False,
        'full_generation_allowed':False,'historical_jobs_replayed':False,'scope':protocol['construction_scope'],
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu_started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write('QUALIFICATION_RESULT.json',result);journal('qualification_completed',**result)
    md._close_process_o7_runtime()


if __name__=='__main__':main()
