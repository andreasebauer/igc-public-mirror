"""Current R100 construction from the accepted level13 prefix, under native SCRIPT."""
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import time

from infinity_grid import maturation_parallel as mp
from infinity_grid import regime_scanner as rs
from infinity_grid import r100_reuse
from infinity_grid import carrier_reuse
from infinity_grid import v2_joint_candidate_consumer as joint
from infinity_grid import semantic_sentinel
from infinity_grid import theorem_registry
from infinity_grid.canon import canonical_sha256

def write(name,obj):
    Path(name).write_text(json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n')

def progress(event,**fields):
    row={'event':event,**fields}
    with Path('PROGRESS.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps(row,sort_keys=True),flush=True)

def plain(value):return json.loads(json.dumps(value,sort_keys=True))
def digest(value):return canonical_sha256(plain(value))

def construct(engine,previous,level,pairs,motifs,spec):
    # Unchanged native scientific functions, same literal route as the accepted
    # V2 calibration. No shared-task guard bypass or project-created worker pool.
    prev=sorted(previous,key=rs.execution_order_key)
    median=statistics.median(sum(s.total_caps) for s in prev)
    center=min(prev,key=lambda s:(abs(sum(s.total_caps)-median),rs.execution_order_key(s)))
    recipes=mp.enumerate_candidate_recipes(motifs,center_supports_twin=center.total_caps[0]>=6,pairs_count=len(pairs))
    assert len(recipes)==193
    mp.install_candidate_context(engine=engine,prev=prev,level=level,pairs=pairs,motifs=motifs,center=center)
    try:rows=[mp._candidate_worker({'recipe':recipe}) for recipe in recipes]
    finally:mp.clear_candidate_context()
    assert all(row['success'] for row in rows) and len(rows)==193
    assert all(row['execution_policy']=='NATIVE_V3' for row in rows)
    assert len({row['construction_digest'] for row in rows})==193
    selected=mp._farthest_select_descriptors(rows,int(spec['panel']['beam']),must_include_motif_ids={'TWIN:A','TWIN:B'})
    # Rebuild the complete candidate cohort and compare the independent live-state
    # selector with the descriptor selector. All retained recipe digests are checked
    # by the unchanged native rebuild function, not inferred from count metadata.
    full=mp.rebuild_selected_states(rows,engine=engine,prev=prev,level=level,pairs=pairs,motifs=motifs,center=center)
    twins=[next(s for s in full if s.motif_id==name) for name in ('TWIN:A','TWIN:B')]
    live_selected=rs._farthest_select(full,int(spec['panel']['beam']),must_include=twins)
    assert [s.construction_digest for s in live_selected]==[row['construction_digest'] for row in selected]
    by_digest={s.construction_digest:s for s in full}
    chosen=[by_digest[row['construction_digest']] for row in selected]
    assert len(chosen)==24 and all(s.level==level for s in chosen)
    return full,chosen,{'level':level,'recipes':193,'successful':193,'unique_candidates':193,
        'selected':24,'descriptor_live_selector_equal':True,
        'ordered_candidate_digests':[s.construction_digest for s in full],
        'ordered_selected_digests':[s.construction_digest for s in chosen],
        'candidate_caps7_sha256':digest([[s.construction_digest,list(s.total_caps)] for s in full])}

def main():
    started=time.monotonic();cpu_started=time.process_time()
    paths=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,sha in params['input_hashes'].items():
        assert hashlib.sha256(Path(paths[role]).read_bytes()).hexdigest()==sha
    protocol=json.loads(Path(paths['protocol']).read_text())
    original=json.loads(Path(paths['binding']).read_text())
    declaration=json.loads(Path(paths['compatibility']).read_text())
    data_root=Path('candidate_inputs');data_root.mkdir()
    for role,pin in original['artifacts'].items():shutil.copyfile(paths[role],data_root/pin['file_name'])
    joint.preflight_joint(data_root,original,declaration,Path(paths['producer_registry']).read_bytes())
    baseline=semantic_sentinel.verify_native_semantics(raise_on_change=False)
    theorems=theorem_registry.verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    historical=json.loads(gzip.decompress(Path(paths['historical_bootstrap']).read_bytes()))
    old=historical['dag'];assert historical['level']==100 and len(old['roots'])==193
    assert old['schema_id']=='IG_MATURATION_STATE_DAG_V1' and 'execution_policy' not in old
    assert all('execution_policy' not in row for row in old['nodes'].values())
    write('HISTORICAL_SCOPE_REVIEW.json',{'status':'RETAINED_AS_HISTORICAL_ONLY','roots':193,
        'level':100,'nodes':len(old['nodes']),'schema_id':old['schema_id'],'current_policy_present':False,
        'old_bytes_relabelled':False,'old_observer_jobs_replayed':False,
        'bootstrap_sha256':params['input_hashes']['historical_bootstrap']})
    del historical,old
    dag=json.loads(Path(paths['dag']).read_text());index=json.loads(Path(paths['index']).read_text())
    engine,loaded=mp._states_from_dag(dag)
    previous=[state for state,row in zip(loaded,index['roots']) if row['domain']=='selected' and row['level']==13]
    assert len(previous)==24 and all(s.execution_policy=='NATIVE_V3' and s.level==13 for s in previous)
    for state,row in zip(loaded,index['roots']):
        assert digest(mp.state_wire(state))==row['public_state_sha256']
    del dag,loaded,index
    motifs=rs.load_motif_library();spec=rs.load_regime_scanner_spec()
    seed=json.loads(Path(paths['streaming_seed']).read_text());pairs=[tuple(pair) for pair in seed['bridge_pairs']]
    assert int(spec['panel']['candidate_cap'])==protocol['frozen_candidate_cap'] and protocol['complete_candidate_count_per_level']<=int(spec['panel']['candidate_cap']) and int(spec['panel']['beam'])==24
    levels=[];progress('qualified_prefix_loaded',level=13,selected=24,old_levels_rebuilt=0)
    for level in range(14,101):
        level_started=time.monotonic()
        full,previous,row=construct(engine,previous,level,pairs,motifs,spec)
        row['wall_seconds']=time.monotonic()-level_started;levels.append(row)
        write('CONTINUATION_LEVELS.json',{'status':'RUNNING','completed_through':level,'levels':levels})
        progress('current_level_completed',level=level,candidates=193,selected=24,wall_seconds=row['wall_seconds'])
        if level!=100:del full
    states=sorted(full,key=lambda s:s.construction_digest)
    assert len(states)==193 and len(levels)==87 and all(s.level==100 and s.execution_policy=='NATIVE_V3' for s in states)
    cohort={'schema_id':'IG_CURRENT_R100_CONSTRUCTION_COHORT_V1','scope':'CURRENT_NATIVE_V3_LITERAL_CONSTRUCTION_ONLY',
        'level':100,'root_count':193,'qualified_prefix_level':13,'missing_levels_constructed':87,
        'recipes_per_level':193,'recipe_evaluations':87*193,'selector_checks':87,
        'ordered_root_digests':[s.construction_digest for s in states],
        'public_descriptors':[r100_reuse.public_descriptor(s) for s in states],
        'source_engine_sha256':protocol['engine_source_sha256'],'protocol_sha256':params['input_hashes']['protocol'],
        'prefix_dag_sha256':params['input_hashes']['dag'],'prefix_index_sha256':params['input_hashes']['index'],
        'production_authority':'NONE','theorems_adopted':False,'current_campaign_admitted':False}
    cohort['science_sha256']=digest(cohort);write('COHORT_AUTHORITY.json',cohort)
    binding={'source_sha256':protocol['engine_source_sha256'],'registry_sha256':protocol['registry_sha256'],
             'producer_experiment_id':'MASTER.STEP3.CURRENT.R100.Q02','authority_stage':'G1:R100:CURRENT_NATIVE_V3_LITERAL_CONSTRUCTION',
             'authority_science_sha256':cohort['science_sha256']}
    saved_root=Path('current_r100');saved_root.mkdir()
    manifest=carrier_reuse.save_certified_carrier_set(saved_root,
        carriers={s.construction_digest:(s,r100_reuse.public_descriptor(s)) for s in states},
        **binding,certification_status='CERTIFIED_PASS')
    assert manifest['carrier_count']==193 and manifest['exact_payload_schema']=='IG_MATURATION_STATE_DAG_V1'
    write('PRODUCER_BINDING.json',binding)
    admission=r100_reuse.preflight(saved_root,binding)
    restored=r100_reuse.load(admission)
    assert [s.construction_digest for s in restored]==cohort['ordered_root_digests']
    assert [r100_reuse.public_descriptor(s) for s in restored]==cohort['public_descriptors']
    reserialized=mp.state_dag_wire(restored)
    saved_dag=json.loads((saved_root/'EXACT_STATE_DAG_V1.json').read_text())
    assert reserialized==saved_dag and saved_dag['execution_policy']=='NATIVE_V3'
    write('SAVED_READER_RESULT.json',{'status':'PASS','complete_root_count':193,'public_descriptor_matches':193,
        'ordered_roots_equal':True,'exact_dag_roundtrip_equal':True,'dag_nodes':len(saved_dag['nodes']),
        'dag_raw_sha256':hashlib.sha256((saved_root/'EXACT_STATE_DAG_V1.json').read_bytes()).hexdigest(),
        'manifest_raw_sha256':hashlib.sha256((saved_root/'CERTIFIED_CARRIER_SET_V1.json').read_bytes()).hexdigest(),
        'producer_binding_sha256':digest(binding),'production_authority':'NONE','actual_campaign_started':False})
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==baseline
    assert theorem_registry.verify_all_theorems(raise_on_stale=False)==theorems
    write('GUARD_STATUS.json',{'status':'PASS','semantic_status':'REOPEN_REQUIRED','stale_theorems':4,
        'baseline_unchanged':True,'production_generation_allowed':False,'current_campaign_admitted':False,
        'parent_congruence_proven':False,'complete_counter_correspondence_proven':False,'exact_read_premises_closed':False,
        'baseline_adopted':False,'theorems_adopted':False})
    write('CONTINUATION_LEVELS.json',{'status':'PASS','completed_through':100,'levels':levels})
    write('QUALIFICATION_RESULT.json',{'schema_id':'IG_CURRENT_R100_CREATION_AND_READER_QUALIFICATION_V1','status':'PASS',
        'scientific_outcome':'CURRENT_R100_CREATION_AND_SAVED_READER_QUALIFIED_NO_PRODUCTION_AUTHORITY',
        'current_roots':193,'qualified_prefix_level':13,'new_levels':87,'recipe_evaluations':16791,
        'selector_checks':87,'saved_public_matches':193,'exact_dag_roundtrip_equal':True,
        'historical_roots_relabelled':False,'closed_calibration_jobs_replayed':False,'new_marker_tasks':0,
        'independent_counter_oracle_replays':0,'production_generation_allowed':False,'current_campaign_admitted':False,
        'semantic_status':'REOPEN_REQUIRED','stale_theorems':4,'production_authority':'NONE',
        'wall_seconds':time.monotonic()-started,'cpu_seconds':time.process_time()-cpu_started,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    progress('current_R100_complete',roots=193,dag_nodes=len(saved_dag['nodes']))

if __name__=='__main__':main()
