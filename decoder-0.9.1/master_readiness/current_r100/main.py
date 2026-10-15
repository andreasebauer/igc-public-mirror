"""Prospective exact reconstruction/reader job, reusing Q02's completed science."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import os
import resource
import statistics
import time

from infinity_grid import maturation_parallel as mp, regime_scanner as rs
from infinity_grid import r100_reuse, carrier_reuse, semantic_sentinel, theorem_registry
from infinity_grid import v2_joint_candidate_consumer as joint
from infinity_grid.canon import canonical_sha256

def write(name, value):
    Path(name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def progress(event, **fields):
    row={'event':event,**fields}
    with Path('PROGRESS.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    print(json.dumps(row,sort_keys=True),flush=True)

def source_admission(paths, protocol):
    package=Path(mp.__file__).resolve().parent
    declaration=json.loads(Path(paths['compatibility']).read_text())
    literal=json.loads(Path(paths['serializer_delta']).read_text())
    closure=declaration['execution_closure']
    assert joint._files(package)==set(closure) and len(closure)==1050
    changed={Path(item['path']).name:item for item in literal['files']}
    assert set(changed)=={'maturation_parallel.py','maturation_dag_contract.py'}
    def excluding(text,name):
        return [ast.dump(node,include_attributes=False) for node in ast.parse(text).body
                if not (isinstance(node,ast.FunctionDef) and node.name==name)]
    for name,pin in closure.items():
        actual=(package/name).read_bytes()
        if name not in changed:
            assert len(actual)==pin['size_bytes'] and sha(actual)==pin['sha256']
            continue
        item=changed[name];assert sha(actual)==item['new_raw_sha256']
        old=actual.decode()
        for patch in reversed(item['patches']):
            assert old.count(patch['after'])==1
            old=old.replace(patch['after'],patch['before'],1)
        assert sha(old.encode())==item['old_raw_sha256']==pin['sha256'] and len(old.encode())==pin['size_bytes']
        function='state_dag_wire' if name=='maturation_parallel.py' else 'exact_data'
        assert excluding(old,function)==excluding(actual.decode(),function)
    original=json.loads(Path(paths['binding']).read_text())
    try:joint.preflight_joint(Path('.'),original,declaration,Path(paths['producer_registry']).read_bytes())
    except Exception as exc:
        assert str(exc)=='JOINT_EXECUTION_BYTES_CHANGED:maturation_dag_contract.py'
    else:raise AssertionError('ORIGINAL_SOURCE_GUARD_MUST_REFUSE_NEW_SOURCE')
    return {'status':'PASS','changed_package_files':2,'unchanged_package_files':1048,
        'changed_functions':['state_dag_wire','exact_data'],'full_coordinate_equality_retained':True,
        'construction_worker_and_rebuild_AST_unchanged':True,'original_joint_guard_refuses_changed_source':True,
        'historical_producing_binding_relabelled':False,'production_authority':'NONE',
        'changed_file_pins':changed,
        'new_engine_source_sha256':protocol['engine_source_sha256']}

def serializer_controls():
    def fixture(parent_color='A', payload=(1,2)):
        ctx=SimpleNamespace(n=1,source_order=('S',),parents=(SimpleNamespace(exact_color=parent_color,h6=payload),))
        s=rs.O7State(None,ctx,(),(0,)*7,'FIXTURE','SERIALIZER_CONTROL',execution_policy='NATIVE_V3')
        s.__dict__['construction_digest']='a'*64
        return s
    a,b=fixture(),fixture()
    duplicated=mp.state_dag_wire([a,b])
    assert duplicated['roots']==['a'*64]*2 and len(duplicated['nodes'])==1
    refusals=[]
    for other in [fixture(parent_color='B'),fixture(payload=(1,3))]:
        try:mp.state_dag_wire([a,other])
        except ValueError as exc:
            assert str(exc)=='MATURATION_DAG_CONSTRUCTION_ALIAS_COLLISION_REVIEW_REQUIRED'
            refusals.append(str(exc))
        else:raise AssertionError('EXACT_COORDINATE_COLLISION_NOT_REFUSED')
    bad=fixture(payload=(True,2))
    try:mp.state_dag_wire([bad])
    except ValueError as exc:
        assert str(exc)=='MATURATION_DAG_INEXACT_VALUE_bool';refusals.append(str(exc))
    else:raise AssertionError('INEXACT_COORDINATES_NOT_REFUSED')
    from infinity_grid.maturation_dag_contract import exact_data
    leaf=('V',1); shared=(leaf,leaf); assert exact_data(shared) is shared
    mutable=([1],shared); first=exact_data(mutable); mutable[0][0]=2
    second=exact_data(mutable); assert first!=second and first[0]==(1,) and second[0]==(2,)
    return {'status':'PASS','transitive_immutable_tuple_shared':True,'mutable_coordinate_changes_observed':True,
            'repeated_root_occurrences_preserved':True,'refusal_checks':3,
            'refusals':refusals,'synthetic_controls_used_as_R100':False}

def main():
    began=time.monotonic();cpu=time.process_time()
    paths=json.loads(Path(os.environ['DECODER_INPUTS_PATH']).read_text())
    params=json.loads(Path(os.environ['DECODER_PARAMETERS_PATH']).read_text())
    for role,pin in params['input_hashes'].items():assert sha(Path(paths[role]).read_bytes())==pin
    protocol=json.loads(Path(paths['protocol']).read_text())
    compatibility=source_admission(paths,protocol)
    controls=serializer_controls();write('SERIALIZER_CONTROLS.json',controls)
    baseline=semantic_sentinel.verify_native_semantics(raise_on_change=False)
    theorems=theorem_registry.verify_all_theorems(raise_on_stale=False)
    assert baseline['status']=='REOPEN_REQUIRED' and len(theorems['stale_theorems'])==4
    origin=json.loads(Path(paths['q02_native_completion']).read_text())
    assert origin['status']=='RESULT_REJECTED' and origin['result']['exit_code']==-9
    assert origin['completion_sha256']==protocol['q02_completion_sha256']
    assert origin['source_sha256']==protocol['q02_combined_source_sha256']
    for role,artifact in [('retained_levels','script/work/CONTINUATION_LEVELS.json'),
                          ('retained_cohort','script/work/COHORT_AUTHORITY.json')]:
        row=next(x for x in origin['evidence'] if x['path']==artifact)
        assert row['sha256']==params['input_hashes'][role]
        assert row['size_bytes']==Path(paths[role]).stat().st_size
    levels=json.loads(Path(paths['retained_levels']).read_text())
    cohort=json.loads(Path(paths['retained_cohort']).read_text())
    assert levels['status']=='RUNNING' and levels['completed_through']==100
    assert [r['level'] for r in levels['levels']]==list(range(14,101))
    assert all(r['recipes']==r['successful']==r['unique_candidates']==193 and r['selected']==24
               and r['descriptor_live_selector_equal'] for r in levels['levels'])
    assert cohort['root_count']==193 and cohort['level']==100
    assert cohort['source_engine_sha256']==protocol['previous_engine_source_sha256']
    assert cohort['prefix_dag_sha256']==params['input_hashes']['dag']
    assert cohort['science_sha256']==canonical_sha256({k:v for k,v in cohort.items() if k!='science_sha256'})
    dag=json.loads(Path(paths['dag']).read_text());index=json.loads(Path(paths['index']).read_text())
    engine,loaded=mp._states_from_dag(dag)
    assert len(loaded)==747
    assert all(canonical_sha256(mp.state_wire(s))==r['public_state_sha256'] for s,r in zip(loaded,index['roots']))
    prefix_wire=mp.state_dag_wire(loaded)
    assert prefix_wire==dag
    compatibility.update(prefix_root_occurrences=747,prefix_dag_exactly_equal=True)
    write('SOURCE_COMPATIBILITY.json',compatibility)
    previous=[s for s,r in zip(loaded,index['roots']) if r['domain']=='selected' and r['level']==13]
    assert len(previous)==24
    del loaded,index,dag,prefix_wire
    motifs=rs.load_motif_library();spec=rs.load_regime_scanner_spec()
    seed=json.loads(Path(paths['streaming_seed']).read_text());pairs=[tuple(x) for x in seed['bridge_pairs']]
    assert spec['panel']['candidate_cap']==220 and spec['panel']['beam']==24
    rebuilt_count=0;restoration=[]
    for record in levels['levels']:
        level=record['level'];prev=sorted(previous,key=rs.execution_order_key)
        median=statistics.median(sum(s.total_caps) for s in prev)
        center=min(prev,key=lambda s:(abs(sum(s.total_caps)-median),rs.execution_order_key(s)))
        recipes=mp.enumerate_candidate_recipes(motifs,center_supports_twin=center.total_caps[0]>=6,pairs_count=len(pairs))
        assert len(recipes)==193
        by_id={d:{'recipe':recipe,'construction_digest':d,'execution_policy':'NATIVE_V3',
                  'execution_order_key':d,'motif_id':recipe['motif_id']}
               for d,recipe in zip(record['ordered_candidate_digests'],recipes)}
        desired=record['ordered_candidate_digests'] if level==100 else record['ordered_selected_digests']
        rows=[by_id[d] for d in desired]
        states=mp.rebuild_selected_states(rows,engine=engine,prev=prev,level=level,pairs=pairs,motifs=motifs,center=center)
        assert [s.construction_digest for s in states]==desired
        rebuilt_count+=len(states)
        restoration.append({'level':level,'exact_saved_digest_matches':len(states)})
        previous=states
        progress('retained_level_reconstructed',level=level,states=len(states),new_recipe_workers=0,new_selector_checks=0)
    states=sorted(states,key=lambda s:s.construction_digest)
    assert rebuilt_count==2257 and [s.construction_digest for s in states]==cohort['ordered_root_digests']
    public=[r100_reuse.public_descriptor(s) for s in states]
    assert public==cohort['public_descriptors']
    write('RECONSTRUCTION_RESULT.json',{'status':'PASS','retained_level_records':87,
        'retained_recipe_evaluations':16791,'retained_selector_checks':87,'reconstructed_states':2257,
        'new_recipe_worker_calls':0,'new_selector_checks':0,'current_roots':193,'public_matches':193,
        'origin_whole_job_status':'RESULT_REJECTED','origin_completed_subsections_reused':True,
        'level_reconstruction':restoration,'production_authority':'NONE'})
    progress('exact_R100_serialization_started',roots=193)
    authority={'schema_id':'IG_CURRENT_R100_RECONSTRUCTION_READER_AUTHORITY_V1','root_count':193,'level':100,
        'scope':'EXACT_SAVED_RECORD_RECONSTRUCTION_AND_READER_ONLY','producing_engine_source_sha256':protocol['engine_source_sha256'],
        'original_literal_cohort_raw_sha256':params['input_hashes']['retained_cohort'],
        'original_construction_engine_source_sha256':cohort['source_engine_sha256'],
        'original_construction_science_sha256':cohort['science_sha256'],
        'q02_capture_id':protocol['q02_capture_id'],'q02_completion_sha256':protocol['q02_completion_sha256'],
        'retained_record_reconstructed_states':2257,'old_observers_replayed':False,'current_campaign_admitted':False,
        'production_authority':'NONE'}
    authority['science_sha256']=canonical_sha256(authority);write('COHORT_AUTHORITY.json',authority)
    binding={'source_sha256':protocol['engine_source_sha256'],'registry_sha256':protocol['registry_sha256'],
        'producer_experiment_id':'MASTER.STEP3.CURRENT.R100.Q05','authority_stage':'G1:R100:CURRENT_NATIVE_V3_RECONSTRUCTION_AND_READER',
        'authority_science_sha256':authority['science_sha256']}
    saved=Path('current_r100');saved.mkdir()
    carrier_reuse.save_certified_carrier_set(saved,carriers={s.construction_digest:(s,p) for s,p in zip(states,public)},
        **binding,certification_status='CERTIFIED_PASS')
    write('PRODUCER_BINDING.json',binding)
    progress('exact_R100_saved',size_bytes=(saved/'EXACT_STATE_DAG_V1.json').stat().st_size)
    restored=r100_reuse.load(r100_reuse.preflight(saved,binding))
    assert [s.construction_digest for s in restored]==cohort['ordered_root_digests']
    assert [r100_reuse.public_descriptor(s) for s in restored]==public
    exact=json.loads((saved/'EXACT_STATE_DAG_V1.json').read_text())
    assert mp.state_dag_wire(restored)==exact
    write('SAVED_READER_RESULT.json',{'status':'PASS','complete_root_count':193,'public_descriptor_matches':193,
        'ordered_roots_equal':True,'exact_dag_roundtrip_equal':True,'dag_nodes':len(exact['nodes']),
        'dag_raw_sha256':sha((saved/'EXACT_STATE_DAG_V1.json').read_bytes()),
        'manifest_raw_sha256':sha((saved/'CERTIFIED_CARRIER_SET_V1.json').read_bytes()),
        'production_authority':'NONE','actual_campaign_started':False})
    assert semantic_sentinel.verify_native_semantics(raise_on_change=False)==baseline
    assert theorem_registry.verify_all_theorems(raise_on_stale=False)==theorems
    write('GUARD_STATUS.json',{'status':'PASS','semantic_status':'REOPEN_REQUIRED','stale_theorems':4,
        'baseline_unchanged_within_capture':True,'production_generation_allowed':False,'current_campaign_admitted':False,
        'parent_congruence_proven':False,'complete_counter_correspondence_proven':False,'exact_read_premises_closed':False,
        'baseline_adopted':False,'theorems_adopted':False})
    write('QUALIFICATION_RESULT.json',{'status':'PASS',
        'scientific_outcome':'CURRENT_R100_SAVED_READER_AND_BOUNDED_SERIALIZER_QUALIFIED_NO_PRODUCTION_AUTHORITY',
        'current_roots':193,'retained_levels':87,'retained_recipe_evaluations':16791,'retained_selector_checks':87,
        'reconstructed_states':2257,'new_recipe_worker_calls':0,'new_selector_checks':0,'saved_public_matches':193,
        'exact_dag_roundtrip_equal':True,'prefix_dag_exactly_equal':True,'prefix_roots':747,'serializer_refusal_checks':3,
        'historical_roots_relabelled':False,'closed_observer_jobs_replayed':False,'new_marker_tasks':0,
        'independent_counter_oracle_replays':0,'production_generation_allowed':False,'current_campaign_admitted':False,
        'semantic_status':'REOPEN_REQUIRED','stale_theorems':4,'production_authority':'NONE',
        'wall_seconds':time.monotonic()-began,'cpu_seconds':time.process_time()-cpu,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    progress('current_R100_complete',roots=193,dag_nodes=len(exact['nodes']))

if __name__=='__main__':main()
