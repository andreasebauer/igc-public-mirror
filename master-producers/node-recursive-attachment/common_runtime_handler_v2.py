from pathlib import Path
import hashlib,json
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult

def handler(stage,runtime):
    inputs=stage['input_artifacts'];params=stage['execution']['parameters'];raw=Path(inputs['recipe_contract']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=params['recipe_sha256']:raise ValueError('RECIPE_HASH')
    recipe=json.loads(raw)
    for name in ['parent_packet','primitive_packet']:
        if hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()!=recipe[name+'_sha256']:raise ValueError('INPUT_HASH')
    if hashlib.sha256(Path(inputs['dry_recipe']).read_bytes()).hexdigest()!=recipe['identity_recipe_sha256']:raise ValueError('IDENTITY_RECIPE_HASH')
    scope=recipe['scope'];all_parents=json.loads(Path(inputs['parent_packet']).read_text())['rows'];parents=all_parents[scope['parent_start']:scope['parent_stop_exclusive']];packet=json.loads(Path(inputs['primitive_packet']).read_text());events=packet['events']
    if len(parents)!=scope['parents'] or len(events)!=13 or packet['accepted_catalog_sha256']!=recipe['catalog_sha256']:raise ValueError('SCOPE')
    tasks=[]
    for p in parents:
        if len(p['ordered_record'][1])!=4:raise ValueError('PARENT_PORT_COUNT')
        for e in events:
            payload={'parent':p,'attachment':e,'recipe_sha256':recipe['identity_recipe_sha256']}
            tasks.append(TaskSpec(p['object_id']+'_'+e['event_id'],'NODE_RECURSIVE_DEPTH2',canonical_sha256(payload),payload))
    result=runtime.run_content_indexed_generation(phase_id='NODE_RECURSIVE_DEPTH2',tasks=tasks,evaluator_ref='project.recursive_attachment:evaluate',max_tasks=scope['tasks'],max_generated_occurrences=scope['endpoint_attempts'],max_task_occurrences=12,max_result_bytes=131072,max_work_seconds=None)
    rows=[];seen=set();boundaries=set()
    for g in runtime.iter_generated_states(phase_id='NODE_RECURSIVE_DEPTH2'):
        row=g['state']
        if g['occurrence_count']!=1 or row['object_id'] in seen:raise ValueError('OCCURRENCE_COLLAPSE')
        seen.add(row['object_id']);rows.append(row);boundaries.add(canonical_sha256(row['projected_boundary']))
    if len(rows)!=scope['expected_lawful_rooted_constructions'] or canonical_sha256(sorted(boundaries))!=recipe['projected_set_sha256']:raise ValueError('COVERAGE_OR_PROJECTION_MISMATCH')
    rows.sort(key=lambda x:x['object_id']);shards=[]
    for start in range(0,len(rows),1024):
        value={'schema_id':'IG_NODE_RECURSIVE_NATIVE_QUALIFICATION_SHARD_V1','recipe_sha256':params['recipe_sha256'],'identity_recipe_sha256':recipe['identity_recipe_sha256'],'parent_packet_sha256':recipe['parent_packet_sha256'],'rows':rows[start:start+1024]}
        if len(canonical_bytes(value))>min(recipe['resources']['shard_byte_limit'],4*1024*1024):raise ValueError('SHARD_LIMIT')
        shards.append(runtime.publish_bulk_shard('NODE_RECURSIVE_ROWS_'+str(start//1024),value))
    return ChainExecutionResult(result={'outcome':'PASS','rooted_constructions':len(rows),'formation_occurrences':len(rows),'projected_boundaries':len(boundaries),'data_shards':shards,'official_master_admission':False,'historical_l13_reproduced':False,'generation':result.summary})
