"""Registered export of an unchanged preserved native generation store."""
from pathlib import Path
import hashlib,json,zipfile,sqlite3
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.v05_chain import ChainExecutionResult

def handler(stage,runtime):
    inputs=stage['input_artifacts'];params=stage['execution']['parameters'];raw=Path(inputs['recipe_contract']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=params['recipe_sha256']:raise ValueError('RECIPE_HASH')
    recipe=json.loads(raw);binding=json.loads(Path(inputs['generation_binding']).read_text())
    for name in ['parent_packet','primitive_packet']:
        if hashlib.sha256(Path(inputs[name]).read_bytes()).hexdigest()!=recipe[name+'_sha256']:raise ValueError('INPUT_HASH')
    state=Path(inputs['generation_state']).read_bytes()
    if hashlib.sha256(state).hexdigest()!=binding['checkpoint_object_sha256']:raise ValueError('STATE_OBJECT_HASH')
    with zipfile.ZipFile(inputs['generation_state']) as z:raw_db=z.read(binding['database_member'])
    if hashlib.sha256(raw_db).hexdigest()!=binding['database_sha256']:raise ValueError('DATABASE_HASH')
    c=sqlite3.connect(':memory:');c.deserialize(raw_db);del raw_db;c.execute('PRAGMA query_only=ON')
    scope=json.loads(c.execute("SELECT v FROM meta WHERE k='scope'").fetchone()[0])
    if scope!=binding['generation_scope']:raise ValueError('GENERATION_SCOPE')
    parents=json.loads(Path(inputs['parent_packet']).read_text())['rows'][recipe['scope']['parent_start']:recipe['scope']['parent_stop_exclusive']];events=json.loads(Path(inputs['primitive_packet']).read_text())['events'];expected={}
    for p in parents:
        for e in events:
            payload={'parent':p,'attachment':e,'recipe_sha256':recipe['identity_recipe_sha256']};expected[p['object_id']+'_'+e['event_id']]=canonical_sha256(payload)
    actual=dict(c.execute('SELECT task_id,payload_sha256 FROM generation_tasks'))
    if actual!=expected:raise ValueError('TASK_COVERAGE_OR_PAYLOAD_BINDING')
    rows=[];shards=[];bs=set();count=0;byte_count=0
    def publish():
        value={'schema_id':'IG_NODE_RECURSIVE_NATIVE_QUALIFICATION_SHARD_V1','recipe_sha256':params['recipe_sha256'],'identity_recipe_sha256':recipe['identity_recipe_sha256'],'parent_packet_sha256':recipe['parent_packet_sha256'],'rows':list(rows)}
        if len(canonical_bytes(value))>4*1024*1024:raise ValueError('SHARD_LIMIT')
        shards.append(runtime.publish_bulk_shard('NODE_RECURSIVE_ROWS_'+str(len(shards)),value))
    for identity_bytes,state_json,occurrences in c.execute('SELECT canonical_bytes,state_json,occurrence_count FROM states ORDER BY index_digest,class_index'):
        row=json.loads(state_json)
        if occurrences!=1 or bytes(identity_bytes)!=canonical_bytes(row['identity']) or canonical_sha256(row['identity'])!=row['object_id']:raise ValueError('IDENTITY_OR_MULTIPLICITY')
        size=len(canonical_bytes(row))+1
        if rows and byte_count+size>4*1024*1024-4096:publish();rows=[];byte_count=0
        rows.append(row);byte_count+=size;count+=1;bs.add(canonical_sha256(row['projected_boundary']))
    if rows:publish()
    if count!=recipe['scope']['expected_lawful_rooted_constructions'] or canonical_sha256(sorted(bs))!=recipe['projected_set_sha256']:raise ValueError('RESULT_COVERAGE')
    c.close()
    return ChainExecutionResult(result={'outcome':'PASS','rooted_constructions':count,'formation_occurrences':count,'projected_boundaries':len(bs),'data_shards':shards,'official_master_admission':False,'historical_l13_reproduced':False,'generation':{'preserved_generation_tasks_reused':len(actual),'new_endpoint_generation_tasks':0,'parent_capture_id':binding['failed_capture_id']},'export_recovery':'REPACK_PRESERVED_NATIVE_STATE_WITHOUT_REGENERATION'})
