"""Bounded two-component qualification constructor, preserving endpoint choices.

This creates rooted constructions; anonymous boundaries are explicit projections.
It does not implement historical frontier selection or claim mature-node coverage.
"""
from pathlib import Path
import hashlib,json
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult

IDENTITY='IG_NODE_ATTACHMENT_ROOTED_CONSTRUCTION_V1'
PROJECTION='SORT_PORT_PM_AND_DESTINATIONS_WITH_EXPLICIT_PREIMAGE_V1'

def project(record):
    order=sorted(range(len(record[1])),key=lambda i:(record[1][i],i))
    destinations=record[3] if isinstance(record[3],list) else [record[3]]
    targets=sorted(range(len(destinations)),key=lambda i:(destinations[i],i))
    return [record[0],[record[1][i] for i in order],record[2],[destinations[i] for i in targets],record[4]],order,targets

def evaluate(payload):
    left=payload['left'];right=payload['right'];states=[];attempts=0
    for sa in range(3):
        for sb in range(3):
            attempts+=1;a=left['record'];b=right['record'];pa,ma=a[1][sa];pb,mb=b[1][sb]
            if not ((ma==0 or ma&pb) and (mb==0 or mb&pa)):continue
            external=[['seed',i] for i in range(3) if i!=sa]+[['attachment',i] for i in range(3) if i!=sb]
            ordered=[a[0]|b[0],[a[1][i] for i in range(3) if i!=sa]+[b[1][i] for i in range(3) if i!=sb],min(a[2],b[2]),[a[3],b[3]],int(a[4] and b[4])]
            projected,ports,targets=project(ordered)
            components=[{'role':role,'j3_id':x['j3_id'],'event_id':x['event_id'],'historical_selector':x['selector'],'option_realizations_ref':x['event_id']} for role,x in [('seed',left),('attachment',right)]]
            identity={'schema_id':IDENTITY,'recipe_sha256':payload['recipe_sha256'],'components':components,'bridge_slots':[sa,sb]}
            oid=canonical_sha256(identity)
            state={'object_id':oid,'identity':identity,'ordered_record':ordered,'projected_boundary':projected,'projection_profile':PROJECTION,'projected_port_to_ordered_port':ports,'projected_target_to_component':targets,'external_port_origins':external,'bridge':[['seed',sa],['attachment',sb]],'formation_id':canonical_sha256({'schema_id':'IG_NODE_ATTACHMENT_FORMATION_V1','identity':identity}),'components':components,'microscopic_realization_product_count':left['realization_count']*right['realization_count']}
            states.append({'identity':identity,'state':state})
    return {'states':states,'metrics':{'attempted_endpoint_pairs':attempts,'lawful_endpoint_pairs':len(states)}}

def handler(stage,runtime):
    params=stage['execution']['parameters'];inputs=stage['input_artifacts']
    raw=Path(inputs['primitive_packet']).read_bytes();recipe_raw=Path(inputs['recipe_contract']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=params['input_sha256'] or hashlib.sha256(recipe_raw).hexdigest()!=params['recipe_sha256']:raise ValueError('NODE_INPUT_HASH')
    packet=json.loads(raw);recipe=json.loads(recipe_raw)
    if recipe['scope']!={'attachment_depth':1,'ordered_roles':['seed','attachment'],'primitive_selector_count':9,'native_event_count':13,'ports_per_input':3,'attempted_endpoint_pairs':1521,'selection':'EXHAUSTIVE_WITHIN_DECLARED_INPUT_ALPHABET'}:raise ValueError('NODE_SCOPE')
    if packet['accepted_catalog_sha256']!=params['accepted_catalog_sha256'] or len(packet['events'])!=13:raise ValueError('NODE_INPUT_BINDING')
    tasks=[]
    for a in packet['events']:
        for b in packet['events']:
            payload={'left':a,'right':b,'recipe_sha256':params['recipe_sha256']}
            tasks.append(TaskSpec(a['event_id']+'_'+b['event_id'],'NODE_ATTACHMENT_DEPTH1',canonical_sha256(payload),payload))
    result=runtime.run_content_indexed_generation(phase_id='NODE_DEPTH1',tasks=tasks,evaluator_ref='project.node_attachment:evaluate',max_tasks=169,max_generated_occurrences=1521,max_task_occurrences=9,max_result_bytes=65536,max_work_seconds=None)
    rows=[];seen=set()
    for g in runtime.iter_generated_states(phase_id='NODE_DEPTH1'):
        row=g['state']
        if g['occurrence_count']!=1 or row['object_id'] in seen:raise ValueError('NODE_OCCURRENCE_COLLAPSE')
        seen.add(row['object_id']);rows.append(row)
    rows.sort(key=lambda r:r['object_id'])
    if len(rows)!=recipe['qualification_expected_lawful_occurrences']:raise ValueError('NODE_COVERAGE')
    shards=[]
    for start in range(0,len(rows),256):
        value={'schema_id':'IG_NODE_ATTACHMENT_QUALIFICATION_SHARD_V1','recipe_sha256':params['recipe_sha256'],'primitive_packet_sha256':params['input_sha256'],'rows':rows[start:start+256]}
        if len(canonical_bytes(value))>1048576:raise ValueError('NODE_SHARD_LIMIT')
        shards.append(runtime.publish_bulk_shard(f'NODE_ROWS_{start//256:03d}',value))
    return ChainExecutionResult(result={'outcome':'PASS','scope':'NODE_ATTACHMENT_DEPTH1_QUALIFICATION','rooted_constructions':len(rows),'formation_occurrences':len(rows),'projected_boundaries':len({canonical_sha256(r['projected_boundary']) for r in rows}),'attempted_endpoint_pairs':1521,'data_shards':shards,'official_master_admission':False,'historical_mature_node_census_reproduced':False,'existing_j3_regenerated':0,'generation':result.summary})
