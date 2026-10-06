"""Pure recursive rooted attachment adapter; no generation/admission runtime.

Parent IDs and ordered slots are preserved, so projection never replaces lineage.
Caller must bind parent/primitive packets and a recipe before official use.
"""
from canonical_kernel import canonical_sha256
SCHEMA='IG_NODE_RECURSIVE_ROOTED_ATTACHMENT_V1'

def extend(parent,event,sa,sb,recipe_sha256):
    a=parent['ordered_record'];b=event['record']
    if not (0<=sa<len(a[1]) and 0<=sb<len(b[1])):raise ValueError('ENDPOINT_RANGE')
    if len(parent['external_port_origins'])!=len(a[1]):raise ValueError('PARENT_PORT_ORIGINS')
    if canonical_sha256(parent['identity'])!=parent['object_id']:raise ValueError('PARENT_ID')
    ps=parent['identity']['schema_id']
    fs='IG_NODE_ATTACHMENT_FORMATION_V1' if ps=='IG_NODE_ATTACHMENT_ROOTED_CONSTRUCTION_V1' else SCHEMA+'_FORMATION' if ps==SCHEMA else None
    if fs is None or parent['formation_id']!=canonical_sha256({'schema_id':fs,'identity':parent['identity']}):raise ValueError('PARENT_FORMATION_ID')
    pa,ma=a[1][sa];pb,mb=b[1][sb]
    if not ((ma==0 or ma&pb) and (mb==0 or mb&pa)):return None
    ports=[['parent',i] for i in range(len(a[1])) if i!=sa]+[['attachment',i] for i in range(len(b[1])) if i!=sb]
    ordered=[a[0]|b[0],[a[1][i] for i in range(len(a[1])) if i!=sa]+[b[1][i] for i in range(len(b[1])) if i!=sb],min(a[2],b[2]),list(a[3])+[b[3]],int(a[4] and b[4])]
    po=sorted(range(len(ordered[1])),key=lambda i:(ordered[1][i],i));to=sorted(range(len(ordered[3])),key=lambda i:(ordered[3][i],i))
    identity={'schema_id':SCHEMA,'recipe_sha256':recipe_sha256,'parent_object_id':parent['object_id'],'parent_formation_id':parent['formation_id'],'attachment_j3_id':event['j3_id'],'attachment_event_id':event['event_id'],'historical_selector':event['selector'],'bridge_slots':[sa,sb]}
    return {'identity':identity,'object_id':canonical_sha256(identity),'formation_id':canonical_sha256({'schema_id':SCHEMA+'_FORMATION','identity':identity}),'parent_state_sha256':canonical_sha256(parent),'ordered_record':ordered,'projected_boundary':[ordered[0],[ordered[1][i] for i in po],ordered[2],[ordered[3][i] for i in to],ordered[4]],'projected_port_to_ordered_port':po,'projected_target_to_ordered_target':to,'external_port_origins':ports,'bridge':[['parent',sa],['attachment',sb]],'target_origins':[['parent',i] for i in range(len(a[3]))]+[['attachment',0]],'microscopic_realization_product_count':parent['microscopic_realization_product_count']*event['realization_count']}

def evaluate(payload):
    parent=payload['parent'];event=payload['attachment'];states=[];attempts=0
    for sa in range(len(parent['ordered_record'][1])):
        for sb in range(len(event['record'][1])):
            attempts+=1;row=extend(parent,event,sa,sb,payload['recipe_sha256'])
            if row is not None:states.append({'identity':row['identity'],'state':row})
    return {'states':states,'metrics':{'attempted_endpoint_pairs':attempts,'lawful_endpoint_pairs':len(states)}}
