"""Bind an already captured native DAG to byte-verified lower inputs.

Consumes state_dag_wire output; never imports Decoder or materializes states.
Compressed DAG references are lossless: logical occurrences are root ordinal
plus ordered child-slot path, not a count of unique stored DAG nodes.
"""
import hashlib
import json
from marker_full_export import exact_json, require


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(exact_json(obj), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def assemble_g3_dag_export(*, dag, binding, required_inputs, input_bytes,
                           authority_source_ids):
    require(type(binding) is dict and set(binding) == {'recipe_ref','source_ref','runtime_ref','input_root_ref','task_ref','identity_scope'}, 'DAG binding fields')
    require(all(type(v) is str and v for v in binding.values()), 'empty DAG binding')
    scopes={'DECODER091_NATIVE_V3':'NATIVE_V3','DECODER091_HISTORICAL_COORDINATE_V1':'HISTORICAL_COORDINATE_V1'}
    require(binding['identity_scope'] in scopes, 'unregistered identity scope')
    policy=scopes[binding['identity_scope']]
    require(type(dag) is dict and dag.get('schema_id')=='IG_MATURATION_STATE_DAG_V2', 'explicit-policy DAG V2 required; historical V1 is not relabelled')
    require(dag.get('execution_policy')==policy, 'identity/policy scope mismatch')
    require(dag.get('science_sha256')==canonical_hash({k:v for k,v in dag.items() if k!='science_sha256'}), 'DAG content seal mismatch')
    require(type(required_inputs) is dict and required_inputs and set(input_bytes)==set(required_inputs), 'exact lower input file set')
    receipts=[]
    for name, ref in sorted(required_inputs.items()):
        raw=input_bytes[name]
        require(type(raw) is bytes and type(ref) is dict and set(ref)=={'sha256','bytes'}, 'lower input byte/reference fields')
        require(type(ref['bytes']) is int and len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'], 'lower input digest mismatch')
        receipts.append({'path':name,'sha256':ref['sha256'],'bytes':ref['bytes']})
    require(binding['input_root_ref']==canonical_hash(receipts), 'input root binding mismatch')
    ids=tuple(authority_source_ids)
    require(ids and all(type(x) is str and x for x in ids) and len(set(ids))==len(ids), 'authority source IDs')
    nodes,roots=dag.get('nodes'),dag.get('roots')
    require(type(nodes) is dict and type(roots) is list and roots, 'DAG roots/nodes')
    require(all(type(k) is str and k for k in nodes) and all(type(r) is str and r in nodes for r in roots), 'unknown root')
    base_fields={'kind','construction_digest','execution_policy','execution_order_key','lane'}
    for key,row in nodes.items():
        require(type(row) is dict and base_fields <= set(row), 'complete node header')
        require(row['construction_digest']==key and row['execution_policy']==policy and type(row['execution_order_key']) is str, 'node scope/alias/order mismatch')
        if row['kind']=='O7':
            require({'source_id','reserve_counts'} <= set(row) and row['source_id'] in ids, 'unresolved O7 authority leaf')
            require(len(row['reserve_counts'])==7 and all(type(x) is int and x>=0 for x in row['reserve_counts']), 'O7 overlay counters')
        else:
            require(row['kind']=='LIFT' and {'level','children','top_edges_full','motif_id'} <= set(row), 'complete lift fields')
            require(type(row['level']) is int and type(row['children']) is list and row['children'], 'lift level/children')
            require(all(type(ch) is str and ch in nodes for ch in row['children']), 'unresolved child')
    # Establish acyclicity before resolving any witness path.
    active,visited=set(),set()
    def walk(key):
        if key in visited:return
        require(key not in active, 'cycle in carrier dependency graph')
        active.add(key)
        for ch in nodes[key].get('children',[]):walk(ch)
        active.remove(key);visited.add(key)
    for root in roots:walk(root)
    require(visited==set(nodes), 'unbound extra DAG nodes')
    def witness(key,w,endpoint):
        row=nodes[key]
        if row['kind']=='O7':
            require(type(w) in (list,tuple) and len(w)==2 and all(type(x) is int for x in w), 'leaf reservation witness')
            require(w[0]==endpoint and 0 <= w[1] < row['reserve_counts'][endpoint], 'leaf reservation ordinal/type')
            return
        require(type(w) in (list,tuple) and len(w)==3 and type(w[0]) is int and type(w[2]) is int, 'recursive owner witness')
        require(w[2]==endpoint and 0 <= w[0] < len(row['children']), 'recursive owner slot/type')
        witness(row['children'][w[0]],w[1],endpoint)
    slots=[]
    for key,row in nodes.items():
        if row['kind']!='LIFT':continue
        for slot,ch in enumerate(row['children']):slots.append({'parent_node_alias':key,'ordered_child_slot':slot,'child_node_alias':ch})
        require(type(row['top_edges_full']) is list, 'full edge list required')
        for edge in row['top_edges_full']:
            require(type(edge) in (list,tuple) and len(edge)==6, 'typed edge and two full witnesses required')
            u,v,a,b,wu,wv=edge
            require(all(type(x) is int for x in (u,v,a,b)), 'exact edge coordinates')
            require(u!=v and 0 <= u < len(row['children']) and 0 <= v < len(row['children']) and 0 <= a < 7 and 0 <= b < 7, 'typed edge slot/type range')
            witness(row['children'][u],wu,a);witness(row['children'][v],wv,b)
    return exact_json({'schema_id':'IG_MASTER_G3_FULL_DAG_EXPORT_CANDIDATE_V1','binding':binding,
        'full_native_DAG':dag,'lower_input_receipts':receipts,'authority_source_ids':ids,
        'root_occurrences':[{'root_ordinal':i,'node_alias':r} for i,r in enumerate(roots)],
        'ordered_child_slot_table':slots,'logical_occurrence_rule':'ROOT_ORDINAL_PLUS_ORDERED_CHILD_SLOT_PATH',
        'shared_nodes_are_not_occurrence_deduplication':True,'scientific_acceptance':False})
